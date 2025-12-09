from pydub.utils import mediainfo
import hashlib
import json
import logging
import os
import uuid
import subprocess
import math
import time
import psutil
import html
import base64
from functools import lru_cache
from pydub import AudioSegment
from pydub.silence import split_on_silence
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fnmatch import fnmatch
import aiohttp
import aiofiles
import requests
import mimetypes

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.headers import include_user_info_headers
from open_webui.config import (
    WHISPER_MODEL_AUTO_UPDATE,
    WHISPER_MODEL_DIR,
    CACHE_DIR,
    WHISPER_LANGUAGE,
    ELEVENLABS_API_BASE_URL,
)

from open_webui.constants import ERROR_MESSAGES
from open_webui.env import (
    ENV,
    AIOHTTP_CLIENT_SESSION_SSL,
    AIOHTTP_CLIENT_TIMEOUT,
    SRC_LOG_LEVELS,
    DEVICE_TYPE,
    ENABLE_FORWARD_USER_INFO_HEADERS,
)


router = APIRouter()

# Constants
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes
AZURE_MAX_FILE_SIZE_MB = 200
AZURE_MAX_FILE_SIZE = AZURE_MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["AUDIO"])

SPEECH_CACHE_DIR = CACHE_DIR / "audio" / "speech"
SPEECH_CACHE_DIR.mkdir(parents=True, exist_ok=True)


##########################################
#
# Utility functions
#
##########################################


def is_audio_conversion_required(file_path):
    """
    Check if the given audio file needs conversion to mp3.
    """
    SUPPORTED_FORMATS = {"flac", "m4a", "mp3", "mp4", "mpeg", "wav", "webm"}

    if not os.path.isfile(file_path):
        log.error(f"File not found: {file_path}")
        return False

    try:
        info = mediainfo(file_path)
        codec_name = info.get("codec_name", "").lower()
        codec_type = info.get("codec_type", "").lower()
        codec_tag_string = info.get("codec_tag_string", "").lower()

        if codec_name == "aac" and codec_type == "audio" and codec_tag_string == "mp4a":
            # File is AAC/mp4a audio, recommend mp3 conversion
            return True

        # If the codec name is in the supported formats
        if codec_name in SUPPORTED_FORMATS:
            return False

        return True
    except Exception as e:
        log.error(f"Error getting audio format: {e}")
        return False


def convert_audio_to_mp3(file_path):
    """Convert audio file to mp3 format using ffmpeg (no memory loading)."""
    start_time = time.time()
    start_memory = get_memory_usage()

    try:
        output_path = os.path.splitext(file_path)[0] + ".mp3"
        original_size = os.path.getsize(file_path)

        log.info(
            f"[CONVERT] Starting conversion of {original_size/(1024*1024):.1f}MB file")
        log.info(f"[MEMORY] Initial: RSS={start_memory['rss']:.1f}MB")

        cmd = [
            'ffmpeg',
            '-i', file_path,
            '-acodec', 'mp3',
            '-y',  # Overwrite output
            output_path
        ]

        ffmpeg_start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        ffmpeg_duration = time.time() - ffmpeg_start

        if result.returncode == 0:
            output_size = os.path.getsize(output_path)
            log_performance("Conversion", start_time, start_memory)

            log.info(
                f"[CONVERT] Success: {original_size/(1024*1024):.1f}MB → {output_size/(1024*1024):.1f}MB")
            log.info(f"[CONVERT] FFmpeg time: {ffmpeg_duration:.2f}s")
            return output_path
        else:
            log.error(f"ffmpeg conversion failed: {result.stderr}")
            return None

    except Exception as e:
        log.error(f"Error converting audio file: {e}")
        return None


def set_faster_whisper_model(model: str, auto_update: bool = False):
    whisper_model = None
    if model:
        from faster_whisper import WhisperModel

        faster_whisper_kwargs = {
            "model_size_or_path": model,
            "device": DEVICE_TYPE if DEVICE_TYPE and DEVICE_TYPE == "cuda" else "cpu",
            "compute_type": "int8",
            "download_root": WHISPER_MODEL_DIR,
            "local_files_only": not auto_update,
        }

        try:
            whisper_model = WhisperModel(**faster_whisper_kwargs)
        except Exception:
            log.warning(
                "WhisperModel initialization failed, attempting download with local_files_only=False"
            )
            faster_whisper_kwargs["local_files_only"] = False
            whisper_model = WhisperModel(**faster_whisper_kwargs)
    return whisper_model


##########################################
#
# Audio API
#
##########################################


class TTSConfigForm(BaseModel):
    OPENAI_API_BASE_URL: str
    OPENAI_API_KEY: str
    OPENAI_PARAMS: Optional[dict] = None
    API_KEY: str
    ENGINE: str
    MODEL: str
    VOICE: str
    SPLIT_ON: str
    AZURE_SPEECH_REGION: str
    AZURE_SPEECH_BASE_URL: str
    AZURE_SPEECH_OUTPUT_FORMAT: str


class STTConfigForm(BaseModel):
    OPENAI_API_BASE_URL: str
    OPENAI_API_KEY: str
    ENGINE: str
    MODEL: str
    SUPPORTED_CONTENT_TYPES: list[str] = []
    WHISPER_MODEL: str
    DEEPGRAM_API_KEY: str
    AZURE_API_KEY: str
    AZURE_REGION: str
    AZURE_LOCALES: str
    AZURE_BASE_URL: str
    AZURE_MAX_SPEAKERS: str
    MISTRAL_API_KEY: str
    MISTRAL_API_BASE_URL: str
    MISTRAL_USE_CHAT_COMPLETIONS: bool


class AudioConfigUpdateForm(BaseModel):
    tts: TTSConfigForm
    stt: STTConfigForm


@router.get("/config")
async def get_audio_config(request: Request, user=Depends(get_admin_user)):
    return {
        "tts": {
            "OPENAI_API_BASE_URL": request.app.state.config.TTS_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.TTS_OPENAI_API_KEY,
            "OPENAI_PARAMS": request.app.state.config.TTS_OPENAI_PARAMS,
            "API_KEY": request.app.state.config.TTS_API_KEY,
            "ENGINE": request.app.state.config.TTS_ENGINE,
            "MODEL": request.app.state.config.TTS_MODEL,
            "VOICE": request.app.state.config.TTS_VOICE,
            "SPLIT_ON": request.app.state.config.TTS_SPLIT_ON,
            "AZURE_SPEECH_REGION": request.app.state.config.TTS_AZURE_SPEECH_REGION,
            "AZURE_SPEECH_BASE_URL": request.app.state.config.TTS_AZURE_SPEECH_BASE_URL,
            "AZURE_SPEECH_OUTPUT_FORMAT": request.app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT,
        },
        "stt": {
            "OPENAI_API_BASE_URL": request.app.state.config.STT_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.STT_OPENAI_API_KEY,
            "ENGINE": request.app.state.config.STT_ENGINE,
            "MODEL": request.app.state.config.STT_MODEL,
            "SUPPORTED_CONTENT_TYPES": request.app.state.config.STT_SUPPORTED_CONTENT_TYPES,
            "WHISPER_MODEL": request.app.state.config.WHISPER_MODEL,
            "DEEPGRAM_API_KEY": request.app.state.config.DEEPGRAM_API_KEY,
            "AZURE_API_KEY": request.app.state.config.AUDIO_STT_AZURE_API_KEY,
            "AZURE_REGION": request.app.state.config.AUDIO_STT_AZURE_REGION,
            "AZURE_LOCALES": request.app.state.config.AUDIO_STT_AZURE_LOCALES,
            "AZURE_BASE_URL": request.app.state.config.AUDIO_STT_AZURE_BASE_URL,
            "AZURE_MAX_SPEAKERS": request.app.state.config.AUDIO_STT_AZURE_MAX_SPEAKERS,
            "MISTRAL_API_KEY": request.app.state.config.AUDIO_STT_MISTRAL_API_KEY,
            "MISTRAL_API_BASE_URL": request.app.state.config.AUDIO_STT_MISTRAL_API_BASE_URL,
            "MISTRAL_USE_CHAT_COMPLETIONS": request.app.state.config.AUDIO_STT_MISTRAL_USE_CHAT_COMPLETIONS,
        },
    }


@router.post("/config/update")
async def update_audio_config(
    request: Request, form_data: AudioConfigUpdateForm, user=Depends(get_admin_user)
):
    request.app.state.config.TTS_OPENAI_API_BASE_URL = form_data.tts.OPENAI_API_BASE_URL
    request.app.state.config.TTS_OPENAI_API_KEY = form_data.tts.OPENAI_API_KEY
    request.app.state.config.TTS_OPENAI_PARAMS = form_data.tts.OPENAI_PARAMS
    request.app.state.config.TTS_API_KEY = form_data.tts.API_KEY
    request.app.state.config.TTS_ENGINE = form_data.tts.ENGINE
    request.app.state.config.TTS_MODEL = form_data.tts.MODEL
    request.app.state.config.TTS_VOICE = form_data.tts.VOICE
    request.app.state.config.TTS_SPLIT_ON = form_data.tts.SPLIT_ON
    request.app.state.config.TTS_AZURE_SPEECH_REGION = form_data.tts.AZURE_SPEECH_REGION
    request.app.state.config.TTS_AZURE_SPEECH_BASE_URL = (
        form_data.tts.AZURE_SPEECH_BASE_URL
    )
    request.app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT = (
        form_data.tts.AZURE_SPEECH_OUTPUT_FORMAT
    )

    request.app.state.config.STT_OPENAI_API_BASE_URL = form_data.stt.OPENAI_API_BASE_URL
    request.app.state.config.STT_OPENAI_API_KEY = form_data.stt.OPENAI_API_KEY
    request.app.state.config.STT_ENGINE = form_data.stt.ENGINE
    request.app.state.config.STT_MODEL = form_data.stt.MODEL
    request.app.state.config.STT_SUPPORTED_CONTENT_TYPES = (
        form_data.stt.SUPPORTED_CONTENT_TYPES
    )

    request.app.state.config.WHISPER_MODEL = form_data.stt.WHISPER_MODEL
    request.app.state.config.DEEPGRAM_API_KEY = form_data.stt.DEEPGRAM_API_KEY
    request.app.state.config.AUDIO_STT_AZURE_API_KEY = form_data.stt.AZURE_API_KEY
    request.app.state.config.AUDIO_STT_AZURE_REGION = form_data.stt.AZURE_REGION
    request.app.state.config.AUDIO_STT_AZURE_LOCALES = form_data.stt.AZURE_LOCALES
    request.app.state.config.AUDIO_STT_AZURE_BASE_URL = form_data.stt.AZURE_BASE_URL
    request.app.state.config.AUDIO_STT_AZURE_MAX_SPEAKERS = (
        form_data.stt.AZURE_MAX_SPEAKERS
    )
    request.app.state.config.AUDIO_STT_MISTRAL_API_KEY = form_data.stt.MISTRAL_API_KEY
    request.app.state.config.AUDIO_STT_MISTRAL_API_BASE_URL = (
        form_data.stt.MISTRAL_API_BASE_URL
    )
    request.app.state.config.AUDIO_STT_MISTRAL_USE_CHAT_COMPLETIONS = (
        form_data.stt.MISTRAL_USE_CHAT_COMPLETIONS
    )

    if request.app.state.config.STT_ENGINE == "":
        request.app.state.faster_whisper_model = set_faster_whisper_model(
            form_data.stt.WHISPER_MODEL, WHISPER_MODEL_AUTO_UPDATE
        )
    else:
        request.app.state.faster_whisper_model = None

    return {
        "tts": {
            "ENGINE": request.app.state.config.TTS_ENGINE,
            "MODEL": request.app.state.config.TTS_MODEL,
            "VOICE": request.app.state.config.TTS_VOICE,
            "OPENAI_API_BASE_URL": request.app.state.config.TTS_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.TTS_OPENAI_API_KEY,
            "OPENAI_PARAMS": request.app.state.config.TTS_OPENAI_PARAMS,
            "API_KEY": request.app.state.config.TTS_API_KEY,
            "SPLIT_ON": request.app.state.config.TTS_SPLIT_ON,
            "AZURE_SPEECH_REGION": request.app.state.config.TTS_AZURE_SPEECH_REGION,
            "AZURE_SPEECH_BASE_URL": request.app.state.config.TTS_AZURE_SPEECH_BASE_URL,
            "AZURE_SPEECH_OUTPUT_FORMAT": request.app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT,
        },
        "stt": {
            "OPENAI_API_BASE_URL": request.app.state.config.STT_OPENAI_API_BASE_URL,
            "OPENAI_API_KEY": request.app.state.config.STT_OPENAI_API_KEY,
            "ENGINE": request.app.state.config.STT_ENGINE,
            "MODEL": request.app.state.config.STT_MODEL,
            "SUPPORTED_CONTENT_TYPES": request.app.state.config.STT_SUPPORTED_CONTENT_TYPES,
            "WHISPER_MODEL": request.app.state.config.WHISPER_MODEL,
            "DEEPGRAM_API_KEY": request.app.state.config.DEEPGRAM_API_KEY,
            "AZURE_API_KEY": request.app.state.config.AUDIO_STT_AZURE_API_KEY,
            "AZURE_REGION": request.app.state.config.AUDIO_STT_AZURE_REGION,
            "AZURE_LOCALES": request.app.state.config.AUDIO_STT_AZURE_LOCALES,
            "AZURE_BASE_URL": request.app.state.config.AUDIO_STT_AZURE_BASE_URL,
            "AZURE_MAX_SPEAKERS": request.app.state.config.AUDIO_STT_AZURE_MAX_SPEAKERS,
            "MISTRAL_API_KEY": request.app.state.config.AUDIO_STT_MISTRAL_API_KEY,
            "MISTRAL_API_BASE_URL": request.app.state.config.AUDIO_STT_MISTRAL_API_BASE_URL,
            "MISTRAL_USE_CHAT_COMPLETIONS": request.app.state.config.AUDIO_STT_MISTRAL_USE_CHAT_COMPLETIONS,
        },
    }


def load_speech_pipeline(request):
    from transformers import pipeline
    from datasets import load_dataset

    if request.app.state.speech_synthesiser is None:
        request.app.state.speech_synthesiser = pipeline(
            "text-to-speech", "microsoft/speecht5_tts"
        )

    if request.app.state.speech_speaker_embeddings_dataset is None:
        request.app.state.speech_speaker_embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )


@router.post("/speech")
async def speech(request: Request, user=Depends(get_verified_user)):
    body = await request.body()
    name = hashlib.sha256(
        body
        + str(request.app.state.config.TTS_ENGINE).encode("utf-8")
        + str(request.app.state.config.TTS_MODEL).encode("utf-8")
    ).hexdigest()

    file_path = SPEECH_CACHE_DIR.joinpath(f"{name}.mp3")
    file_body_path = SPEECH_CACHE_DIR.joinpath(f"{name}.json")

    # Check if the file already exists in the cache
    if file_path.is_file():
        return FileResponse(file_path)

    payload = None
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as e:
        log.exception(e)
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    r = None
    if request.app.state.config.TTS_ENGINE == "openai":
        payload["model"] = request.app.state.config.TTS_MODEL

        try:
            timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True
            ) as session:
                payload = {
                    **payload,
                    **(request.app.state.config.TTS_OPENAI_PARAMS or {}),
                }

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {request.app.state.config.TTS_OPENAI_API_KEY}",
                }
                if ENABLE_FORWARD_USER_INFO_HEADERS:
                    headers = include_user_info_headers(headers, user)

                r = await session.post(
                    url=f"{request.app.state.config.TTS_OPENAI_API_BASE_URL}/audio/speech",
                    json=payload,
                    headers=headers,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                )

                r.raise_for_status()

                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(await r.read())

                async with aiofiles.open(file_body_path, "w") as f:
                    await f.write(json.dumps(payload))

            return FileResponse(file_path)

        except Exception as e:
            log.exception(e)
            detail = None

            status_code = 500
            detail = f"Open WebUI: Server Connection Error"

            if r is not None:
                status_code = r.status

                try:
                    res = await r.json()
                    if "error" in res:
                        detail = f"External: {res['error']}"
                except Exception:
                    detail = f"External: {e}"

            raise HTTPException(
                status_code=status_code,
                detail=detail,
            )

    elif request.app.state.config.TTS_ENGINE == "elevenlabs":
        voice_id = payload.get("voice", "")

        if voice_id not in get_available_voices(request):
            raise HTTPException(
                status_code=400,
                detail="Invalid voice id",
            )

        try:
            timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True
            ) as session:
                async with session.post(
                    f"{ELEVENLABS_API_BASE_URL}/v1/text-to-speech/{voice_id}",
                    json={
                        "text": payload["input"],
                        "model_id": request.app.state.config.TTS_MODEL,
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
                    },
                    headers={
                        "Accept": "audio/mpeg",
                        "Content-Type": "application/json",
                        "xi-api-key": request.app.state.config.TTS_API_KEY,
                    },
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    r.raise_for_status()

                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(await r.read())

                    async with aiofiles.open(file_body_path, "w") as f:
                        await f.write(json.dumps(payload))

            return FileResponse(file_path)

        except Exception as e:
            log.exception(e)
            detail = None

            try:
                if r.status != 200:
                    res = await r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
            except Exception:
                detail = f"External: {e}"

            raise HTTPException(
                status_code=getattr(r, "status", 500) if r else 500,
                detail=detail if detail else "Open WebUI: Server Connection Error",
            )

    elif request.app.state.config.TTS_ENGINE == "azure":
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            log.exception(e)
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        region = request.app.state.config.TTS_AZURE_SPEECH_REGION or "eastus"
        base_url = request.app.state.config.TTS_AZURE_SPEECH_BASE_URL
        language = request.app.state.config.TTS_VOICE
        locale = "-".join(request.app.state.config.TTS_VOICE.split("-")[:1])
        output_format = request.app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT

        try:
            data = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{locale}">
                <voice name="{language}">{html.escape(payload["input"])}</voice>
            </speak>"""
            timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
            async with aiohttp.ClientSession(
                timeout=timeout, trust_env=True
            ) as session:
                async with session.post(
                    (base_url or f"https://{region}.tts.speech.microsoft.com")
                    + "/cognitiveservices/v1",
                    headers={
                        "Ocp-Apim-Subscription-Key": request.app.state.config.TTS_API_KEY,
                        "Content-Type": "application/ssml+xml",
                        "X-Microsoft-OutputFormat": output_format,
                    },
                    data=data,
                    ssl=AIOHTTP_CLIENT_SESSION_SSL,
                ) as r:
                    r.raise_for_status()

                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(await r.read())

                    async with aiofiles.open(file_body_path, "w") as f:
                        await f.write(json.dumps(payload))

                    return FileResponse(file_path)

        except Exception as e:
            log.exception(e)
            detail = None

            try:
                if r.status != 200:
                    res = await r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
            except Exception:
                detail = f"External: {e}"

            raise HTTPException(
                status_code=getattr(r, "status", 500) if r else 500,
                detail=detail if detail else "Open WebUI: Server Connection Error",
            )

    elif request.app.state.config.TTS_ENGINE == "transformers":
        payload = None
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            log.exception(e)
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        import torch
        import soundfile as sf

        load_speech_pipeline(request)

        embeddings_dataset = request.app.state.speech_speaker_embeddings_dataset

        speaker_index = 6799
        try:
            speaker_index = embeddings_dataset["filename"].index(
                request.app.state.config.TTS_MODEL
            )
        except Exception:
            pass

        speaker_embedding = torch.tensor(
            embeddings_dataset[speaker_index]["xvector"]
        ).unsqueeze(0)

        speech = request.app.state.speech_synthesiser(
            payload["input"],
            forward_params={"speaker_embeddings": speaker_embedding},
        )

        sf.write(file_path, speech["audio"], samplerate=speech["sampling_rate"])

        async with aiofiles.open(file_body_path, "w") as f:
            await f.write(json.dumps(payload))

        return FileResponse(file_path)


def transcription_handler(request, file_path, metadata, user=None):
    filename = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    id = filename.split(".")[0]

    metadata = metadata or {}

    languages = [
        metadata.get("language", None) if not WHISPER_LANGUAGE else WHISPER_LANGUAGE,
        None,  # Always fallback to None in case transcription fails
    ]

    if request.app.state.config.STT_ENGINE == "":
        if request.app.state.faster_whisper_model is None:
            request.app.state.faster_whisper_model = set_faster_whisper_model(
                request.app.state.config.WHISPER_MODEL
            )

        model = request.app.state.faster_whisper_model
        segments, info = model.transcribe(
            file_path,
            beam_size=5,
            vad_filter=request.app.state.config.WHISPER_VAD_FILTER,
            language=languages[0],
        )
        log.info(
            "Detected language '%s' with probability %f"
            % (info.language, info.language_probability)
        )

        transcript = "".join([segment.text for segment in list(segments)])
        data = {"text": transcript.strip()}

        # save the transcript to a json file
        transcript_file = f"{file_dir}/{id}.json"
        with open(transcript_file, "w") as f:
            json.dump(data, f)

        log.debug(data)
        return data
    elif request.app.state.config.STT_ENGINE == "openai":
        r = None
        try:
            for language in languages:
                payload = {
                    "model": request.app.state.config.STT_MODEL,
                }

                if language:
                    payload["language"] = language

                headers = {
                    "Authorization": f"Bearer {request.app.state.config.STT_OPENAI_API_KEY}"
                }
                if user and ENABLE_FORWARD_USER_INFO_HEADERS:
                    headers = include_user_info_headers(headers, user)

                r = requests.post(
                    url=f"{request.app.state.config.STT_OPENAI_API_BASE_URL}/audio/transcriptions",
                    headers=headers,
                    files={"file": (filename, open(file_path, "rb"))},
                    data=payload,
                )

                if r.status_code == 200:
                    # Successful transcription
                    break

            r.raise_for_status()
            data = r.json()

            # save the transcript to a json file
            transcript_file = f"{file_dir}/{id}.json"
            with open(transcript_file, "w") as f:
                json.dump(data, f)

            return data
        except Exception as e:
            log.exception(e)

            detail = None
            if r is not None:
                try:
                    res = r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
                except Exception:
                    detail = f"External: {e}"

            raise Exception(detail if detail else "Open WebUI: Server Connection Error")

    elif request.app.state.config.STT_ENGINE == "deepgram":
        try:
            # Determine the MIME type of the file
            mime, _ = mimetypes.guess_type(file_path)
            if not mime:
                mime = "audio/wav"  # fallback to wav if undetectable

            # Read the audio file
            with open(file_path, "rb") as f:
                file_data = f.read()

            # Build headers and parameters
            headers = {
                "Authorization": f"Token {request.app.state.config.DEEPGRAM_API_KEY}",
                "Content-Type": mime,
            }

            for language in languages:
                params = {}
                if request.app.state.config.STT_MODEL:
                    params["model"] = request.app.state.config.STT_MODEL

                if language:
                    params["language"] = language

                # Make request to Deepgram API
                r = requests.post(
                    "https://api.deepgram.com/v1/listen?smart_format=true",
                    headers=headers,
                    params=params,
                    data=file_data,
                )

                if r.status_code == 200:
                    # Successful transcription
                    break

            r.raise_for_status()
            response_data = r.json()

            # Extract transcript from Deepgram response
            try:
                transcript = response_data["results"]["channels"][0]["alternatives"][
                    0
                ].get("transcript", "")
            except (KeyError, IndexError) as e:
                log.error(f"Malformed response from Deepgram: {str(e)}")
                raise Exception(
                    "Failed to parse Deepgram response - unexpected response format"
                )
            data = {"text": transcript.strip()}

            # Save transcript
            transcript_file = f"{file_dir}/{id}.json"
            with open(transcript_file, "w") as f:
                json.dump(data, f)

            return data

        except Exception as e:
            log.exception(e)
            detail = None
            if r is not None:
                try:
                    res = r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
                except Exception:
                    detail = f"External: {e}"
            raise Exception(detail if detail else "Open WebUI: Server Connection Error")

    elif request.app.state.config.STT_ENGINE == "azure":
        # Check file exists and size
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="Audio file not found")

        # Check file size (Azure has a larger limit of 200MB)
        file_size = os.path.getsize(file_path)
        if file_size > AZURE_MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds Azure's limit of {AZURE_MAX_FILE_SIZE_MB}MB",
            )

        api_key = request.app.state.config.AUDIO_STT_AZURE_API_KEY
        region = request.app.state.config.AUDIO_STT_AZURE_REGION or "eastus"
        locales = request.app.state.config.AUDIO_STT_AZURE_LOCALES
        base_url = request.app.state.config.AUDIO_STT_AZURE_BASE_URL
        max_speakers = request.app.state.config.AUDIO_STT_AZURE_MAX_SPEAKERS or 3

        # IF NO LOCALES, USE DEFAULTS
        if len(locales) < 2:
            locales = [
                "en-US",
                "es-ES",
                "es-MX",
                "fr-FR",
                "hi-IN",
                "it-IT",
                "de-DE",
                "en-GB",
                "en-IN",
                "ja-JP",
                "ko-KR",
                "pt-BR",
                "zh-CN",
            ]
            locales = ",".join(locales)

        if not api_key or not region:
            raise HTTPException(
                status_code=400,
                detail="Azure API key is required for Azure STT",
            )

        r = None
        try:
            # Prepare the request
            data = {
                "definition": json.dumps(
                    {
                        "locales": locales.split(","),
                        "diarization": {"maxSpeakers": max_speakers, "enabled": True},
                    }
                    if locales
                    else {}
                )
            }

            url = (
                base_url or f"https://{region}.api.cognitive.microsoft.com"
            ) + "/speechtotext/transcriptions:transcribe?api-version=2024-11-15"

            # Use context manager to ensure file is properly closed
            with open(file_path, "rb") as audio_file:
                r = requests.post(
                    url=url,
                    files={"audio": audio_file},
                    data=data,
                    headers={
                        "Ocp-Apim-Subscription-Key": api_key,
                    },
                )

            r.raise_for_status()
            response = r.json()

            # Extract transcript from response
            if not response.get("combinedPhrases"):
                raise ValueError("No transcription found in response")

            # Get the full transcript from combinedPhrases
            transcript = response["combinedPhrases"][0].get("text", "").strip()
            if not transcript:
                raise ValueError("Empty transcript in response")

            data = {"text": transcript}

            # Save transcript to json file (consistent with other providers)
            transcript_file = f"{file_dir}/{id}.json"
            with open(transcript_file, "w") as f:
                json.dump(data, f)

            log.debug(data)
            return data

        except (KeyError, IndexError, ValueError) as e:
            log.exception("Error parsing Azure response")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse Azure response: {str(e)}",
            )
        except requests.exceptions.RequestException as e:
            log.exception(e)
            detail = None

            try:
                if r is not None and r.status_code != 200:
                    res = r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
            except Exception:
                detail = f"External: {e}"

            raise HTTPException(
                status_code=getattr(r, "status_code", 500) if r else 500,
                detail=detail if detail else "Open WebUI: Server Connection Error",
            )

    elif request.app.state.config.STT_ENGINE == "mistral":
        # Check file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="Audio file not found")

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds limit of {MAX_FILE_SIZE_MB}MB",
            )

        api_key = request.app.state.config.AUDIO_STT_MISTRAL_API_KEY
        api_base_url = (
            request.app.state.config.AUDIO_STT_MISTRAL_API_BASE_URL
            or "https://api.mistral.ai/v1"
        )
        use_chat_completions = (
            request.app.state.config.AUDIO_STT_MISTRAL_USE_CHAT_COMPLETIONS
        )

        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="Mistral API key is required for Mistral STT",
            )

        r = None
        try:
            # Use voxtral-mini-latest as the default model for transcription
            model = request.app.state.config.STT_MODEL or "voxtral-mini-latest"

            log.info(
                f"Mistral STT - model: {model}, "
                f"method: {'chat_completions' if use_chat_completions else 'transcriptions'}"
            )

            if use_chat_completions:
                # Use chat completions API with audio input
                # This method requires mp3 or wav format
                audio_file_to_use = file_path

                if is_audio_conversion_required(file_path):
                    log.debug("Converting audio to mp3 for chat completions API")
                    converted_path = convert_audio_to_mp3(file_path)
                    if converted_path:
                        audio_file_to_use = converted_path
                    else:
                        log.error("Audio conversion failed")
                        raise HTTPException(
                            status_code=500,
                            detail="Audio conversion failed. Chat completions API requires mp3 or wav format.",
                        )

                # Read and encode audio file as base64
                with open(audio_file_to_use, "rb") as audio_file:
                    audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")

                # Prepare chat completions request
                url = f"{api_base_url}/chat/completions"

                # Add language instruction if specified
                language = metadata.get("language", None) if metadata else None
                if language:
                    text_instruction = f"Transcribe this audio exactly as spoken in {language}. Do not translate it."
                else:
                    text_instruction = "Transcribe this audio exactly as spoken in its original language. Do not translate it to another language."

                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": audio_base64,
                                },
                                {"type": "text", "text": text_instruction},
                            ],
                        }
                    ],
                }

                r = requests.post(
                    url=url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )

                r.raise_for_status()
                response = r.json()

                # Extract transcript from chat completion response
                transcript = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if not transcript:
                    raise ValueError("Empty transcript in response")

                data = {"text": transcript}

            else:
                # Use dedicated transcriptions API
                url = f"{api_base_url}/audio/transcriptions"

                # Determine the MIME type
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    mime_type = "audio/webm"

                # Use context manager to ensure file is properly closed
                with open(file_path, "rb") as audio_file:
                    files = {"file": (filename, audio_file, mime_type)}
                    data_form = {"model": model}

                    # Add language if specified in metadata
                    language = metadata.get("language", None) if metadata else None
                    if language:
                        data_form["language"] = language

                    r = requests.post(
                        url=url,
                        files=files,
                        data=data_form,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                        },
                    )

                r.raise_for_status()
                response = r.json()

                # Extract transcript from response
                transcript = response.get("text", "").strip()
                if not transcript:
                    raise ValueError("Empty transcript in response")

                data = {"text": transcript}

            # Save transcript to json file (consistent with other providers)
            transcript_file = f"{file_dir}/{id}.json"
            with open(transcript_file, "w") as f:
                json.dump(data, f)

            log.debug(data)
            return data

        except ValueError as e:
            log.exception("Error parsing Mistral response")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse Mistral response: {str(e)}",
            )
        except requests.exceptions.RequestException as e:
            log.exception(e)
            detail = None

            try:
                if r is not None and r.status_code != 200:
                    res = r.json()
                    if "error" in res:
                        detail = f"External: {res['error'].get('message', '')}"
                    else:
                        detail = f"External: {r.text}"
            except Exception:
                detail = f"External: {e}"

            raise HTTPException(
                status_code=getattr(r, "status_code", 500) if r else 500,
                detail=detail if detail else "Open WebUI: Server Connection Error",
            )


def transcribe(
    request: Request, file_path: str, metadata: Optional[dict] = None, user=None
):
    log.info(f"transcribe: {file_path} {metadata}")

    # Track overall time and memory
    start_time = time.time()
    start_memory = get_memory_usage()
    original_size = os.path.getsize(file_path)
    
    # Track all files created during processing
    original_file = file_path
    files_to_cleanup = set()  # Use set to avoid duplicates
    
    try:
        # Step 1: Convert if needed
        if is_audio_conversion_required(file_path):
            conversion_start = time.time()
            new_file = convert_audio_to_mp3(file_path)
            files_to_cleanup.add(file_path)  # Add old file to cleanup
            file_path = new_file
            log.info(f"[TIMING] Conversion took {time.time() - conversion_start:.2f}s")

        # Step 2: Compress
        compression_start = time.time()
        compressed_file = compress_audio(file_path)
        if compressed_file != file_path:
            files_to_cleanup.add(file_path)  # Add uncompressed file to cleanup
        file_path = compressed_file
        log.info(f"[TIMING] Compression took {time.time() - compression_start:.2f}s")

        # Step 3: Split into chunks
        split_start = time.time()
        chunk_paths = split_audio(file_path)
        log.info(f"[TIMING] Splitting took {time.time() - split_start:.2f}s, created {len(chunk_paths)} chunks")
        
        # Add compressed file to cleanup if it's not one of the chunks
        if file_path not in chunk_paths:
            files_to_cleanup.add(file_path)

        # Step 4: Transcribe chunks in parallel
        transcription_start = time.time()
        results = []
        
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(transcription_handler, request, chunk_path, metadata)
                for chunk_path in chunk_paths
            ]
            
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Error transcribing chunk: {e}",
                    )

        log.info(f"[TIMING] Transcription took {time.time() - transcription_start:.2f}s")
        
        # Combine results
        combined_text = " ".join([result["text"] for result in results])
        
        # Success - clean up everything including original
        cleanup_start = time.time()
        files_to_cleanup.update(chunk_paths)  # Add all chunks
        files_to_cleanup.add(original_file)    # Add original file
        
        cleanup_count = 0
        for file_to_remove in files_to_cleanup:
            if os.path.isfile(file_to_remove):
                try:
                    os.remove(file_to_remove)
                    cleanup_count += 1
                except Exception as e:
                    log.warning(f"Failed to remove {file_to_remove}: {e}")
        
        log.info(f"[TIMING] Cleanup took {time.time() - cleanup_start:.2f}s, removed {cleanup_count} files")
        
        # Log final metrics
        total_time = time.time() - start_time
        end_memory = get_memory_usage()
        memory_delta = end_memory['rss'] - start_memory['rss']

        log.info(f"[METRICS] Total time: {total_time:.2f}s")
        log.info(f"[METRICS] Original size: {original_size/(1024*1024):.1f}MB")
        log.info(f"[METRICS] Processing speed: {original_size/(1024*1024)/total_time:.2f} MB/s")
        log.info(f"[METRICS] Memory used: {memory_delta:+.1f}MB")
        log.info(f"[METRICS] Words transcribed: {len(combined_text.split())}")

        return {"text": combined_text}
        
    except Exception as e:
        # On error, only clean up temporary files (not original)
        cleanup_count = 0
        for file_to_remove in files_to_cleanup:
            if os.path.isfile(file_to_remove):
                try:
                    os.remove(file_to_remove)
                    cleanup_count += 1
                except Exception:
                    pass
                                   
        # Also clean up chunks if they exist
        if 'chunk_paths' in locals():
            for chunk in chunk_paths:
                if os.path.isfile(chunk):
                    try:
                        os.remove(chunk)
                        cleanup_count += 1
                    except Exception:
                        pass
        
        log.info(f"[ERROR CLEANUP] Removed {cleanup_count} temporary files, kept original")
        
        # Re-raise the original exception
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


def compress_audio(file_path):
    """
    Compress audio file to reduce size, regardless of current size.
    Always compresses to 32kbps mono for consistency.
    Uses ffmpeg to avoid loading file into memory.
    """
    start_time = time.time()
    start_memory = get_memory_usage()

    id = os.path.splitext(os.path.basename(file_path))[0]
    file_dir = os.path.dirname(file_path)
    compressed_path = os.path.join(file_dir, f"{id}_compressed.mp3")

    # Check if already compressed (to avoid double compression)
    if "_compressed" in id:
        return file_path

    try:
        # Get original file size for logging
        original_size = os.path.getsize(file_path)
        log.info(
            f"[COMPRESS] Starting compression of {original_size/(1024*1024):.1f}MB file")
        log.info(
            f"[MEMORY] Initial: RSS={start_memory['rss']:.1f}MB, VMS={start_memory['vms']:.1f}MB")

        # Direct ffmpeg conversion without loading into memory
        cmd = [
            'ffmpeg',
            '-i', file_path,
            '-ar', '16000',     # Sample rate 16kHz
            '-ac', '1',         # Mono channel
            '-b:a', '32k',      # Bitrate 32kbps
            '-y',               # Overwrite output
            '-loglevel', 'info',
            compressed_path
        ]

        ffmpeg_start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        ffmpeg_duration = time.time() - ffmpeg_start

        if result.returncode == 0:
            compressed_size = os.path.getsize(compressed_path)
            compression_ratio = original_size / compressed_size

            log_performance("Compression", start_time, start_memory)

            log.info(
                f"[COMPRESS] Success: {original_size/(1024*1024):.1f}MB → {compressed_size/(1024*1024):.1f}MB")
            log.info(
                f"[COMPRESS] Ratio: {compression_ratio:.1f}:1, FFmpeg time: {ffmpeg_duration:.2f}s")

            return compressed_path
        else:
            log.error(f"ffmpeg compression failed: {result.stderr}")
            return file_path

    except FileNotFoundError:
        log.error("ffmpeg not found. Please install ffmpeg for audio compression.")
        return file_path
    except Exception as e:
        log.error(f"Error in compress_audio: {e}")
        return file_path


def split_audio(file_path, max_bytes=MAX_FILE_SIZE, format="mp3", bitrate="32k"):
    """
    Split audio into chunks not exceeding max_bytes (20MB).
    Returns only valid chunks (size > 0 and duration > 1s).
    """
    start_time = time.time()
    start_memory = get_memory_usage()

    file_size = os.path.getsize(file_path)
    log.info(f"[SPLIT] Starting split of {file_size/(1024*1024):.1f}MB file")
    log.info(f"[MEMORY] Initial: RSS={start_memory['rss']:.1f}MB, VMS={start_memory['vms']:.1f}MB")

    if file_size <= max_bytes:
        log.info(f"[SPLIT] File size {file_size/(1024*1024):.1f}MB <= {max_bytes/(1024*1024):.1f}MB limit, no split needed")
        return [file_path]

    base, _ = os.path.splitext(file_path)

    # Try segment muxer first
    chunks = split_audio_segment_muxer(file_path, base, max_bytes, format, bitrate)
    # Fallback to manual chunking if segment muxer fails
    if not chunks:
        log.info("[SPLIT] Segment muxer failed, trying manual chunking")
        chunks = split_audio_manual(file_path, base, max_bytes, format, bitrate)

    # Filter out invalid chunks
    valid_chunks = []
    for chunk in chunks:
        if os.path.exists(chunk) and os.path.getsize(chunk) > 0:
            duration = get_audio_duration_ffmpeg(chunk)
            if duration > 1.0:
                valid_chunks.append(chunk)
            else:
                os.remove(chunk)
        elif os.path.exists(chunk):
            os.remove(chunk)
    log.info(f"[SPLIT] Returning {len(valid_chunks)} valid chunks")
    log_performance("Split", start_time, start_memory)
    return valid_chunks


def split_audio_segment_muxer(file_path, base_name, max_bytes, format="mp3", bitrate="32k"):
    """
    Use ffmpeg's segment muxer for efficient splitting.
    This is the most efficient method as it processes the file in a single pass.
    """
    segment_start = time.time()
    segment_memory = get_memory_usage()

    try:
        # Get actual duration of the compressed file
        duration = get_audio_duration_ffmpeg(file_path)
        if duration == 0:
            return []

        # Calculate segment time based on bitrate and max size (20MB)
        bitrate_numeric = int(bitrate.rstrip('k')) * 1000  # 32k = 32000 bits/second

        # Maximum seconds for 20MB at given bitrate with 90% safety margin
        max_seconds = (max_bytes * 8 * 0.9) / bitrate_numeric

        # Use actual file size to determine if we need conservative chunking
        file_size = os.path.getsize(file_path)
        if file_size < max_bytes * 2:
            # If file is less than 40MB, use larger chunks
            segment_seconds = min(max_seconds, duration / 2)
        else:
            # For larger files, use optimal chunk size
            segment_seconds = max_seconds

        expected_chunks = math.ceil(duration / segment_seconds)
        log.info(
            f"[SEGMENT] File: {file_size/(1024*1024):.1f}MB, Duration: {duration:.1f}s")
        log.info(
            f"[SEGMENT] Chunk duration: {segment_seconds:.1f}s, Expected chunks: {expected_chunks}")

        segment_pattern = f"{base_name}_chunk_%03d.{format}"

        cmd = [
            'ffmpeg',
            '-i', file_path,
            '-f', 'segment',
            '-segment_time', str(int(segment_seconds)),
            '-c', 'copy',  # Copy codec to avoid re-encoding
            '-reset_timestamps', '1',
            '-avoid_negative_ts', 'make_zero',
            segment_pattern
        ]

        ffmpeg_start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        ffmpeg_duration = time.time() - ffmpeg_start

        if result.returncode == 0:
            # Collect generated chunks
            chunks = []
            total_chunk_size = 0
            i = 0
            while True:
                chunk_path = f"{base_name}_chunk_{i:03d}.{format}"
                if os.path.exists(chunk_path):
                    chunk_size = os.path.getsize(chunk_path)
                    # Verify each chunk is under 20MB
                    if chunk_size <= max_bytes:
                        chunks.append(chunk_path)
                        total_chunk_size += chunk_size
                        log.debug(
                            f"[CHUNK] {i}: {chunk_size / (1024*1024):.2f}MB")
                    else:
                        # If any chunk exceeds 20MB, clean up and return empty
                        log.warning(
                            f"[CHUNK] {chunk_path} exceeds 20MB limit: {chunk_size / (1024*1024):.2f}MB")
                        for chunk in chunks:
                            if os.path.exists(chunk):
                                os.remove(chunk)
                        if os.path.exists(chunk_path):
                            os.remove(chunk_path)
                        return []
                    i += 1
                else:
                    break

            segment_duration = time.time() - segment_start
            memory_delta = get_memory_usage()['rss'] - segment_memory['rss']

            log.info(
                f"[SEGMENT] Created {len(chunks)} chunks, Total size: {total_chunk_size/(1024*1024):.1f}MB")
            log.info(
                f"[SEGMENT] FFmpeg time: {ffmpeg_duration:.2f}s, Total time: {segment_duration:.2f}s")
            log.info(f"[SEGMENT] Memory delta: {memory_delta:+.1f}MB")

            return chunks if chunks else []

    except Exception as e:
        log.debug(f"Segment muxer failed: {e}")
        return []


def split_audio_manual(file_path, base_name, max_bytes, format="mp3", bitrate="32k"):
    """
    Manual chunking for already compressed files.
    """
    chunks = []

    try:
        # Get duration
        duration = get_audio_duration_ffmpeg(file_path)
        if duration == 0:
            log.error("Could not determine audio duration")
            return [file_path]

        # Get actual file size
        file_size = os.path.getsize(file_path)
        log.info(
            f"Splitting compressed file: {file_size/(1024*1024):.1f}MB, duration: {duration:.1f}s")

        # Since file is already compressed at 32kbps, calculate based on actual file metrics
        bytes_per_second = file_size / duration
        seconds_per_chunk = (max_bytes * 0.9) / bytes_per_second  # 90% of 20MB

        # Calculate number of chunks needed
        num_chunks = math.ceil(duration / seconds_per_chunk)
        actual_chunk_duration = duration / num_chunks

        log.info(
            f"Creating {num_chunks} chunks of {actual_chunk_duration:.1f}s each")

        # Process chunks
        for i in range(num_chunks):
            start_time = i * actual_chunk_duration
            chunk_path = f"{base_name}_chunk_{i}.{format}"

            # Use fast seek and copy codec (no re-encoding)
            cmd = [
                'ffmpeg',
                '-ss', str(start_time),  # Fast seek
                '-i', file_path,
                '-t', str(actual_chunk_duration),
                '-c', 'copy',  # Copy codec - no re-encoding
                '-y',
                chunk_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                chunk_size = os.path.getsize(chunk_path)
                log.debug(
                    f"Created chunk {i}: {chunk_size / (1024*1024):.2f}MB")

                if chunk_size <= max_bytes:
                    chunks.append(chunk_path)
                else:
                    # This shouldn't happen with proper calculation, but handle it
                    os.remove(chunk_path)
                    log.error(
                        f"Chunk {i} unexpectedly large: {chunk_size / (1024*1024):.2f}MB")

                    # Use shorter duration
                    shorter_duration = actual_chunk_duration * \
                        (max_bytes / chunk_size) * 0.9
                    cmd[cmd.index('-t') + 1] = str(shorter_duration)

                    result = subprocess.run(
                        cmd, capture_output=True, text=True)
                    if result.returncode == 0 and os.path.getsize(chunk_path) <= max_bytes:
                        chunks.append(chunk_path)
            else:
                log.error(f"Failed to create chunk {i}: {result.stderr}")
                # Clean up and return original
                for chunk in chunks:
                    if os.path.exists(chunk):
                        os.remove(chunk)
                return [file_path]

        log.info(f"Successfully created {len(chunks)} chunks, all under 20MB")
        return chunks

    except Exception as e:
        log.error(f"Error in manual splitting: {e}")
        # Clean up
        for chunk in chunks:
            if os.path.exists(chunk):
                os.remove(chunk)
        return [file_path]


def get_memory_usage():
    """Get current memory usage of the process."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / (1024 * 1024),  # MB
            'vms': memory_info.vms / (1024 * 1024),  # MB
            'percent': process.memory_percent()
        }
    except:
        return {'rss': 0, 'vms': 0, 'percent': 0}


def log_performance(operation, start_time, start_memory):
    """Log performance metrics for an operation."""
    end_time = time.time()
    end_memory = get_memory_usage()

    duration = end_time - start_time
    memory_delta = end_memory['rss'] - start_memory['rss']

    log.info(f"[PERFORMANCE] {operation}:")
    log.info(f"  - Duration: {duration:.2f} seconds")
    log.info(
        f"  - Memory: {start_memory['rss']:.1f}MB → {end_memory['rss']:.1f}MB (Δ {memory_delta:+.1f}MB)")
    log.info(f"  - Memory %: {end_memory['percent']:.1f}%")


def get_audio_duration_ffmpeg(file_path):
    """Get audio duration using ffprobe without loading file into memory."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        log.error(f"ffprobe failed: {e}")

    return 0


@router.post("/transcriptions")
def transcription(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    user=Depends(get_verified_user),
):
    log.info(f"file.content_type: {file.content_type}")

    stt_supported_content_types = getattr(
        request.app.state.config, "STT_SUPPORTED_CONTENT_TYPES", []
    )

    if not any(
        fnmatch(file.content_type, content_type)
        for content_type in (
            stt_supported_content_types
            if stt_supported_content_types
            and any(t.strip() for t in stt_supported_content_types)
            else ["audio/*", "video/webm"]
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.FILE_NOT_SUPPORTED,
        )

    try:
        ext = file.filename.split(".")[-1]
        id = uuid.uuid4()

        filename = f"{id}.{ext}"
        contents = file.file.read()

        file_dir = f"{CACHE_DIR}/audio/transcriptions"
        os.makedirs(file_dir, exist_ok=True)
        file_path = f"{file_dir}/{filename}"

        with open(file_path, "wb") as f:
            f.write(contents)

        try:
            metadata = None

            if language:
                metadata = {"language": language}

            result = transcribe(request, file_path, metadata, user)

            return {
                **result,
                "filename": os.path.basename(file_path),
            }

        except Exception as e:
            log.exception(e)

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e),
            )

    except Exception as e:
        log.exception(e)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


def get_available_models(request: Request) -> list[dict]:
    available_models = []
    if request.app.state.config.TTS_ENGINE == "openai":
        # Use custom endpoint if not using the official OpenAI API URL
        if not request.app.state.config.TTS_OPENAI_API_BASE_URL.startswith(
            "https://api.openai.com"
        ):
            try:
                response = requests.get(
                    f"{request.app.state.config.TTS_OPENAI_API_BASE_URL}/audio/models"
                )
                response.raise_for_status()
                data = response.json()
                available_models = data.get("models", [])
            except Exception as e:
                log.error(f"Error fetching models from custom endpoint: {str(e)}")
                available_models = [{"id": "tts-1"}, {"id": "tts-1-hd"}]
        else:
            available_models = [{"id": "tts-1"}, {"id": "tts-1-hd"}]
    elif request.app.state.config.TTS_ENGINE == "elevenlabs":
        try:
            response = requests.get(
                f"{ELEVENLABS_API_BASE_URL}/v1/models",
                headers={
                    "xi-api-key": request.app.state.config.TTS_API_KEY,
                    "Content-Type": "application/json",
                },
                timeout=5,
            )
            response.raise_for_status()
            models = response.json()

            available_models = [
                {"name": model["name"], "id": model["model_id"]} for model in models
            ]
        except requests.RequestException as e:
            log.error(f"Error fetching voices: {str(e)}")
    return available_models


@router.get("/models")
async def get_models(request: Request, user=Depends(get_verified_user)):
    return {"models": get_available_models(request)}


def get_available_voices(request) -> dict:
    """Returns {voice_id: voice_name} dict"""
    available_voices = {}
    if request.app.state.config.TTS_ENGINE == "openai":
        # Use custom endpoint if not using the official OpenAI API URL
        if not request.app.state.config.TTS_OPENAI_API_BASE_URL.startswith(
            "https://api.openai.com"
        ):
            try:
                response = requests.get(
                    f"{request.app.state.config.TTS_OPENAI_API_BASE_URL}/audio/voices"
                )
                response.raise_for_status()
                data = response.json()
                voices_list = data.get("voices", [])
                available_voices = {voice["id"]: voice["name"] for voice in voices_list}
            except Exception as e:
                log.error(f"Error fetching voices from custom endpoint: {str(e)}")
                available_voices = {
                    "alloy": "alloy",
                    "echo": "echo",
                    "fable": "fable",
                    "onyx": "onyx",
                    "nova": "nova",
                    "shimmer": "shimmer",
                }
        else:
            available_voices = {
                "alloy": "alloy",
                "echo": "echo",
                "fable": "fable",
                "onyx": "onyx",
                "nova": "nova",
                "shimmer": "shimmer",
            }
    elif request.app.state.config.TTS_ENGINE == "elevenlabs":
        try:
            available_voices = get_elevenlabs_voices(
                api_key=request.app.state.config.TTS_API_KEY
            )
        except Exception:
            # Avoided @lru_cache with exception
            pass
    elif request.app.state.config.TTS_ENGINE == "azure":
        try:
            region = request.app.state.config.TTS_AZURE_SPEECH_REGION
            base_url = request.app.state.config.TTS_AZURE_SPEECH_BASE_URL
            url = (
                base_url or f"https://{region}.tts.speech.microsoft.com"
            ) + "/cognitiveservices/voices/list"
            headers = {
                "Ocp-Apim-Subscription-Key": request.app.state.config.TTS_API_KEY
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            voices = response.json()

            for voice in voices:
                available_voices[voice["ShortName"]] = (
                    f"{voice['DisplayName']} ({voice['ShortName']})"
                )
        except requests.RequestException as e:
            log.error(f"Error fetching voices: {str(e)}")

    return available_voices


@lru_cache
def get_elevenlabs_voices(api_key: str) -> dict:
    """
    Note, set the following in your .env file to use Elevenlabs:
    AUDIO_TTS_ENGINE=elevenlabs
    AUDIO_TTS_API_KEY=sk_...  # Your Elevenlabs API key
    AUDIO_TTS_VOICE=EXAVITQu4vr4xnSDxMaL  # From https://api.elevenlabs.io/v1/voices
    AUDIO_TTS_MODEL=eleven_multilingual_v2
    """

    try:
        # TODO: Add retries
        response = requests.get(
            f"{ELEVENLABS_API_BASE_URL}/v1/voices",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        voices_data = response.json()

        voices = {}
        for voice in voices_data.get("voices", []):
            voices[voice["voice_id"]] = voice["name"]
    except requests.RequestException as e:
        # Avoid @lru_cache with exception
        log.error(f"Error fetching voices: {str(e)}")
        raise RuntimeError(f"Error fetching voices: {str(e)}")

    return voices


@router.get("/voices")
async def get_voices(request: Request, user=Depends(get_verified_user)):
    return {
        "voices": [
            {"id": k, "name": v} for k, v in get_available_voices(request).items()
        ]
    }
