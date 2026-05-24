import base64
import concurrent.futures
import gc
import io
import logging
import math
import os
import sys
import threading
import time
import requests
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from open_webui.env import GLOBAL_LOG_LEVEL

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)

_PDFIUM_LOCK = threading.Lock()

class VisionLLMLoader:
    """
    Content extraction loader that uses a vision-capable LLM via OpenAI-compatible
    chat completions API (e.g. LiteLLM proxy) to extract text from documents.

    Processes one page per API call, with pages fanned out across a thread
    pool (size = max_workers). PDF rendering stays serial because pypdfium2
    is not safe across one PdfDocument from multiple threads.
    Auto-scales DPI to fit the model's context window.
    Continues on page errors to ensure the whole document is processed.
    """

    SUPPORTED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"}
    SUPPORTED_PDF_EXTENSIONS = {"pdf"}
    SUPPORTED_EXTENSIONS = SUPPORTED_PDF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS

    # Vision token constants (OpenAI-compatible high-detail model)
    TILE_SIZE = 512
    TOKENS_PER_TILE = 170
    BASE_TOKENS_PER_IMAGE = 85
    PROMPT_OVERHEAD_TOKENS = 500
    MIN_DPI = 100

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model: str,
        file_path: str,
        prompt: str = "",
        max_tokens: int = 8192,
        max_context_tokens: int = 32768,
        timeout: int = 120,
        max_retries: int = 2,
        image_dpi: int = 200,
        max_workers: Optional[int] = None,
    ):
        if not api_base_url:
            raise ValueError("API base URL is required for Vision LLM loader.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.file_path = file_path
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.image_dpi = image_dpi

        if max_workers is None:
            max_workers = int(os.environ.get("VISION_LLM_CONCURRENCY", "8"))
        self.max_workers = max(1, max_workers)

        self.file_name = os.path.basename(file_path)
        self.file_ext = self.file_name.rsplit(".", 1)[-1].lower() if "." in self.file_name else ""

    # ── Token estimation ──────────────────────────────────────────────

    @classmethod
    def estimate_image_tokens(cls, width: int, height: int) -> int:
        """Estimate input tokens for an image using OpenAI-style tile calculation."""
        tiles_x = math.ceil(width / cls.TILE_SIZE)
        tiles_y = math.ceil(height / cls.TILE_SIZE)
        return cls.BASE_TOKENS_PER_IMAGE + cls.TOKENS_PER_TILE * tiles_x * tiles_y

    @classmethod
    def estimate_page_tokens_at_dpi(cls, page_width_pt: float, page_height_pt: float, dpi: int) -> int:
        """Estimate input tokens for a PDF page rendered at a given DPI."""
        px_w = int(page_width_pt * dpi / 72.0)
        px_h = int(page_height_pt * dpi / 72.0)
        return cls.estimate_image_tokens(px_w, px_h)

    def _available_input_tokens(self) -> int:
        """Tokens available for images = context - prompt overhead - reserved output."""
        return max(
            self.BASE_TOKENS_PER_IMAGE,
            self.max_context_tokens - self.PROMPT_OVERHEAD_TOKENS - self.max_tokens,
        )

    # ── Image rendering ───────────────────────────────────────────────

    def _get_pdf_page_sizes(self) -> List[Tuple[float, float]]:
        """Return (width_pt, height_pt) for each page without rendering."""
        try:
            import pypdfium2 as pdfium
        except ImportError:
            raise ImportError(
                "pypdfium2 is required for PDF processing with Vision LLM. "
                "Install it with: pip install pypdfium2"
            )
        sizes = []
        with _PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(self.file_path)
            try:
                for i in range(len(pdf)):
                    page = pdf[i]
                    sizes.append((page.get_width(), page.get_height()))
                    page.close()  # free PDFium handle now, not at GC time
            finally:
                pdf.close()
                # Run any remaining pypdfium2 finalizers under the lock so they
                # can't fire later on another thread mid-render (segfault).
                gc.collect()
        return sizes

    def _resolve_dpi(self, page_sizes: List[Tuple[float, float]]) -> int:
        """Pick the highest DPI (up to self.image_dpi) where a single page
        fits within the available input token budget."""
        budget = self._available_input_tokens()
        max_w = max(w for w, _ in page_sizes)
        max_h = max(h for _, h in page_sizes)

        dpi = self.image_dpi
        while dpi > self.MIN_DPI:
            tokens = self.estimate_page_tokens_at_dpi(max_w, max_h, dpi)
            if tokens <= budget:
                break
            dpi = max(self.MIN_DPI, dpi - 50)

        if dpi < self.image_dpi:
            tokens_at_orig = self.estimate_page_tokens_at_dpi(max_w, max_h, self.image_dpi)
            tokens_at_new = self.estimate_page_tokens_at_dpi(max_w, max_h, dpi)
            log.info(
                f"Vision LLM: auto-scaled DPI {self.image_dpi} -> {dpi} "
                f"(page tokens {tokens_at_orig} -> {tokens_at_new}, budget {budget})"
            )
        return dpi

    def _pdf_pages_to_images(self, dpi: int) -> List[bytes]:
        """Render each PDF page to a PNG image using pypdfium2."""
        try:
            import pypdfium2 as pdfium
        except ImportError:
            raise ImportError(
                "pypdfium2 is required for PDF processing with Vision LLM. "
                "Install it with: pip install pypdfium2"
            )

        images = []
        with _PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(self.file_path)
            try:
                scale = dpi / 72.0
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    bitmap = page.render(scale=scale)
                    pil_image = bitmap.to_pil()
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    images.append(buf.getvalue())
                    log.debug(f"Rendered PDF page {page_num + 1}/{len(pdf)} ({len(images[-1])} bytes)")
                    # Free PDFium handles deterministically (PIL bytes already
                    # copied into buf above), not at GC time on another thread.
                    bitmap.close()
                    page.close()
            finally:
                pdf.close()
                # Run any remaining pypdfium2 finalizers under the lock so they
                # can't fire later mid-render on another thread (segfault).
                gc.collect()

        return images

    def _read_image_file(self) -> bytes:
        """Read an image file as bytes."""
        with open(self.file_path, "rb") as f:
            return f.read()

    def _get_mime_type(self, ext: str = "") -> str:
        """Get MIME type for image encoding."""
        ext = ext or self.file_ext
        mime_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
        }
        return mime_map.get(ext, "image/png")

    # ── API call ──────────────────────────────────────────────────────

    def _call_vision_api(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Send a single image to the vision model and return extracted text."""
        b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_image}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.api_base_url}/chat/completions"

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                choices = result.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()

                log.warning("Vision API returned empty choices")
                return ""

            except requests.exceptions.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else 0
                if status == 429 or status >= 500:
                    wait = min((2 ** attempt) + 0.5, 30)
                    log.warning(f"Vision API HTTP {status}, retrying in {wait}s (attempt {attempt + 1})")
                    time.sleep(wait)
                    continue
                log.error(f"Vision API HTTP error: {e} - {e.response.text if e.response is not None else ''}")
                raise
            except requests.exceptions.ConnectionError as e:
                last_error = e
                wait = min((2 ** attempt) + 0.5, 30)
                log.warning(f"Vision API connection error, retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < self.max_retries:
                    log.warning(f"Vision API timeout, retrying (attempt {attempt + 1})")
                    continue
                raise

        raise last_error

    # ── Document loading ──────────────────────────────────────────────

    def load(self) -> List[Document]:
        """Execute the full extraction workflow."""
        start_time = time.time()

        try:
            if self.file_ext in self.SUPPORTED_PDF_EXTENSIONS:
                return self._load_pdf()
            elif self.file_ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                return self._load_image()
            else:
                raise ValueError(
                    f"Unsupported file type: .{self.file_ext}. "
                    f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
                )
        except Exception as e:
            total_time = time.time() - start_time
            log.error(f"Vision LLM extraction failed after {total_time:.2f}s: {e}")
            return [
                Document(
                    page_content=f"Error during Vision LLM extraction: {e}",
                    metadata={"error": "processing_failed", "file_name": self.file_name},
                )
            ]

    def _load_pdf(self) -> List[Document]:
        """Extract text from every page of a PDF in parallel.

        Page rendering runs serially (pypdfium2 isn't safe across one
        PdfDocument from multiple threads), then API calls are dispatched
        through a ThreadPoolExecutor. Per-page failures are recorded as
        error Documents so the whole document is always processed.
        """
        log.info(f"Vision LLM: processing PDF '{self.file_name}'")

        page_sizes = self._get_pdf_page_sizes()
        total_pages = len(page_sizes)
        effective_dpi = self._resolve_dpi(page_sizes)

        workers = min(self.max_workers, total_pages)
        log.info(
            f"Vision LLM: {total_pages} pages at {effective_dpi} DPI, "
            f"parallel workers={workers}"
        )

        page_images = self._pdf_pages_to_images(effective_dpi)

        results: Dict[int, Tuple[Optional[str], Optional[Exception]]] = {}

        def _process_page(idx: int, img_bytes: bytes):
            label = idx + 1
            t0 = time.time()
            log.info(f"Vision LLM: page {label}/{total_pages} start")
            try:
                text = self._call_vision_api(img_bytes, "image/png")
                elapsed = time.time() - t0
                log.info(
                    f"Vision LLM: page {label}/{total_pages} done in "
                    f"{elapsed:.1f}s ({len(text) if text else 0} chars)"
                )
                return idx, text, None
            except Exception as e:
                elapsed = time.time() - t0
                log.error(
                    f"Vision LLM: page {label}/{total_pages} failed after "
                    f"{elapsed:.1f}s: {e}"
                )
                return idx, None, e

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_process_page, idx, img_bytes)
                for idx, img_bytes in enumerate(page_images)
            ]
            for fut in concurrent.futures.as_completed(futures):
                idx, text, err = fut.result()
                results[idx] = (text, err)

        documents: List[Document] = []
        failed = 0
        for idx in range(total_pages):
            text, err = results.get(idx, (None, None))
            label = idx + 1
            if err is not None:
                failed += 1
                documents.append(
                    Document(
                        page_content=f"[Error extracting page {label}: {err}]",
                        metadata={
                            "page": idx,
                            "page_label": label,
                            "total_pages": total_pages,
                            "file_name": self.file_name,
                            "error": "page_failed",
                        },
                    )
                )
            elif text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": idx,
                            "page_label": label,
                            "total_pages": total_pages,
                            "file_name": self.file_name,
                            "processing_engine": "vision_llm",
                            "model": self.model,
                            "dpi": effective_dpi,
                            "content_length": len(text),
                        },
                    )
                )
            else:
                log.warning(f"Vision LLM: empty response for page {label}")

        if not documents:
            return [
                Document(
                    page_content="No text content extracted from document",
                    metadata={"error": "no_content", "file_name": self.file_name, "total_pages": total_pages},
                )
            ]

        status = f"extracted {len(documents)} pages"
        if failed:
            status += f" ({failed} failed)"
        log.info(f"Vision LLM: {status} from '{self.file_name}'")
        return documents

    def _load_image(self) -> List[Document]:
        """Extract text from a single image file."""
        log.info(f"Vision LLM: processing image '{self.file_name}'")
        img_bytes = self._read_image_file()
        mime_type = self._get_mime_type()

        text = self._call_vision_api(img_bytes, mime_type)

        if not text:
            return [
                Document(
                    page_content="No text content extracted from image",
                    metadata={"error": "no_content", "file_name": self.file_name},
                )
            ]

        return [
            Document(
                page_content=text,
                metadata={
                    "file_name": self.file_name,
                    "processing_engine": "vision_llm",
                    "model": self.model,
                    "content_length": len(text),
                },
            )
        ]
