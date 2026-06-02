import json
import logging
import aiohttp

from open_webui.config import WEBUI_FAVICON_URL
from open_webui.env import AIOHTTP_CLIENT_TIMEOUT, VERSION

log = logging.getLogger(__name__)


async def post_webhook(name: str, url: str, message: str, event_data: dict) -> bool:
    try:
        log.debug(f"post_webhook: {url}, {message}, {event_data}")
        payload = {}

        # Slack and Google Chat Webhooks
        if "https://hooks.slack.com" in url or "https://chat.googleapis.com" in url:
            payload["text"] = message
        # Discord Webhooks
        elif "https://discord.com/api/webhooks" in url:
            payload["content"] = (
                message
                if len(message) < 2000
                else f"{message[: 2000 - 20]}... (truncated)"
            )
        # Microsoft Teams — legacy Office 365 connector incoming webhook (retired)
        elif "webhook.office.com" in url:
            action = event_data.get("action", "undefined")
            facts = [
                {"name": name, "value": value}
                for name, value in json.loads(event_data.get("user", {})).items()
            ]
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "0076D7",
                "summary": message,
                "sections": [
                    {
                        "activityTitle": f"{message}, Environment: {event_data.get('environment')}",
                        "activitySubtitle": f"{name} ({VERSION}) - {action}",
                        "activityImage": WEBUI_FAVICON_URL,
                        "facts": facts,
                        "markdown": True,
                    }
                ],
            }
        # Microsoft Teams — new "Workflows" (Power Automate / Logic Apps) webhook.
        # The "Post card in a chat or channel" action requires the bot message
        # envelope with an Adaptive Card inside; anything whose top-level type is
        # not "AdaptiveCard" fails with "Property 'type' must be 'AdaptiveCard'".
        elif any(
            host in url
            for host in (
                "logic.azure.com",
                "azure-apim.net",
                "powerplatform.com",
                "powerautomate.com",
            )
        ):
            action = event_data.get("action", "undefined")
            user = json.loads(event_data.get("user") or "{}")
            facts = [{"title": f"{key}:", "value": str(value)} for key, value in user.items()]
            payload = {
                "type": "message",
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "contentUrl": None,
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": [
                                {
                                    "type": "ColumnSet",
                                    "columns": [
                                        {
                                            "type": "Column",
                                            "width": "auto",
                                            "verticalContentAlignment": "Center",
                                            "items": [
                                                {
                                                    "type": "Image",
                                                    "url": WEBUI_FAVICON_URL,
                                                    "size": "Small",
                                                    "altText": name,
                                                }
                                            ],
                                        },
                                        {
                                            "type": "Column",
                                            "width": "stretch",
                                            "verticalContentAlignment": "Center",
                                            "items": [
                                                {
                                                    "type": "TextBlock",
                                                    "size": "Medium",
                                                    "weight": "Bolder",
                                                    "text": message,
                                                    "wrap": True,
                                                },
                                                {
                                                    "type": "TextBlock",
                                                    "spacing": "None",
                                                    "isSubtle": True,
                                                    "wrap": True,
                                                    "text": (
                                                        f"{name} ({VERSION}) - {action} - "
                                                        f"Environment: {event_data.get('environment')}"
                                                    ),
                                                },
                                            ],
                                        },
                                    ],
                                },
                                {"type": "FactSet", "facts": facts},
                            ],
                        },
                    }
                ],
            }
        # Default Payload
        else:
            payload = {**event_data}

        log.debug(f"payload: {payload}")
        async with aiohttp.ClientSession(
            trust_env=True, timeout=aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT)
        ) as session:
            async with session.post(url, json=payload) as r:
                r_text = await r.text()
                r.raise_for_status()
                log.debug(f"r.text: {r_text}")

        return True
    except Exception as e:
        log.exception(e)
        return False
