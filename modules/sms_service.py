import logging
import os

import requests

logger = logging.getLogger(__name__)


def send_sms(number: str, message: str) -> dict:
    api_key = os.getenv("SEMAPHORE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing SEMAPHORE_API_KEY environment variable.")

    try:
        response = requests.post(
            "https://api.semaphore.co/api/v4/messages",
            data={
                "apikey":  api_key,
                "number":  number,
                "message": message,
            },
            timeout=15,
        )
        response.raise_for_status()
        logger.info("SMS sent successfully to %s", number)
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error("SMS HTTP error sending to %s: %s — response: %s", number, e, e.response.text)
        raise
    except requests.exceptions.RequestException as e:
        logger.error("SMS request failed sending to %s: %s", number, e)
        raise