import os

import requests

SEMAPHORE_API_KEY = os.getenv("SEMAPHORE_API_KEY")


def send_sms(number: str, message: str):
    if not SEMAPHORE_API_KEY:
        raise RuntimeError("Missing SEMAPHORE_API_KEY environment variable.")

    response = requests.post(
        "https://api.semaphore.co/api/v4/messages",
        data={
            "apikey": SEMAPHORE_API_KEY,
            "number": number,
            "message": message,
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()
