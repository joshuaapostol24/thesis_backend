import os
import requests

from fastapi import APIRouter
from pydantic import BaseModel

from dotenv import load_dotenv

from supabase import (
    create_client,
    Client
)

load_dotenv()

router = APIRouter(
    prefix="/sms",
    tags=["SMS"]
)

supabase: Client = create_client(

    os.getenv("SUPABASE_URL"),

    os.getenv("SUPABASE_KEY")

)


# ─────────────────────────────
# MODELS
# ─────────────────────────────

class TestSMS(BaseModel):

    phone: str
    message: str


class BulkSMSRequest(BaseModel):

    user_ids: list[str]
    message: str


# ─────────────────────────────
# FORMAT PHONE
# ─────────────────────────────

def format_phone(number: str):

    number = number.strip()

    if number.startswith("09"):

        return "63" + number[1:]

    if number.startswith("+63"):

        return number.replace("+", "")

    return number


# ─────────────────────────────
# SINGLE SMS
# ─────────────────────────────

@router.post("/test")
def send_test_sms(data: TestSMS):

    response = requests.post(

        "https://www.iprogsms.com/api/v1/sms_messages",

        params={

            "api_token":
                os.getenv(
                    "IPROGSMS_API_TOKEN"
                ),

            "message":
                data.message,

            "phone_number":
                format_phone(
                    data.phone
                )

        }

    )

    return response.json()


# ─────────────────────────────
# BULK SMS
# ─────────────────────────────

@router.post("/send-selected")
def send_selected_sms(
    data: BulkSMSRequest
):

    result = supabase.table("user") \
        .select("phone") \
        .in_("id", data.user_ids) \
        .eq("status", "verified") \
        .execute()

    numbers = [

        row["phone"]

        for row in result.data

        if row.get("phone")

    ]

    sent = 0

    for number in numbers:

        try:

            requests.post(

                "https://www.iprogsms.com/api/v1/sms_messages",

                params={

                    "api_token":
                        os.getenv(
                            "IPROGSMS_API_TOKEN"
                        ),

                    "message":
                        data.message,

                    "phone_number":
                        format_phone(number)

                }

            )

            sent += 1

        except Exception as e:

            print(e)

    return {

        "success": True,

        "sent_to": sent

    }