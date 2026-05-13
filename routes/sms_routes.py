import os
import requests

from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

router = APIRouter(prefix="/sms", tags=["SMS"])

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


class TestSMS(BaseModel):
    phone: str
    message: str


class BulkSMSRequest(BaseModel):
    user_ids: list[str]
    message: str


@router.post("/test")
def send_test_sms(data: TestSMS):
    response = requests.post(
        "   ",
        params={
            "api_token": os.getenv("IPROGSMS_API_TOKEN"),
            "message": data.message,
            "phone_number": data.phone
        }
    )
    return response.json()


@router.post("/send-selected")
def send_selected_sms(data: BulkSMSRequest):
    result = supabase.table("user") \
        .select("phone") \
        .in_("id", data.user_ids) \
        .eq("status", "verified") \
        .execute()

    numbers = [row["phone"] for row in result.data if row.get("phone")]

    for number in numbers:
        requests.post(
            "https://www.iprogsms.com/api/v1/sms_messages",
            params={
                "api_token": os.getenv("IPROGSMS_API_TOKEN"),
                "message": data.message,
                "phone_number": number
            }
        )

    return {
        "success": True,
        "sent_to": len(numbers)
    }