from fastapi import APIRouter
from pydantic import BaseModel

from modules.sms_service import (
    send_sms
)

router = APIRouter(
    prefix="/sms",
    tags=["SMS"]
)


class SMSRequest(BaseModel):
    number: str
    message: str


@router.post("/send")
def send_sms_route(data: SMSRequest):

    result = send_sms(
        data.number,
        data.message
    )

    return {
        "success": True,
        "result": result
    }