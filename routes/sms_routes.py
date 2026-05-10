import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from modules.sms_service import send_sms
from routes.auth_routes import verify_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sms", tags=["SMS"])


class SMSRequest(BaseModel):
    number: str
    message: str


@router.post("/send")
def send_sms_route(
    data: SMSRequest,
    _email: str = Depends(verify_token),   # requires valid JWT
):
    """
    Send an SMS via Semaphore. Requires authentication.
    Only authenticated users can trigger outbound messages.
    """
    try:
        result = send_sms(data.number, data.message)
        return {"success": True, "result": result}
    except RuntimeError as e:
        logger.error("SMS configuration error: %s", e)
        raise HTTPException(status_code=503, detail="SMS service not configured")
    except Exception as e:
        logger.error("SMS send failed to %s: %s", data.number, e)
        raise HTTPException(status_code=502, detail="Failed to send SMS")