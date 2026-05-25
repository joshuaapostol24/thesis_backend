import logging

logger = logging.getLogger(__name__)

VALID_RISK_LEVELS = {
    "VERY LOW",
    "LOW",
    "MODERATE",
    "HIGH",
    "VERY HIGH",
}


def generate_alert(hazard: str, barangay: str, risk_level: str) -> dict:
    if risk_level not in VALID_RISK_LEVELS:
        logger.warning("Unexpected risk level '%s'. Proceeding anyway.", risk_level)
    return {
        "risk_level": risk_level,
        "hazard": hazard,
        "barangay": barangay,
        "message": f"[{risk_level}] {hazard} risk detected in {barangay}"
    }


def send_alert(alert: dict) -> None:
    logger.info("ALERT SENT: %s", alert["message"])
    print("Sending Alert:", alert["message"])
