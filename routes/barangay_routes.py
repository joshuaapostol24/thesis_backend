import logging

from fastapi import APIRouter, HTTPException

from modules.database import (
    get_barangay_centroid,
    get_barangay_name,
    get_barangay_hazard_profile,
    get_recent_weather,
    list_barangay_profiles,
)

router = APIRouter(tags=["Barangays"])
logger = logging.getLogger(__name__)


@router.get("/barangays")
def list_barangays():
    """
    Returns all barangays with their hazard profiles.
    A single call to list_barangay_profiles() is the source of truth.
    Any DB error surfaces as a 503 so the caller knows something is wrong
    rather than receiving silent empty data.
    """
    try:
        return list_barangay_profiles()
    except Exception as e:
        logger.error("list_barangays failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve barangay data. Please try again later."
        )


@router.get("/barangays/{barangay_id}")
def get_barangay(barangay_id: int):
    try:
        lat, lon = get_barangay_centroid(barangay_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Barangay {barangay_id} not found")

    name           = get_barangay_name(barangay_id)
    hazard         = get_barangay_hazard_profile(barangay_id)
    recent_weather = get_recent_weather(barangay_id, limit=5)

    return {
        "barangay_id":    barangay_id,
        "name":           name,
        "lat":            lat,
        "lon":            lon,
        **hazard,
        "recent_weather": recent_weather,
    }