import logging

from fastapi import APIRouter

from ml_service_python.modules.database import (
    get_connection,
    get_barangay_centroid,
    get_barangay_name,
    get_barangay_hazard_profile,
    get_recent_weather,
    list_barangay_profiles,
)

router = APIRouter(
    tags=["Barangays"]
)

logger = logging.getLogger(__name__)


@router.get("/barangays")
def list_barangays():

    try:
        return list_barangay_profiles()

    except Exception as api_exc:
        logger.warning(
            "Supabase API fallback failed: %s",
            api_exc
        )

    conn = get_connection()
    cur = conn.cursor()

    try:

        cur.execute("""
            SELECT
                b.barangay_id,
                b.name,
                b.lat,
                b.lon,
                h.flood_hazard_level,
                h.flood_hazard_score,
                h.max_ssa_level,
                h.storm_surge_score,
                h.overall_hazard
            FROM barangay_list b
            LEFT JOIN barangay_hazard_profile h
            ON b.barangay_id = h.barangay_id
            ORDER BY b.barangay_id
        """)

        rows = cur.fetchall()

        return [
            {
                "barangay_id": row[0],
                "name": row[1],
                "lat": float(row[2]),
                "lon": float(row[3]),
                "flood_hazard_level": row[4],
                "flood_hazard_score":
                    float(row[5]) if row[5] else 0.0,
                "max_ssa_level":
                    int(row[6]) if row[6] else 0,
                "storm_surge_score":
                    float(row[7]) if row[7] else 0.0,
                "overall_hazard": row[8],
            }
            for row in rows
        ]

    finally:
        cur.close()
        conn.close()


@router.get("/barangays/{barangay_id}")
def get_barangay(barangay_id: int):

    lat, lon = get_barangay_centroid(
        barangay_id
    )

    name = get_barangay_name(
        barangay_id
    )

    hazard = get_barangay_hazard_profile(
        barangay_id
    )

    recent_weather = get_recent_weather(
        barangay_id,
        limit=5
    )

    return {
        "barangay_id": barangay_id,
        "name": name,
        "lat": lat,
        "lon": lon,
        **hazard,
        "recent_weather": recent_weather,
    }