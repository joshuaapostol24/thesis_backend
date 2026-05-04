"""
modules/flood_hazard.py
───────────────────────
Queries flood hazard data directly from PostgreSQL (barangays table)
instead of reading from a local shapefile.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Hazard mapping (Var value → flood score)
_HAZARD_MAP = {
    1.0: 0.2,   # Low
    2.0: 0.6,   # Medium
    3.0: 1.0,   # High
}

_DEFAULT_FLOOD_SCORE = 0.0


def lookup_flood_hazard(lat: float, lon: float) -> float:
    """
    Query the barangays table in PostgreSQL to get flood hazard score
    for the given coordinates using PostGIS spatial query.
    """
    try:
        import psycopg2
        DATABASE_URL = "postgresql://postgres:123apostol@127.0.0.1:5432/thesis_db"
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # Primary: check if point is inside a polygon
        cur.execute("""
            SELECT "Var"
            FROM barangays
            WHERE ST_Contains(geometry, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
            LIMIT 1
        """, (lon, lat))

        row = cur.fetchone()

        # Fallback: find nearest polygon if point is outside all polygons
        if not row:
            cur.execute("""
                SELECT "Var"
                FROM barangays
                ORDER BY geometry <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                LIMIT 1
            """, (lon, lat))
            row = cur.fetchone()

        cur.close()
        conn.close()

        if row:
            var = float(row[0])
            score = _HAZARD_MAP.get(var, _DEFAULT_FLOOD_SCORE)
            logger.info("Flood hazard lookup: lat=%.5f lon=%.5f → Var=%.1f score=%.2f", lat, lon, var, score)
            return score

        return _DEFAULT_FLOOD_SCORE

    except Exception as e:
        logger.error("Flood hazard DB lookup error: %s", e)
        return _DEFAULT_FLOOD_SCORE


def enrich_E_with_shapefile(HR: dict, E: dict) -> None:
    """
    Enrich E dictionary with flood hazard score from PostgreSQL.
    Function name kept for compatibility with existing main_api.py.
    """
    lat = E.get("osm_lat") or HR.get("lat")
    lon = E.get("osm_lon") or HR.get("lon")

    if lat is None or lon is None:
        logger.warning("Missing coordinates → fallback flood score used")
        E["flood"] = _DEFAULT_FLOOD_SCORE
        E["flood_source"] = "fallback"
        return

    score = lookup_flood_hazard(float(lat), float(lon))

    E["flood"] = score
    E["flood_source"] = "database"

    logger.info(
        "Flood enriched from DB: lat=%.5f lon=%.5f → score=%.2f",
        lat, lon, score
    )


def generate_training_flood_column(barangay_coords: dict) -> dict:
    return {
        bid: lookup_flood_hazard(lat, lon)
        for bid, (lat, lon) in barangay_coords.items()
    }


MAMBURAO_BARANGAY_COORDS = {
    1:  (13.2232, 120.5987),
    2:  (13.2181, 120.5921),
    3:  (13.2296, 120.6051),
    4:  (13.2150, 120.6100),
    5:  (13.2350, 120.5900),
    6:  (13.2100, 120.5850),
    7:  (13.2400, 120.6000),
    8:  (13.2050, 120.6150),
    9:  (13.2450, 120.5950),
    10: (13.2000, 120.6200),
    11: (13.2500, 120.6050),
    12: (13.1950, 120.6250),
    13: (13.2550, 120.5850),
    14: (13.2250, 120.6150),
    15: (13.2320, 120.5820),
}