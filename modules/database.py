import logging

import psycopg2
from sqlalchemy import create_engine

from ml_service_python.modules.config import get_database_url

logger = logging.getLogger(__name__)

try:
    DATABASE_URL = get_database_url()
except RuntimeError:
    DATABASE_URL = (
        "postgresql://postgres.jpovamcznyzoemcnjrgs:"
        "123Apostol%40Coco"
        "@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres"
        "?sslmode=require"
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def list_barangay_profiles() -> list:
    """Return barangay rows with their hazard profile."""
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
                "lat": float(row[2]) if row[2] is not None else 0.0,
                "lon": float(row[3]) if row[3] is not None else 0.0,
                "flood_hazard_level": row[4] or "Low",
                "flood_hazard_score": float(row[5]) if row[5] is not None else 0.0,
                "max_ssa_level": int(row[6]) if row[6] is not None else 0,
                "storm_surge_score": float(row[7]) if row[7] is not None else 0.0,
                "overall_hazard": row[8] or "LOW",
            }
            for row in rows
        ]

    except Exception as e:
        logger.error("list_barangay_profiles error: %s", e)
        raise

    finally:
        cur.close()
        conn.close()


# ── Barangay list ─────────────────────────────────────────────────────────────

def get_barangay_centroid(barangay_id: int) -> tuple:
    """Returns (lat, lon) from barangay_list table."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT lat, lon FROM barangay_list WHERE barangay_id = %s",
            (barangay_id,)
        )
        row = cur.fetchone()
        if row:
            return float(row[0]), float(row[1])
    except Exception as e:
        logger.error("get_barangay_centroid error: %s", e)
    finally:
        cur.close()
        conn.close()
    raise ValueError(f"Could not find coordinates for barangay_id={barangay_id}")


def get_barangay_name(barangay_id: int) -> str:
    """Returns barangay name from barangay_list."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT name FROM barangay_list WHERE barangay_id = %s",
            (barangay_id,)
        )
        row = cur.fetchone()
        return row[0] if row else f"Barangay {barangay_id}"
    except Exception as e:
        logger.error("get_barangay_name error: %s", e)
        return f"Barangay {barangay_id}"
    finally:
        cur.close()
        conn.close()


# ── Hazard profile ────────────────────────────────────────────────────────────

def get_barangay_hazard_profile(barangay_id: int) -> dict:
    """
    Returns full hazard profile from barangay_hazard_profile table.
    This is the B (Barangay Context) parameter from Algorithm 3.
    Much faster than spatial queries — O(1) lookup by barangay_id.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                flood_hazard_level,
                flood_hazard_score,
                in_ssa1, in_ssa2, in_ssa3,
                max_ssa_level,
                storm_surge_score,
                overall_hazard
            FROM barangay_hazard_profile
            WHERE barangay_id = %s
        """, (barangay_id,))
        row = cur.fetchone()
        if row:
            return {
                "flood_hazard_level": row[0],
                "flood_hazard_score": float(row[1]),
                "in_ssa1":            bool(row[2]),
                "in_ssa2":            bool(row[3]),
                "in_ssa3":            bool(row[4]),
                "max_ssa_level":      int(row[5]),
                "storm_surge_score":  float(row[6]),
                "overall_hazard":     row[7],
                # ── aliases for context.py ────────────────────────────────
                "overall":            row[7],
                "ssa_level":          int(row[5]),
                "flood_score":        float(row[1]),
            }
    except Exception as e:
        logger.error("get_barangay_hazard_profile error: %s", e)
    finally:
        cur.close()
        conn.close()

    # Fallback — safe defaults, never crash the API
    logger.warning(
        "No hazard profile found for barangay_id=%d — using LOW defaults",
        barangay_id
    )
    return {
        "flood_hazard_level": "Low",
        "flood_hazard_score": 0.20,
        "in_ssa1":            False,
        "in_ssa2":            False,
        "in_ssa3":            False,
        "max_ssa_level":      0,
        "storm_surge_score":  0.0,
        "overall_hazard":     "LOW",
        # ── aliases for context.py ────────────────────────────────────────
        "overall":            "LOW",
        "ssa_level":          0,
        "flood_score":        0.20,
    }


# ── Weather data ──────────────────────────────────────────────────────────────

def get_barangay_features(barangay_id: int) -> dict:
    """
    Returns latest weather features from barangay_weather table.
    Used as fallback in main_api.py when live weather API returns 0.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT rainfall, flood
            FROM barangay_weather
            WHERE barangay_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (barangay_id,))
        row = cur.fetchone()
        return {
            "rainfall": float(row[0]) if row else 0.0,
            "flood":    float(row[1]) if row else 0.0,
        }
    except Exception as e:
        logger.error("get_barangay_features error: %s", e)
        return {"rainfall": 0.0, "flood": 0.0}
    finally:
        cur.close()
        conn.close()


def get_recent_weather(barangay_id: int, limit: int = 10) -> list:
    """
    Returns recent weather observations for a barangay.
    NOTE: Currently queries by city='Mamburao' since weather_final.csv
    uses a single coordinate for all barangays. Per-barangay weather
    data is a known limitation documented in the paper.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT timestamp, temperature, pressure, humidity,
                   wind_speed, rainfall, rainfall_category,
                   season, risk_level
            FROM weather_observations
            WHERE city = 'Mamburao'
            ORDER BY timestamp DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        return [
            {
                "timestamp":         str(row[0]),
                "temperature":       float(row[1]) if row[1] else 0.0,
                "pressure":          float(row[2]) if row[2] else 0.0,
                "humidity":          float(row[3]) if row[3] else 0.0,
                "wind_speed":        float(row[4]) if row[4] else 0.0,
                "rainfall":          float(row[5]) if row[5] else 0.0,
                "rainfall_category": row[6] if row[6] else "None",
                "season":            row[7] if row[7] else "Unknown",
                "risk_level":        row[8] if row[8] else "LOW",
            }
            for row in rows
        ]
    except Exception as e:
        logger.error("get_recent_weather error: %s", e)
        return []
    finally:
        cur.close()
        conn.close()


# ── Spatial lookups ───────────────────────────────────────────────────────────

def get_storm_surge_score(lat: float, lon: float) -> float:
    """
    Fast lookup using barangay_hazard_profile instead of spatial query.
    Finds nearest barangay and returns its storm surge score.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT h.storm_surge_score,
                   SQRT(POWER(b.lat - %s, 2) + POWER(b.lon - %s, 2)) AS dist
            FROM barangay_list b
            JOIN barangay_hazard_profile h ON b.barangay_id = h.barangay_id
            ORDER BY dist ASC
            LIMIT 1
        """, (lat, lon))
        row = cur.fetchone()
        return float(row[0]) if row else 0.0
    except Exception as e:
        logger.error("get_storm_surge_score error: %s", e)
        return 0.0
    finally:
        cur.close()
        conn.close()


def get_flood_score(lat: float, lon: float) -> float:
    """
    Fast lookup using barangay_hazard_profile instead of spatial query.
    Finds nearest barangay and returns its flood hazard score.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT h.flood_hazard_score,
                   SQRT(POWER(b.lat - %s, 2) + POWER(b.lon - %s, 2)) AS dist
            FROM barangay_list b
            JOIN barangay_hazard_profile h ON b.barangay_id = h.barangay_id
            ORDER BY dist ASC
            LIMIT 1
        """, (lat, lon))
        row = cur.fetchone()
        return float(row[0]) if row else 0.0
    except Exception as e:
        logger.error("get_flood_score error: %s", e)
        return 0.0
    finally:
        cur.close()
        conn.close()


# ── Utilities ─────────────────────────────────────────────────────────────────

def list_tables():
    """Lists all public tables in the Supabase database."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]
        logger.info("Tables: %s", tables)
        return tables
    except Exception as e:
        logger.error("list_tables error: %s", e)
        return []
    finally:
        cur.close()
        conn.close()
