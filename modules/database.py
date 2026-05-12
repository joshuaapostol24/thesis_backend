import logging

import psycopg2
from sqlalchemy import create_engine

from modules.config import get_database_url

logger = logging.getLogger(__name__)

# Raises RuntimeError on startup if env vars are missing — intentional.
DATABASE_URL = get_database_url()

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)

# ── Hazard profile cache (loaded at app startup) ───────────────────────────
_hazard_cache: dict = {}


def load_hazard_cache() -> None:
    """Load all hazard profiles into memory at startup."""
    global _hazard_cache
    try:
        profiles = list_barangay_profiles()
        _hazard_cache = {p["barangay_id"]: p for p in profiles}
        logger.info("Loaded %d hazard profiles into cache", len(_hazard_cache))
    except Exception as e:
        logger.error("Failed to load hazard cache: %s. Will use per-request DB queries.", e)


def get_connection():
    """
    Returns a raw psycopg2 connection.
    Prefer using `engine` via SQLAlchemy for new code — this exists for
    legacy callers that use cursor-based queries.
    """
    return psycopg2.connect(DATABASE_URL)


def list_barangay_profiles() -> list:
    """Return barangay rows with their hazard profile."""
    with engine.connect() as conn:
        from sqlalchemy import text
        rows = conn.execute(text("""
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
        """)).fetchall()

    return [
    {
        "barangay_id":        row[0],
        "name":               row[1],
        "lat":                float(row[2]) if row[2] is not None else 0.0,
        "lon":                float(row[3]) if row[3] is not None else 0.0,
        "flood_hazard_level": row[4] or "Low",
        "flood_hazard_score": float(row[5]) if row[5] is not None else 0.0,
        "max_ssa_level":      int(row[6]) if row[6] is not None else 0,
        "storm_surge_score":  float(row[7]) if row[7] is not None else 0.0,
        "overall_hazard":     row[8] or "LOW",
        # ── aliases for context.py ────────────────────────────────────────
        "overall":            row[8] or "LOW",
        "ssa_level":          int(row[6]) if row[6] is not None else 0,
        "flood_score":        float(row[5]) if row[5] is not None else 0.0,
    }
    for row in rows
]


# ── Barangay list ─────────────────────────────────────────────────────────────

def get_barangay_centroid(barangay_id: int) -> tuple:
    """Returns (lat, lon) from barangay_list table."""
    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT lat, lon FROM barangay_list WHERE barangay_id = :id"),
            {"id": barangay_id}
        ).fetchone()
    if row:
        return float(row[0]), float(row[1])
    raise ValueError(f"Could not find coordinates for barangay_id={barangay_id}")


def get_barangay_name(barangay_id: int) -> str:
    """Returns barangay name from barangay_list."""
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT name FROM barangay_list WHERE barangay_id = :id"),
                {"id": barangay_id}
            ).fetchone()
        return row[0] if row else f"Barangay {barangay_id}"
    except Exception as e:
        logger.error("get_barangay_name error: %s", e)
        return f"Barangay {barangay_id}"


# ── Hazard profile ────────────────────────────────────────────────────────────

def get_barangay_hazard_profile(barangay_id: int) -> dict:
    """
    Returns full hazard profile — from cache if available, else from DB.
    Cache is loaded at application startup for production use.
    This is the B (Barangay Context) parameter from Algorithm 3.
    """
    # Try cache first (O(1) lookup, non-blocking)
    if _hazard_cache and barangay_id in _hazard_cache:
        return _hazard_cache[barangay_id]
    
    # Fallback to DB query if cache empty (development, or cache load failed)
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT
                    flood_hazard_level,
                    flood_hazard_score,
                    in_ssa1, in_ssa2, in_ssa3,
                    max_ssa_level,
                    storm_surge_score,
                    overall_hazard
                FROM barangay_hazard_profile
                WHERE barangay_id = :id
            """), {"id": barangay_id}).fetchone()

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
                "overall":            row[7],
                "ssa_level":          int(row[5]),
                "flood_score":        float(row[1]),
            }
    except Exception as e:
        logger.error("get_barangay_hazard_profile DB error: %s", e)

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
        "overall":            "LOW",
        "ssa_level":          0,
        "flood_score":        0.20,
    }


# ── Weather data ──────────────────────────────────────────────────────────────

def get_barangay_features(barangay_id: int) -> dict:
    """
    Returns the most recent weather reading for a barangay from weather_data.
    Used as a fallback when the live weather API returns zeros.
    """
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT rainfall, humidity
                FROM weather_data
                WHERE city = 'Mamburao'
                ORDER BY timestamp DESC
                LIMIT 1
            """)).fetchone()
        rainfall = float(row[0]) if row and row[0] is not None else 0.0
        humidity = float(row[1]) if row and row[1] is not None else 0.0
    except Exception as e:
        logger.error("get_barangay_features weather query error: %s", e)
        rainfall = 0.0
        humidity = 0.0

    profile = get_barangay_hazard_profile(barangay_id)
    flood_score = profile.get("flood_score", 0.20)

    return {
        "rainfall": rainfall,
        "humidity": humidity,
        "flood":    flood_score,
    }


def get_recent_weather(barangay_id: int, limit: int = 10) -> list:
    """
    Returns recent weather observations for a **specific barangay**.
    
    Uses barangay_id to retrieve spatially-aware weather sequence.
    Falls back to "Mamburao" city data if no barangay-specific records.
    """
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            # Try to fetch barangay-specific weather first
            rows = conn.execute(text("""
                SELECT timestamp, temperature, pressure, humidity,
                       wind_speed, rainfall, rainfall_category,
                       season, risk_level
                FROM weather_data
                WHERE barangay_id = :barangay_id
                ORDER BY timestamp DESC
                LIMIT :limit
            """), {"barangay_id": barangay_id, "limit": limit}).fetchall()
            
            # Fallback to city-wide data if no barangay-specific records
            if not rows:
                logger.debug(
                    "No barangay-specific weather for %d. Falling back to city data.",
                    barangay_id
                )
                rows = conn.execute(text("""
                    SELECT timestamp, temperature, pressure, humidity,
                           wind_speed, rainfall, rainfall_category,
                           season, risk_level
                    FROM weather_data
                    WHERE city = 'Mamburao' AND barangay_id IS NULL
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """), {"limit": limit}).fetchall()

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
        logger.error("get_recent_weather error for barangay_id=%d: %s", barangay_id, e)
        return []


# ── Spatial lookups ───────────────────────────────────────────────────────────

def get_storm_surge_score(lat: float, lon: float) -> float:
    """Fast lookup using barangay_hazard_profile instead of spatial query."""
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT h.storm_surge_score,
                       SQRT(POWER(b.lat - :lat, 2) + POWER(b.lon - :lon, 2)) AS dist
                FROM barangay_list b
                JOIN barangay_hazard_profile h ON b.barangay_id = h.barangay_id
                ORDER BY dist ASC
                LIMIT 1
            """), {"lat": lat, "lon": lon}).fetchone()
        return float(row[0]) if row else 0.0
    except Exception as e:
        logger.error("get_storm_surge_score error: %s", e)
        return 0.0


def get_flood_score(lat: float, lon: float) -> float:
    """Fast lookup using barangay_hazard_profile instead of spatial query."""
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT h.flood_hazard_score,
                       SQRT(POWER(b.lat - :lat, 2) + POWER(b.lon - :lon, 2)) AS dist
                FROM barangay_list b
                JOIN barangay_hazard_profile h ON b.barangay_id = h.barangay_id
                ORDER BY dist ASC
                LIMIT 1
            """), {"lat": lat, "lon": lon}).fetchone()
        return float(row[0]) if row else 0.0
    except Exception as e:
        logger.error("get_flood_score error: %s", e)
        return 0.0


# ── Utilities ─────────────────────────────────────────────────────────────────

def list_tables():
    """Lists all public tables in the Supabase database."""
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            tables = [
                row[0] for row in conn.execute(text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)).fetchall()
            ]
        logger.info("Tables: %s", tables)
        return tables
    except Exception as e:
        logger.error("list_tables error: %s", e)
        return []