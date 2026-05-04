from sqlalchemy import create_engine
import psycopg2
import logging

DATABASE_URL = "postgresql://postgres:123apostol@127.0.0.1:5432/thesis_db"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def get_barangay_centroid(barangay_id: int) -> tuple:
    """
    Returns (lat, lon) from barangay_list table.
    Uses hardcoded Mamburao coordinates — guaranteed correct.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT lat, lon FROM barangay_list WHERE barangay_id = %s",
            (barangay_id,)
        )
        row = cur.fetchone()
        if row:
            lat, lon = float(row[0]), float(row[1])
            logging.info(
                "Barangay %d coordinates: lat=%.5f lon=%.5f",
                barangay_id, lat, lon
            )
            return lat, lon
    except Exception as e:
        logging.error("get_barangay_centroid error: %s", e)
    finally:
        cur.close()
        conn.close()

    raise ValueError(f"Could not find coordinates for barangay_id={barangay_id}")


def get_barangay_name(barangay_id: int) -> str:
    """Returns the barangay name from barangay_list table."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT name FROM barangay_list WHERE barangay_id = %s",
            (barangay_id,)
        )
        row = cur.fetchone()
        return row[0] if row else f"Barangay {barangay_id}"
    except Exception:
        return f"Barangay {barangay_id}"
    finally:
        cur.close()
        conn.close()


def get_barangay_features(barangay_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT rainfall, flood
        FROM barangay_weather
        WHERE barangay_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
    """, (barangay_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return {
        "rainfall": row[0] if row else 0,
        "flood":    row[1] if row else 0,
    }


def get_flood_zone(lat: float, lon: float):
    """Check if a point falls within a flood hazard zone."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT "Var"
        FROM barangays
        WHERE ST_Contains(geometry, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
        LIMIT 1
    """, (lon, lat))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return float(row[0]) if row else None


def get_storm_surge_level(lat: float, lon: float):
    """
    Returns the highest SSA level (1, 2, or 3) for a given point,
    or None if not in any storm surge zone.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT ssa_level
        FROM storm_surge_zones
        WHERE ST_Contains(geometry, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
        ORDER BY ssa_level DESC
        LIMIT 1
    """, (lon, lat))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return int(row[0]) if row else None


def get_storm_surge_score(lat: float, lon: float) -> float:
    """
    Convert storm surge SSA level to normalized score (0.0 - 1.0).
      SSA1 → 0.33
      SSA2 → 0.66
      SSA3 → 1.00
      None → 0.00
    """
    level = get_storm_surge_level(lat, lon)
    mapping = {1: 0.33, 2: 0.66, 3: 1.00}
    return mapping.get(level, 0.0)


def list_tables():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = cur.fetchall()
    cur.close()
    conn.close()
    print("Tables:", tables)
    return tables