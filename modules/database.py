import logging

import psycopg2
from sqlalchemy import create_engine
from supabase import create_client

from .config import get_database_url, get_supabase_key, get_supabase_url

DATABASE_URL = get_database_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def get_supabase_client():
    return create_client(get_supabase_url(), get_supabase_key())


def _default_hazard_profile() -> dict:
    return {
        "flood_hazard_level": "Unknown",
        "flood_hazard_score": 0.0,
        "max_ssa_level": 0,
        "storm_surge_score": 0.0,
        "overall_hazard": "LOW",
    }


def get_barangay_centroid(barangay_id: int) -> tuple:
    """
    Returns (lat, lon) from barangay_list table.
    Uses hardcoded Mamburao coordinates — guaranteed correct.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
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
        logging.warning("get_barangay_centroid direct DB failed: %s", e)
        response = (
            get_supabase_client()
            .table("barangay_list")
            .select("lat,lon")
            .eq("barangay_id", barangay_id)
            .limit(1)
            .execute()
        )
        if response.data:
            row = response.data[0]
            return float(row.get("lat") or 0), float(row.get("lon") or 0)
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()

    raise ValueError(f"Could not find coordinates for barangay_id={barangay_id}")


def get_barangay_name(barangay_id: int) -> str:
    """Returns the barangay name from barangay_list table."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM barangay_list WHERE barangay_id = %s",
            (barangay_id,)
        )
        row = cur.fetchone()
        return row[0] if row else f"Barangay {barangay_id}"
    except Exception:
        try:
            response = (
                get_supabase_client()
                .table("barangay_list")
                .select("name")
                .eq("barangay_id", barangay_id)
                .limit(1)
                .execute()
            )
            return response.data[0]["name"] if response.data else f"Barangay {barangay_id}"
        except Exception:
            pass
        return f"Barangay {barangay_id}"
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()


def get_barangay_features(barangay_id: int):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT rainfall, humidity, soil, flood, storm_surge
            FROM barangay_training_data
            WHERE barangay_id = %s
            ORDER BY timestamp DESC NULLS LAST, id DESC
            LIMIT 1
        """, (barangay_id,))
        row = cur.fetchone()
        return {
            "rainfall": float(row[0] or 0) if row else 0.0,
            "humidity": float(row[1] or 0) if row else 0.0,
            "soil": float(row[2] or 0) if row else 0.0,
            "flood": float(row[3] or 0) if row else 0.0,
            "storm_surge": float(row[4] or 0) if row else 0.0,
        }
    except Exception as exc:
        logging.warning("get_barangay_features direct DB failed: %s", exc)
        response = (
            get_supabase_client()
            .table("barangay_training_data")
            .select("rainfall,humidity,soil,flood,storm_surge,timestamp,id")
            .eq("barangay_id", barangay_id)
            .order("timestamp", desc=True, nullsfirst=False)
            .order("id", desc=True)
            .limit(1)
            .execute()
        )
        row = response.data[0] if response.data else {}
        return {
            "rainfall": float(row.get("rainfall") or 0),
            "humidity": float(row.get("humidity") or 0),
            "soil": float(row.get("soil") or 0),
            "flood": float(row.get("flood") or 0),
            "storm_surge": float(row.get("storm_surge") or 0),
        }
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()


def get_barangay_hazard_profile(barangay_id: int) -> dict:
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT
                flood_hazard_level,
                flood_hazard_score,
                max_ssa_level,
                storm_surge_score,
                overall_hazard
            FROM barangay_hazard_profile
            WHERE barangay_id = %s
            LIMIT 1
        """, (barangay_id,))
        row = cur.fetchone()
        if not row:
            return _default_hazard_profile()
        return {
            "flood_hazard_level": row[0],
            "flood_hazard_score": float(row[1] or 0),
            "max_ssa_level": int(row[2] or 0),
            "storm_surge_score": float(row[3] or 0),
            "overall_hazard": row[4] or "LOW",
        }
    except Exception as exc:
        logging.warning("get_barangay_hazard_profile direct DB failed: %s", exc)
        response = (
            get_supabase_client()
            .table("barangay_hazard_profile")
            .select("flood_hazard_level,flood_hazard_score,max_ssa_level,storm_surge_score,overall_hazard")
            .eq("barangay_id", barangay_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            return _default_hazard_profile()
        row = response.data[0]
        return {
            "flood_hazard_level": row.get("flood_hazard_level") or "Unknown",
            "flood_hazard_score": float(row.get("flood_hazard_score") or 0),
            "max_ssa_level": int(row.get("max_ssa_level") or 0),
            "storm_surge_score": float(row.get("storm_surge_score") or 0),
            "overall_hazard": row.get("overall_hazard") or "LOW",
        }
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()


def get_recent_weather(barangay_id: int, limit: int = 24) -> list:
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT timestamp, rainfall, humidity, soil, flood, storm_surge
            FROM barangay_training_data
            WHERE barangay_id = %s
            ORDER BY timestamp DESC NULLS LAST, id DESC
            LIMIT %s
        """, (barangay_id, limit))
        rows = cur.fetchall()
        return [
            {
                "timestamp": row[0].isoformat() if hasattr(row[0], "isoformat") else row[0],
                "rainfall": float(row[1] or 0),
                "humidity": float(row[2] or 0),
                "soil": float(row[3] or 0),
                "flood": float(row[4] or 0),
                "storm_surge": float(row[5] or 0),
            }
            for row in rows
        ]
    except Exception as exc:
        logging.warning("get_recent_weather direct DB failed: %s", exc)
        response = (
            get_supabase_client()
            .table("barangay_training_data")
            .select("timestamp,rainfall,humidity,soil,flood,storm_surge,id")
            .eq("barangay_id", barangay_id)
            .order("timestamp", desc=True, nullsfirst=False)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        return [
            {
                "timestamp": row.get("timestamp"),
                "rainfall": float(row.get("rainfall") or 0),
                "humidity": float(row.get("humidity") or 0),
                "soil": float(row.get("soil") or 0),
                "flood": float(row.get("flood") or 0),
                "storm_surge": float(row.get("storm_surge") or 0),
            }
            for row in response.data
        ]
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()


def list_barangay_profiles() -> list:
    client = get_supabase_client()
    barangays = (
        client.table("barangay_list")
        .select("barangay_id,name,lat,lon")
        .order("barangay_id")
        .execute()
        .data
    )
    profiles = (
        client.table("barangay_hazard_profile")
        .select("barangay_id,flood_hazard_level,flood_hazard_score,max_ssa_level,storm_surge_score,overall_hazard")
        .execute()
        .data
    )
    profile_by_id = {row["barangay_id"]: row for row in profiles}
    results = []
    for barangay in barangays:
        profile = profile_by_id.get(barangay["barangay_id"], {})
        results.append({
            "barangay_id": barangay["barangay_id"],
            "name": barangay.get("name"),
            "lat": float(barangay.get("lat") or 0),
            "lon": float(barangay.get("lon") or 0),
            "flood_hazard_level": profile.get("flood_hazard_level"),
            "flood_hazard_score": float(profile.get("flood_hazard_score") or 0),
            "max_ssa_level": int(profile.get("max_ssa_level") or 0),
            "storm_surge_score": float(profile.get("storm_surge_score") or 0),
            "overall_hazard": profile.get("overall_hazard"),
        })
    return results


def get_hazard_summary() -> dict:
    rows = (
        get_supabase_client()
        .table("barangay_hazard_profile")
        .select("overall_hazard")
        .execute()
        .data
    )
    counts = {}
    for row in rows:
        key = row.get("overall_hazard") or "Unknown"
        counts[key] = counts.get(key, 0) + 1
    return {
        "summary": [
            {"overall_hazard": key, "count": count}
            for key, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ],
        "total_barangays": len(rows),
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
