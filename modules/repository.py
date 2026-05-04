from sqlalchemy import text
from .database import engine


def get_barangay_features(barangay_id: int):
    query = text("""
        SELECT rainfall, flood
        FROM barangay_weather
        WHERE barangay_id = :id
        ORDER BY timestamp DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"id": barangay_id})
        row = result.fetchone()

    if not row:
        return {
            "rainfall": 0,
            "flood": 0,
        }

    return {
        "rainfall": row.rainfall,
        "flood": row.flood,
    }