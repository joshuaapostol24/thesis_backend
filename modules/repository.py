from sqlalchemy import text
from .database import engine


def get_barangay_features(barangay_id: int):
    query = text("""
        SELECT rainfall, humidity, soil, flood, storm_surge
        FROM barangay_training_data
        WHERE barangay_id = :id
        ORDER BY timestamp DESC NULLS LAST, id DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"id": barangay_id})
        row = result.fetchone()

    if not row:
        return {
            "rainfall": 0,
            "humidity": 0,
            "soil": 0,
            "flood": 0,
            "storm_surge": 0,
        }

    return {
        "rainfall": row.rainfall,
        "humidity": row.humidity,
        "soil": row.soil,
        "flood": row.flood,
        "storm_surge": row.storm_surge,
    }
