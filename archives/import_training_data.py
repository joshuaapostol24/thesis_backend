import pandas as pd
import psycopg2
import os

from modules.config import get_database_url

DATABASE_URL = get_database_url()

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

data_dir = r"C:\Users\josua\OneDrive\Desktop\DATA_SET\meta_data\data\synthetic"
for bid in range(1, 16):
    path = os.path.join(data_dir, f"barangay_{bid:02d}.csv")
    df = pd.read_csv(path)
    df["barangay_id"] = bid
    if "humidity" not in df.columns:
        df["humidity"] = 0.0
    if "storm_surge" not in df.columns:
        df["storm_surge"] = 0.0

    records = [
        (
            int(row.barangay_id),
            float(row.rainfall),
            float(row.humidity),
            float(row.soil),
            float(row.flood),
            float(row.storm_surge),
            float(row.risk_label),
        )
        for _, row in df.iterrows()
    ]

    cur.executemany("""
        INSERT INTO barangay_training_data
            (barangay_id, rainfall, humidity, soil, flood, storm_surge, risk_label)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, records)

    conn.commit()
    print(f"✅ Barangay {bid:02d} inserted — {len(records)} rows")

cur.close()
conn.close()
print("Done! All training data stored in database.")
