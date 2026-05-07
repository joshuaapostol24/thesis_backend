import csv
import psycopg2

LOCAL_DB = "postgresql://postgres:123apostol@127.0.0.1:5432/thesis_db"
OUTPUT_CSV = "barangay_training_data.csv"

WEIGHTS = {
    "rainfall": 0.35,
    "humidity": 0.15,
    "soil": 0.20,
    "flood": 0.20,
    "storm_surge": 0.10
}

PDF_WEIGHTED_MEANS = {
    1: {"rainfall":1.52,"soil":1.21,"flood":1.18},
    2: {"rainfall":1.48,"soil":1.16,"flood":1.05},
    3: {"rainfall":1.61,"soil":1.34,"flood":1.42},
    4: {"rainfall":1.43,"soil":1.08,"flood":0.92},
    5: {"rainfall":1.50,"soil":1.19,"flood":1.10},
    6: {"rainfall":1.46,"soil":1.12,"flood":0.98},
    7: {"rainfall":1.54,"soil":1.25,"flood":1.22},
    8: {"rainfall":1.66,"soil":1.49,"flood":1.78},
    9: {"rainfall":1.65,"soil":1.47,"flood":1.74},
    10: {"rainfall":1.63,"soil":1.44,"flood":1.69},
    11: {"rainfall":1.62,"soil":1.43,"flood":1.66},
    12: {"rainfall":1.60,"soil":1.39,"flood":1.58},
    13: {"rainfall":1.58,"soil":1.35,"flood":1.49},
    14: {"rainfall":1.57,"soil":1.33,"flood":1.44},
    15: {"rainfall":1.56,"soil":1.31,"flood":1.39},
}

SEASON_MULTIPLIER = {"Wet Season":1.20, "Summer":1.00, "Dry Season":0.85}
RAINFALL_MULTIPLIER = {"Heavy Rain":1.30, "Moderate Rain":1.10, "Light Rain":1.00, "No Rain":0.90}


def normalize(v, mn, mx):
    if mx == mn:
        return 0.0
    return max(0.0, min(1.0, (v-mn)/(mx-mn)))


def pdf_sensitivity(bid, key):
    raw = PDF_WEIGHTED_MEANS[bid].get(key,1.5)
    return 0.85 + ((raw-1.0)*0.30)


def compute_soil(humidity, rainfall, season, bid):
    base = (humidity/100.0)*2.0
    rain_effect = min(1.0, rainfall/20.0)
    season_mod = 1.1 if season == "Wet Season" else 0.9
    sens = pdf_sensitivity(bid, "soil")
    return round(min(3.0, (base + rain_effect)*season_mod*sens),4)


def compute_label(rainfall, humidity, soil, flood, surge, season, raincat, bid):
    r = normalize(rainfall,0,40) * pdf_sensitivity(bid,"rainfall")
    h = normalize(humidity,0,100)
    s = normalize(soil,0,3) * pdf_sensitivity(bid,"soil")
    f = normalize(flood,0,1) * pdf_sensitivity(bid,"flood")
    ss = normalize(surge,0,3)

    base = (
        r*WEIGHTS["rainfall"] +
        h*WEIGHTS["humidity"] +
        s*WEIGHTS["soil"] +
        f*WEIGHTS["flood"] +
        ss*WEIGHTS["storm_surge"]
    )

    season_amp = SEASON_MULTIPLIER.get(season,1.0)
    rain_amp = RAINFALL_MULTIPLIER.get(raincat,1.0)

    label = base * season_amp * rain_amp * 3.0
    return round(min(3.0,max(0.0,label)),4)


def load_weather_rows():
    conn = psycopg2.connect(LOCAL_DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, humidity, rainfall, season, rainfall_category
        FROM weather_data
        ORDER BY timestamp
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def load_barangay_profiles():
    conn = psycopg2.connect(LOCAL_DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT barangay_id, flood_hazard_score, storm_surge_score
        FROM barangay_hazard_profile
        ORDER BY barangay_id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    profiles = {}
    for bid, flood, surge in rows:
        profiles[bid] = {
            "flood": float(flood),
            "surge": float(surge)
        }
    return profiles


def generate_training_csv():
    weather_rows = load_weather_rows()
    profiles = load_barangay_profiles()

    total = low = mod = high = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "barangay_id","timestamp","rainfall","humidity",
            "soil","flood","storm_surge","risk_label"
        ])

        for row in weather_rows:
            timestamp, humidity, rainfall, season, raincat = row

            for bid in range(1,16):
                flood = profiles[bid]["flood"]
                surge = profiles[bid]["surge"]

                soil = compute_soil(humidity, rainfall, season, bid)
                label = compute_label(rainfall, humidity, soil, flood, surge, season, raincat, bid)

                writer.writerow([
                    bid, timestamp, rainfall, humidity,
                    soil, flood, surge, label
                ])

                total += 1
                if label < 1.0:
                    low += 1
                elif label < 2.0:
                    mod += 1
                else:
                    high += 1

    print("DONE GENERATING:", OUTPUT_CSV)
    print("TOTAL:", total)
    print("LOW:", low)
    print("MOD:", mod)
    print("HIGH:", high)


if __name__ == "__main__":
    generate_training_csv()