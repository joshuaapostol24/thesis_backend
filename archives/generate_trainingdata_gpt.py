import pandas as pd

# =========================================================
# LOAD WEATHER DATA
# =========================================================

df = pd.read_csv(r"C:\Users\josua\OneDrive\Desktop\DATA_SET\weather_part_1.csv")

# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna(
    subset=[
        "rainfall",
        "humidity"
    ]
)

df["rainfall"] = pd.to_numeric(
    df["rainfall"],
    errors="coerce"
)

df["humidity"] = pd.to_numeric(
    df["humidity"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "rainfall",
        "humidity"
    ]
)

# =========================================================
# SORT DESCENDING BY RAINFALL
# =========================================================

df = df.sort_values(
    by="rainfall",
    ascending=False
).reset_index(drop=True)

# =========================================================
# SELECT 30 HIGH / 30 MODERATE / 30 LOW
# =========================================================

high_df = df.head(30).copy()

mid = len(df) // 2

moderate_df = df.iloc[
    mid - 15: mid + 15
].copy()

low_df = df.tail(30).copy()

# =========================================================
# ASSIGN LABELS
# =========================================================

high_df["risk_label"] = 3

moderate_df["risk_label"] = 2

low_df["risk_label"] = 1

# =========================================================
# COMBINE 90 WEATHER ROWS
# =========================================================

selected_df = pd.concat([
    high_df,
    moderate_df,
    low_df
]).reset_index(drop=True)

# =========================================================
# BARANGAY GIS / SHP FEATURES
# =========================================================

BARANGAY_FEATURES = {

    1:  {"soil": 2.07, "flood": 1.80, "storm_surge": 4.20},
    2:  {"soil": 2.03, "flood": 1.80, "storm_surge": 4.20},
    3:  {"soil": 2.10, "flood": 1.80, "storm_surge": 4.20},
    4:  {"soil": 2.22, "flood": 1.80, "storm_surge": 1.00},
    5:  {"soil": 2.20, "flood": 1.80, "storm_surge": 4.20},
    6:  {"soil": 2.18, "flood": 3.20, "storm_surge": 1.00},
    7:  {"soil": 2.17, "flood": 3.20, "storm_surge": 4.20},
    8:  {"soil": 2.14, "flood": 1.80, "storm_surge": 4.20},
    9:  {"soil": 2.12, "flood": 1.80, "storm_surge": 4.20},
    10: {"soil": 2.09, "flood": 1.80, "storm_surge": 4.20},
    11: {"soil": 2.72, "flood": 1.80, "storm_surge": 4.20},
    12: {"soil": 1.96, "flood": 1.80, "storm_surge": 1.00},
    13: {"soil": 2.01, "flood": 1.80, "storm_surge": 4.20},
    14: {"soil": 2.47, "flood": 3.20, "storm_surge": 1.00},
    15: {"soil": 2.67, "flood": 3.20, "storm_surge": 4.20},
}

# =========================================================
# GENERATE TRAINING DATA
# =========================================================

rows = []

for barangay_id, features in BARANGAY_FEATURES.items():

    for _, row in selected_df.iterrows():

        rows.append({

            "barangay_id": barangay_id,

            "timestamp": row.get(
                "timestamp",
                None
            ),

            "rainfall": float(row["rainfall"]),

            "humidity": float(row["humidity"]),

            "soil": features["soil"],

            "flood": features["flood"],

            "storm_surge": features["storm_surge"],

            "risk_label": int(row["risk_label"])
        })

# =========================================================
# FINAL DATAFRAME
# =========================================================

final_df = pd.DataFrame(rows)

# =========================================================
# EXPORT
# =========================================================

final_df.to_csv(
    "barangay_training_data.csv",
    index=False
)

print("\nTraining data generated successfully.")
print(f"Total rows: {len(final_df)}")
print(final_df.head())