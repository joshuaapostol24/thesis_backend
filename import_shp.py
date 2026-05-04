import geopandas as gpd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:123apostol@127.0.0.1:5432/thesis_db"
engine = create_engine(DATABASE_URL)

base_path = r"C:\Users\josua\OneDrive\Desktop\DATA_SET\storm_sursge"

# Load all 3 SSA levels
ssa1 = gpd.read_file(f"{base_path}\\OccidentalMindoro_StormSurge_SSA1.shp")
ssa1["ssa_level"] = 1

ssa2 = gpd.read_file(f"{base_path}\\OccidentalMindoro_StormSurge_SSA2.shp")
ssa2["ssa_level"] = 2

ssa3 = gpd.read_file(f"{base_path}\\OccidentalMindoro_StormSurge_SSA3.shp")
ssa3["ssa_level"] = 3

import pandas as pd
combined = pd.concat([ssa1, ssa2, ssa3], ignore_index=True)

print("Columns:", combined.columns.tolist())
print("Total rows:", len(combined))
print(combined.head())

combined.to_postgis("storm_surge_zones", engine, if_exists="replace", index=False)
print("✅ Done! Table 'storm_surge_zones' created in thesis_db")