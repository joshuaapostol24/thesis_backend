"""
upload_shapefiles_to_supabase.py
─────────────────────────────────
Uploads flood hazard and storm surge shapefiles
as spatial data into Supabase PostGIS tables
"""

import geopandas as gpd
import time
from supabase import create_client

from modules.config import get_supabase_key, get_supabase_url

# ── Supabase config ───────────────────────────────────────────────────────────
SUPABASE_URL = get_supabase_url()
SUPABASE_KEY = get_supabase_key()

# ── Shapefile paths ───────────────────────────────────────────────────────────
FLOOD_SHP = r"C:\Users\josua\OneDrive\Desktop\DATA_SET\Mindoro_FH_5yr.shp"
SSA1_SHP  = r"C:\Users\josua\OneDrive\Desktop\DATA_SET\storm_sursge\OccidentalMindoro_StormSurge_SSA1.shp"
SSA2_SHP  = r"C:\Users\josua\OneDrive\Desktop\DATA_SET\storm_sursge\OccidentalMindoro_StormSurge_SSA2.shp"
SSA3_SHP  = r"C:\Users\josua\OneDrive\Desktop\DATA_SET\storm_sursge\OccidentalMindoro_StormSurge_SSA3.shp"

FLOOD_MAP   = {1.0: 'Low', 2.0: 'Medium', 3.0: 'High'}
FLOOD_SCORE = {1.0: 0.2,   2.0: 0.6,      3.0: 1.0}
SSA_SCORE   = {1: 0.33,    2: 0.66,        3: 1.00}

BATCH_SIZE  = 10


def load_shp(path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def upload_in_batches(supabase, table, records):
    total = len(records)
    for i in range(0, total, BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        supabase.table(table).insert(batch).execute()
        print(f"  Uploaded rows {i+1} to {min(i+BATCH_SIZE, total)} / {total}")
        time.sleep(0.5)


print("Connecting to Supabase...")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Upload flood hazard zones ─────────────────────────────────────────────────
print("\nLoading flood hazard shapefile...")
flood_gdf = load_shp(FLOOD_SHP)
print(f"Found {len(flood_gdf)} flood hazard polygons")

flood_records = []
for _, row in flood_gdf.iterrows():
    var = float(row.get('Var', 1.0))
    flood_records.append({
        'hazard_level': FLOOD_MAP.get(var, 'Low'),
        'hazard_score': FLOOD_SCORE.get(var, 0.2),
        'geometry':     row.geometry.wkt
    })

print("Uploading flood hazard zones...")
upload_in_batches(supabase, "flood_hazard_zones", flood_records)
print(f"✅ Flood hazard zones uploaded! ({len(flood_records)} polygons)")

# ── Upload storm surge zones ──────────────────────────────────────────────────
for ssa_level, shp_path in [(1, SSA1_SHP), (2, SSA2_SHP), (3, SSA3_SHP)]:
    print(f"\nLoading SSA{ssa_level} shapefile...")
    gdf = load_shp(shp_path)
    print(f"Found {len(gdf)} SSA{ssa_level} polygons")

    records = []
    for _, row in gdf.iterrows():
        records.append({
            'ssa_level': ssa_level,
            'ssa_score': SSA_SCORE[ssa_level],
            'geometry':  row.geometry.wkt
        })

    print(f"Uploading SSA{ssa_level} zones...")
    upload_in_batches(supabase, "storm_surge_zones", records)
    print(f"✅ SSA{ssa_level} zones uploaded! ({len(records)} polygons)")

print("\n✅ All shapefiles uploaded to Supabase!")
