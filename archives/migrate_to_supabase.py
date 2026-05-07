"""
categorize_barangays.py
───────────────────────
1. Loads all shapefiles (flood hazard + 3 storm surge levels)
2. For each of the 15 Mamburao barangays, checks which zones they fall in
3. Uploads results to Supabase via REST API (no direct DB connection needed)
"""

import geopandas as gpd
import pandas as pd
from supabase import create_client
from shapely.geometry import Point

from modules.config import get_supabase_key, get_supabase_url

# ── Supabase config ───────────────────────────────────────────────────────────
SUPABASE_URL = get_supabase_url()
SUPABASE_KEY = get_supabase_key()

# ── Barangay coordinates (Mamburao) ──────────────────────────────────────────
BARANGAYS = [
    (1,  'Balansay',      13.207799, 120.640617),
    (2,  'Fatima',        13.172760, 120.664584),
    (3,  'Payompon',      13.219053, 120.602478),
    (4,  'San Luis',      13.210000, 120.650000),
    (5,  'Talabaan',      13.144559, 120.684837),
    (6,  'Tangkalan',     13.271995, 120.629050),
    (7,  'Tayamaan',      13.229811, 120.565745),
    (8,  'Poblacion 1',   13.223767, 120.596847),
    (9,  'Poblacion 2',   13.221329, 120.594450),
    (10, 'Poblacion 3',   13.224442, 120.595053),
    (11, 'Poblacion 4',   13.224109, 120.593705),
    (12, 'Poblacion 5',   13.223584, 120.592086),
    (13, 'Poblacion 6',   13.226094, 120.590585),
    (14, 'Poblacion 7',   13.223684, 120.588975),
    (15, 'Poblacion 8',   13.228705, 120.585884),
]

# ── Shapefile paths ───────────────────────────────────────────────────────────
FLOOD_SHP = r"C:\Users\josua\OneDrive\Desktop\DATA_SET\Mindoro_FH_5yr.shp"
SSA1_SHP  = r"C:\Users\josua\OneDrive\Desktop\DATA_SET\storm_sursge\OccidentalMindoro_StormSurge_SSA1.shp"
SSA2_SHP  = r"C:\Users\josua\OneDrive\Desktop\DATA_SET\storm_sursge\OccidentalMindoro_StormSurge_SSA2.shp"
SSA3_SHP  = r"C:\Users\josua\OneDrive\Desktop\DATA_SET\storm_sursge\OccidentalMindoro_StormSurge_SSA3.shp"

FLOOD_MAP   = {1.0: 'Low', 2.0: 'Medium', 3.0: 'High'}
FLOOD_SCORE = {1.0: 0.2,   2.0: 0.6,      3.0: 1.0}


def load_shp(path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def point_in_gdf(lat, lon, gdf):
    pt = Point(lon, lat)
    matches = gdf[gdf.geometry.contains(pt)]
    if matches.empty:
        gdf2 = gdf.copy()
        gdf2 = gdf2.to_crs("EPSG:3857")          # ← fixes the distance warning
        pt_proj = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        gdf2['dist'] = gdf2.geometry.distance(pt_proj)
        return gdf.loc[gdf2['dist'].idxmin()], True
    return matches.iloc[0], False


# ── Load shapefiles ───────────────────────────────────────────────────────────
print("Loading shapefiles...")
flood_gdf = load_shp(FLOOD_SHP)
ssa1_gdf  = load_shp(SSA1_SHP)
ssa2_gdf  = load_shp(SSA2_SHP)
ssa3_gdf  = load_shp(SSA3_SHP)
print("All shapefiles loaded!")

# ── Categorize barangays ──────────────────────────────────────────────────────
print("\nCategorizing barangays...")
results = []

for bid, name, lat, lon in BARANGAYS:
    pt = Point(lon, lat)

    # Flood hazard
    flood_row, _ = point_in_gdf(lat, lon, flood_gdf)
    flood_var   = float(flood_row.get('Var', 1.0))
    flood_level = FLOOD_MAP.get(flood_var, 'Low')
    flood_score = FLOOD_SCORE.get(flood_var, 0.2)

    # Storm surge
    in_ssa1 = not ssa1_gdf[ssa1_gdf.geometry.contains(pt)].empty
    in_ssa2 = not ssa2_gdf[ssa2_gdf.geometry.contains(pt)].empty
    in_ssa3 = not ssa3_gdf[ssa3_gdf.geometry.contains(pt)].empty

    if in_ssa3:        ssa_level = 3; ssa_score = 1.00
    elif in_ssa2:      ssa_level = 2; ssa_score = 0.66
    elif in_ssa1:      ssa_level = 1; ssa_score = 0.33
    else:              ssa_level = 0; ssa_score = 0.00

    # Overall hazard
    combined = (flood_score * 0.6) + (ssa_score * 0.4)
    if combined >= 0.7:   overall = 'HIGH'
    elif combined >= 0.4: overall = 'MODERATE'
    else:                 overall = 'LOW'

    results.append({
        'barangay_id':        bid,
        'barangay_name':      name,
        'lat':                lat,
        'lon':                lon,
        'flood_hazard_level': flood_level,
        'flood_hazard_score': round(flood_score, 2),
        'in_ssa1':            in_ssa1,
        'in_ssa2':            in_ssa2,
        'in_ssa3':            in_ssa3,
        'max_ssa_level':      ssa_level,
        'storm_surge_score':  round(ssa_score, 2),
        'overall_hazard':     overall,
    })

    print(f"  {bid:02d} {name:15s} | Flood: {flood_level:6s} | SSA: {ssa_level} | Overall: {overall}")

df = pd.DataFrame(results)

print("\n=== Summary ===")
print("Flood Hazard:",  df['flood_hazard_level'].value_counts().to_dict())
print("Max SSA Level:", df['max_ssa_level'].value_counts().to_dict())
print("Overall Hazard:", df['overall_hazard'].value_counts().to_dict())

# Save CSV backup
df.to_csv('barangay_hazard_profile.csv', index=False)
print("\nSaved barangay_hazard_profile.csv!")

# ── Upload to Supabase via REST API ───────────────────────────────────────────
print("\nConnecting to Supabase...")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Convert booleans explicitly (supabase-py needs native Python bool)
records = []
for r in results:
    records.append({
        'barangay_id':        int(r['barangay_id']),
        'barangay_name':      r['barangay_name'],
        'lat':                float(r['lat']),
        'lon':                float(r['lon']),
        'flood_hazard_level': r['flood_hazard_level'],
        'flood_hazard_score': float(r['flood_hazard_score']),
        'in_ssa1':            bool(r['in_ssa1']),
        'in_ssa2':            bool(r['in_ssa2']),
        'in_ssa3':            bool(r['in_ssa3']),
        'max_ssa_level':      int(r['max_ssa_level']),
        'storm_surge_score':  float(r['storm_surge_score']),
        'overall_hazard':     r['overall_hazard'],
    })

print("Uploading barangay_hazard_profile...")
response = supabase.table("barangay_hazard_profile").upsert(records).execute()
print(f"Uploaded {len(response.data)} rows to barangay_hazard_profile!")

# ── Update barangay_list coordinates ─────────────────────────────────────────
print("\nUpdating barangay_list coordinates...")
for bid, name, lat, lon in BARANGAYS:
    supabase.table("barangay_list").update({
        'lat':  float(lat),
        'lon':  float(lon),
        'name': name
    }).eq('barangay_id', bid).execute()
    print(f"  Updated barangay_id={bid} {name}")

print("\nDone! All data uploaded to Supabase.")
