"""
gis_sync.py
─────────────────────────────────────────────────────────────────────────────
QGIS → Supabase sync for THESIS_SYSTEM
Reads exported QGIS GeoPackage layers and populates:
  - flood_hazard_zones     (id, hazard_level, hazard_score, geometry)
  - storm_surge_zones      (id, surge_type, HAZ, geometry)
  - barangay_hazard_profile (per-barangay flood + storm surge scores)

Storm surge note:
  Your GIS data has a single "Prone" zone with no SSA levels.
  Scoring is binary: inside prone zone = 1.0, outside = 0.0.
  SSA columns (in_ssa1, in_ssa2, in_ssa3, max_ssa_level) are set
  based on spatial intersection with the prone zone.

Usage:
  1. Export QGIS layers as GeoPackage to GIS_DATA folder
  2. Add Supabase credentials to .env
  3. Run: python gis_sync.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── CONFIG ────────────────────────────────────────────────────────────────────
GIS_DIR = Path(r"C:\Users\josua\OneDrive\Desktop\GIS_DATA")

FLOOD_GPKG      = GIS_DIR / "flood_pdrrmo.gpkg"
STORMSURGE_GPKG = GIS_DIR / "storm_surge_pdrrmo.gpkg"
BARANGAY_GPKG   = GIS_DIR / "mamburao_pdrrmo.gpkg"
HOUSEHOLD_XLSX  = GIS_DIR / "Mamburao Coordinates.xlsx"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

TARGET_CRS = "EPSG:4326"

# Flood column = "Flood", values = High/Moderate/Low
# Matches normalization.py: High=1.0, Moderate/Medium=0.60, Low=0.20
FLOOD_LEVEL_SCORE = {
    "High":     1.00,
    "Moderate": 0.60,
    "Medium":   0.60,
    "Low":      0.20,
    "None":     0.00,
}

# Storm surge is binary — "Prone" zone only, no SSA levels
# Treated as SSA3 equivalent (maximum exposure) since no level data available
SURGE_SCORE = 1.0

# Barangay name mapping: GIS layer name → system name + barangay_id
# Handles "Brgy." prefix and name mismatches
BARANGAY_NAME_MAP = {
    "Brgy. Tayamaan":  {"id": 7,  "name": "Tayamaan"},
    "Brgy. Talabaan":  {"id": 5,  "name": "Talabaan"},
    "Brgy. Tangkalan": {"id": 6,  "name": "Tangkalan"},
    "Poblacion 1":     {"id": 8,  "name": "Poblacion 1"},
    "Poblacion 2":     {"id": 9,  "name": "Poblacion 2"},
    "Poblacion 3":     {"id": 10, "name": "Poblacion 3"},
    "Poblacion 4":     {"id": 11, "name": "Poblacion 4"},
    "Poblacion 5":     {"id": 12, "name": "Poblacion 5"},
    "Poblacion 6":     {"id": 13, "name": "Poblacion 6"},
    "Poblacion 7":     {"id": 14, "name": "Poblacion 7"},
    "Poblacion 8":     {"id": 15, "name": "Poblacion 8"},
    "Poblacion 9":     {"id": 15, "name": "Poblacion 8"},  # alias
}

# Barangays NOT in GIS layer — use known coordinates + fallback scores
MISSING_BARANGAYS = {
    1:  {"name": "Balansay",  "lat": 13.207799, "lon": 120.640617},
    2:  {"name": "Fatima",    "lat": 13.172760, "lon": 120.664584},
    3:  {"name": "Payompon",  "lat": 13.219053, "lon": 120.602478},
    4:  {"name": "San Luis",  "lat": 13.210000, "lon": 120.650000},
}

# All 15 barangay coordinates (used for centroid fallback)
ALL_BARANGAYS = {
    1:  {"name": "Balansay",    "lat": 13.207799, "lon": 120.640617},
    2:  {"name": "Fatima",      "lat": 13.172760, "lon": 120.664584},
    3:  {"name": "Payompon",    "lat": 13.219053, "lon": 120.602478},
    4:  {"name": "San Luis",    "lat": 13.210000, "lon": 120.650000},
    5:  {"name": "Talabaan",    "lat": 13.144559, "lon": 120.684837},
    6:  {"name": "Tangkalan",   "lat": 13.271995, "lon": 120.629050},
    7:  {"name": "Tayamaan",    "lat": 13.229811, "lon": 120.565745},
    8:  {"name": "Poblacion 1", "lat": 13.223767, "lon": 120.596847},
    9:  {"name": "Poblacion 2", "lat": 13.221329, "lon": 120.594450},
    10: {"name": "Poblacion 3", "lat": 13.224442, "lon": 120.595053},
    11: {"name": "Poblacion 4", "lat": 13.224109, "lon": 120.593705},
    12: {"name": "Poblacion 5", "lat": 13.223584, "lon": 120.592086},
    13: {"name": "Poblacion 6", "lat": 13.226094, "lon": 120.590585},
    14: {"name": "Poblacion 7", "lat": 13.223684, "lon": 120.588975},
    15: {"name": "Poblacion 8", "lat": 13.228705, "lon": 120.585884},
}
# ─────────────────────────────────────────────────────────────────────────────


def load_gpkg(path: Path, label: str) -> gpd.GeoDataFrame:
    """Load a GeoPackage and reproject to WGS84."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        logger.info("Reprojecting %s → WGS84", label)
        gdf = gdf.to_crs(TARGET_CRS)
    logger.info("✅ Loaded %s: %d features", label, len(gdf))
    return gdf


def sync_flood_hazard_zones(flood_gdf: gpd.GeoDataFrame, sb) -> None:
    """Upload flood hazard polygons to flood_hazard_zones table."""
    logger.info("── Syncing flood_hazard_zones ──")

    # Delete existing rows first to avoid duplicates
    sb.table("flood_hazard_zones").delete().neq("id", 0).execute()

    records = []
    for idx, row in flood_gdf.iterrows():
        raw_level    = str(row.get("Flood", "Low")).strip().capitalize()
        hazard_score = FLOOD_LEVEL_SCORE.get(raw_level, 0.20)
        wkt          = row.geometry.wkt

        records.append({
            "hazard_level": raw_level,
            "hazard_score": hazard_score,
            "geometry":     wkt,
        })

    logger.info("Inserting %d flood hazard zone records…", len(records))

    chunk_size = 50
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        sb.table("flood_hazard_zones").insert(chunk).execute()
        logger.info("  Inserted rows %d–%d", i + 1, i + len(chunk))

    logger.info("✅ flood_hazard_zones sync complete: %d records", len(records))


def sync_storm_surge_zones(surge_gdf: gpd.GeoDataFrame, sb) -> None:
    """
    Upload storm surge polygons to storm_surge_zones table.
    Since data only has 'Prone' zone (no SSA levels),
    treat entire zone as maximum exposure (score=1.0).
    """
    logger.info("── Syncing storm_surge_zones ──")

    sb.table("storm_surge_zones").delete().neq("id", 0).execute()

    records = []
    for idx, row in surge_gdf.iterrows():
        wkt = row.geometry.wkt
        records.append({
                "haz":        SURGE_SCORE,      # ← lowercase
                "surge_type": "Prone",
                "geometry":   wkt,
        })

    logger.info("Inserting %d storm surge zone records…", len(records))

    chunk_size = 50
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        sb.table("storm_surge_zones").insert(chunk).execute()
        logger.info("  Inserted rows %d–%d", i + 1, i + len(chunk))

    logger.info("✅ storm_surge_zones sync complete: %d records", len(records))


def compute_barangay_hazard_profiles(
    flood_gdf: gpd.GeoDataFrame,
    surge_gdf: gpd.GeoDataFrame,
) -> list:
    """
    Spatial join: for each of the 15 barangays, compute:
      - flood_hazard_level and flood_hazard_score (from flood zones)
      - storm_surge_score (binary: 1.0 if in prone zone, 0.0 if not)
      - in_ssa1/2/3, max_ssa_level (derived from storm surge presence)
      - overall_hazard (HIGH/MODERATE/LOW based on combined score)

    For barangays missing from GIS layer, uses known coordinates
    for point-in-polygon lookup.
    """
    logger.info("── Computing per-barangay hazard profiles ──")

    profiles = []

    for bid, bgy in ALL_BARANGAYS.items():
        lat = bgy["lat"]
        lon = bgy["lon"]
        pt  = Point(lon, lat)
        pt_gdf = gpd.GeoDataFrame(
            geometry=[pt], crs="EPSG:4326"
        )

        # ── Flood hazard lookup ───────────────────────────────────────────
        flood_match = flood_gdf[flood_gdf.geometry.contains(pt)]
        if flood_match.empty:
            # Nearest polygon fallback
            flood_gdf2 = flood_gdf.copy().to_crs("EPSG:3857")
            pt_proj    = pt_gdf.to_crs("EPSG:3857").geometry.iloc[0]
            flood_gdf2["_dist"] = flood_gdf2.geometry.distance(pt_proj)
            nearest    = flood_gdf2.loc[flood_gdf2["_dist"].idxmin()]
            flood_level = str(nearest.get("Flood", "Low")).strip().capitalize()
        else:
            flood_level = str(
                flood_match.iloc[0].get("Flood", "Low")
            ).strip().capitalize()

        flood_score = FLOOD_LEVEL_SCORE.get(flood_level, 0.20)

        # ── Storm surge lookup ────────────────────────────────────────────
        surge_match   = surge_gdf[surge_gdf.geometry.contains(pt)]
        in_surge_zone = not surge_match.empty
        surge_score   = SURGE_SCORE if in_surge_zone else 0.0

        # Since no SSA levels, treat prone zone as SSA3 equivalent
        in_ssa1    = in_surge_zone
        in_ssa2    = in_surge_zone
        in_ssa3    = in_surge_zone
        max_ssa    = 3 if in_surge_zone else 0

        # ── Overall hazard (matches original formula) ─────────────────────
        combined = (flood_score * 0.6) + (surge_score * 0.4)
        if combined >= 0.7:
            overall = "HIGH"
        elif combined >= 0.4:
            overall = "MODERATE"
        else:
            overall = "LOW"

        profiles.append({
            "barangay_id":        bid,
            "barangay_name":      bgy["name"],
            "lat":                lat,
            "lon":                lon,
            "flood_hazard_level": flood_level,
            "flood_hazard_score": flood_score,
            "in_ssa1":            in_ssa1,
            "in_ssa2":            in_ssa2,
            "in_ssa3":            in_ssa3,
            "max_ssa_level":      max_ssa,
            "storm_surge_score":  surge_score,
            "overall_hazard":     overall,
        })

        logger.info(
            "  Barangay %02d %-12s | flood=%-8s (%.2f) | "
            "surge=%s (%.2f) | overall=%s",
            bid, bgy["name"], flood_level, flood_score,
            "YES" if in_surge_zone else "NO",
            surge_score, overall
        )

    return profiles


def upload_barangay_profiles(profiles: list, sb) -> None:
    """Upsert all 15 barangay hazard profiles to Supabase."""
    logger.info("── Uploading barangay_hazard_profile ──")
    for profile in profiles:
        sb.table("barangay_hazard_profile").upsert(profile).execute()
    logger.info(
        "✅ barangay_hazard_profile updated for %d barangays",
        len(profiles)
    )


def analyze_household_exposure(
    flood_gdf: gpd.GeoDataFrame,
    surge_gdf: gpd.GeoDataFrame,
) -> None:
    """
    Loads household coordinates from Excel and reports
    how many fall within flood and storm surge zones.
    """
    if not HOUSEHOLD_XLSX.exists():
        logger.warning("Household file not found — skipping exposure analysis")
        return

    try:
        df = pd.read_excel(HOUSEHOLD_XLSX)
        household_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326"
        )
        logger.info("Loaded %d household points", len(household_gdf))

        flood_join = gpd.sjoin(
            household_gdf, flood_gdf,
            predicate="within", how="left"
        )
        surge_join = gpd.sjoin(
            household_gdf, surge_gdf,
            predicate="within", how="left"
        )

        flood_exposed = flood_join["index_right"].notna().sum()
        surge_exposed = surge_join["index_right"].notna().sum()

        logger.info(
            "Household exposure | flood=%d / %d | surge=%d / %d",
            flood_exposed, len(household_gdf),
            surge_exposed, len(household_gdf)
        )
    except Exception as e:
        logger.error("Household exposure analysis failed: %s", e)


def print_summary(profiles: list) -> None:
    print("\n" + "═" * 65)
    print("  GIS SYNC SUMMARY — Barangay Hazard Profiles")
    print("═" * 65)
    print(f"  {'ID':<4} {'Barangay':<14} {'Flood':<10} {'Score':<7} {'Surge':<6} {'Overall'}")
    print("─" * 65)
    for p in profiles:
        print(
            f"  {p['barangay_id']:<4} "
            f"{p['barangay_name']:<14} "
            f"{p['flood_hazard_level']:<10} "
            f"{p['flood_hazard_score']:<7.2f} "
            f"{'YES' if p['storm_surge_score'] > 0 else 'NO':<6} "
            f"{p['overall_hazard']}"
        )
    print("═" * 65)
    high     = sum(1 for p in profiles if p["overall_hazard"] == "HIGH")
    moderate = sum(1 for p in profiles if p["overall_hazard"] == "MODERATE")
    low      = sum(1 for p in profiles if p["overall_hazard"] == "LOW")
    print(f"  HIGH: {high}  MODERATE: {moderate}  LOW: {low}")
    print("═" * 65 + "\n")


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError(
            "Missing SUPABASE_URL or SUPABASE_KEY in .env file.\n"
            "Add them to your .env (same one used by THESIS_SYSTEM)."
        )

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("✅ Connected to Supabase")

    # Load layers
    flood_gdf = load_gpkg(FLOOD_GPKG,      "Flood Hazard")
    surge_gdf = load_gpkg(STORMSURGE_GPKG, "Storm Surge")

    # Sync spatial tables
    sync_flood_hazard_zones(flood_gdf, sb)
    sync_storm_surge_zones(surge_gdf, sb)

    # Compute and upload per-barangay profiles
    profiles = compute_barangay_hazard_profiles(flood_gdf, surge_gdf)
    upload_barangay_profiles(profiles, sb)
    print_summary(profiles)

    # Household exposure analysis (optional)
    analyze_household_exposure(flood_gdf, surge_gdf)

    logger.info("🎉 GIS sync complete!")


if __name__ == "__main__":
    main()