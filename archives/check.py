import geopandas as gpd
from pathlib import Path

GIS_DIR = Path(r"C:\Users\josua\OneDrive\Desktop\GIS_DATA")

bgy = gpd.read_file(GIS_DIR / "mamburao_pdrrmo.gpkg")
print("=== BARANGAY COLUMNS ===")
print(bgy.columns.tolist())
for col in bgy.columns:
    if col != "geometry":
        print(f"  {col}: {bgy[col].unique()[:10]}")