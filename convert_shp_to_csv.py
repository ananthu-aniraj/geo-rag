import os
import sys
import argparse
import time
import pandas as pd
import geopandas as gpd


def main():
    parser = argparse.ArgumentParser(
        description="Convert Mapillary shapefiles to standard Geo-RAG pipeline CSV database."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/aaniraj/Documents/Projects/data/Data/Data/Locations/Paved.shp",
        help="Path to the source shapefile (.shp) file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="mapillary_paved.csv",
        help="Path to save the resulting CSV file.",
    )
    args = parser.parse_args()

    print("================================================================================")
    print("🌍 SHAPEFILE TO GEORAG CSV CONVERTER")
    print("================================================================================")

    if not os.path.exists(args.input):
        print(f"Error: Shapefile not found at '{args.input}'")
        sys.exit(1)

    print(f"Loading shapefile: {args.input}...")
    t0 = time.time()
    try:
        gdf = gpd.read_file(args.input)
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f" -> Successfully loaded {len(gdf):,} geometries in {elapsed:.2f} seconds.")

    # 1. Resolve columns (handle slightly different shapefile naming conventions)
    # Check for photo ID column
    id_col = None
    for possible_name in ["id", "image_id", "photo_id"]:
        if possible_name in gdf.columns:
            id_col = possible_name
            break

    if not id_col:
        print(f"Error: Could not find an ID column in the shapefile. Available columns: {gdf.columns.tolist()}")
        sys.exit(1)

    print(f" -> Mapping ID column: '{id_col}'")

    # Check coordinates columns (fall back to extracting from geometry if not explicit)
    lat_col = "lat" if "lat" in gdf.columns else None
    lon_col = "lon" if "lon" in gdf.columns else None

    if not lat_col or not lon_col:
        print(" -> Extracting coordinates directly from Point geometries...")
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        lat_col = "lat"
        lon_col = "lon"

    # Surface type attribute
    surface_val = gdf["surface"] if "surface" in gdf.columns else "unknown"

    # 2. Build standard pipeline DataFrame
    print(" -> Formatting columns to Geo-RAG schema...")
    df = pd.DataFrame({
        "Photo_ID": gdf[id_col].astype(str),
        "Platform": "Mapillary",
        "Latitude": gdf[lat_col],
        "Longitude": gdf[lon_col],
        "Image_URL": "https://www.mapillary.com/app/?pKey=" + gdf[id_col].astype(str),
        "Captured_At": None,
        "Surface_Type": surface_val
    })

    # 3. Save
    print(f"Saving output to: {args.out}...")
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.out, index=False)
    print("================================================================================")
    print(f"🎉 Conversion complete! Standardized CSV saved with {len(df):,} rows.")
    print("================================================================================")


if __name__ == "__main__":
    main()
