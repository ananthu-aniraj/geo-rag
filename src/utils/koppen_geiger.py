import os

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

# Mappings linking numeric values in the maps to the Köppen-Geiger classes
KOPPEN_LEGEND = {
    1: {"code": "Af", "desc": "Tropical, rainforest"},
    2: {"code": "Am", "desc": "Tropical, monsoon"},
    3: {"code": "Aw", "desc": "Tropical, savannah"},
    4: {"code": "BWh", "desc": "Arid, desert, hot"},
    5: {"code": "BWk", "desc": "Arid, desert, cold"},
    6: {"code": "BSh", "desc": "Arid, steppe, hot"},
    7: {"code": "BSk", "desc": "Arid, steppe, cold"},
    8: {"code": "Csa", "desc": "Temperate, dry summer, hot summer"},
    9: {"code": "Csb", "desc": "Temperate, dry summer, warm summer"},
    10: {"code": "Csc", "desc": "Temperate, dry summer, cold summer"},
    11: {"code": "Cwa", "desc": "Temperate, dry winter, hot summer"},
    12: {"code": "Cwb", "desc": "Temperate, dry winter, warm summer"},
    13: {"code": "Cwc", "desc": "Temperate, dry winter, cold summer"},
    14: {"code": "Cfa", "desc": "Temperate, no dry season, hot summer"},
    15: {"code": "Cfb", "desc": "Temperate, no dry season, warm summer"},
    16: {"code": "Cfc", "desc": "Temperate, no dry season, cold summer"},
    17: {"code": "Dsa", "desc": "Cold, dry summer, hot summer"},
    18: {"code": "Dsb", "desc": "Cold, dry summer, warm summer"},
    19: {"code": "Dsc", "desc": "Cold, dry summer, cold summer"},
    20: {"code": "Dsd", "desc": "Cold, dry summer, very cold winter"},
    21: {"code": "Dwa", "desc": "Cold, dry winter, hot summer"},
    22: {"code": "Dwb", "desc": "Cold, dry winter, warm summer"},
    23: {"code": "Dwc", "desc": "Cold, dry winter, cold summer"},
    24: {"code": "Dwd", "desc": "Cold, dry winter, very cold winter"},
    25: {"code": "Dfa", "desc": "Cold, no dry season, hot summer"},
    26: {"code": "Dfb", "desc": "Cold, no dry season, warm summer"},
    27: {"code": "Dfc", "desc": "Cold, no dry season, cold summer"},
    28: {"code": "Dfd", "desc": "Cold, no dry season, very cold winter"},
    29: {"code": "ET", "desc": "Polar, tundra"},
    30: {"code": "EF", "desc": "Polar, frost"},
}


def extract_koppen_geiger(df, tif_path):
    """
    Extracts Köppen-Geiger climate codes and descriptions for coordinates in a DataFrame.
    Expects 'Latitude' and 'Longitude' columns to be present in df.
    """
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        print(
            "[WARNING] 'Latitude' and/or 'Longitude' columns missing. Skipping Köppen-Geiger extraction."
        )
        df["Koppen_Code"] = None
        df["Koppen_Desc"] = None
        return df

    if not tif_path or not os.path.exists(tif_path):
        print(
            f"[WARNING] Köppen-Geiger GeoTIFF file not found at '{tif_path}'. Skipping climate classification."
        )
        df["Koppen_Code"] = None
        df["Koppen_Desc"] = None
        return df

    # Parse coordinates to numeric values safely
    lons = pd.to_numeric(df["Longitude"], errors="coerce").values
    lats = pd.to_numeric(df["Latitude"], errors="coerce").values

    valid_coords = []
    valid_indices = []

    for idx, (lon, lat) in tqdm(
        enumerate(zip(lons, lats)),
        total=len(df),
        desc="Validating coordinates for Köppen-Geiger extraction",
    ):
        if (
            not np.isnan(lon)
            and not np.isnan(lat)
            and -180 <= lon <= 180
            and -90 <= lat <= 90
        ):
            valid_coords.append((lon, lat))
            valid_indices.append(idx)

    # Initialize output columns
    codes = [None] * len(df)
    descriptions = [None] * len(df)

    if valid_coords:
        try:
            with rasterio.open(tif_path) as src:
                # Read the entire band into memory (extremely fast categorical map read)
                band_data = src.read(1)

                # Perform vectorized coordinate to pixel row/col lookup
                valid_lons = [c[0] for c in valid_coords]
                valid_lats = [c[1] for c in valid_coords]
                rows, cols = rasterio.transform.rowcol(
                    src.transform, valid_lons, valid_lats
                )

                for orig_idx, row, col in tqdm(
                    zip(valid_indices, rows, cols),
                    total=len(valid_indices),
                    desc="Extracting Köppen-Geiger codes",
                ):
                    if 0 <= row < band_data.shape[0] and 0 <= col < band_data.shape[1]:
                        val_int = int(band_data[row, col])
                        if val_int in KOPPEN_LEGEND:
                            codes[orig_idx] = KOPPEN_LEGEND[val_int]["code"]
                            descriptions[orig_idx] = KOPPEN_LEGEND[val_int]["desc"]
        except Exception as e:
            print(f"[ERROR] Failed to read Köppen-Geiger GeoTIFF: {e}")

    df["Koppen_Code"] = codes
    df["Koppen_Desc"] = descriptions
    return df
