from pyproj import Transformer

from src.utils.eunis_env_zones_class_mappings import (
    ENV_ZONES_MAPPING,
    EUNIS_ECOSYSTEM_MAPPING,
)

# Global cache for pyproj transformers to prevent recreation overhead
_transformer_cache = {}


def get_crs_transformer(target_crs):
    """
    Returns a pyproj Transformer from WGS84 (EPSG:4326) to the target CRS.
    Handles 'always_axis_order' keyword argument dynamically for backward compatibility.

    Returns:
        (transformer, has_axis_order) - tuple of Transformer object and axis-order support flag.
    """
    cache_key = str(target_crs)
    if cache_key in _transformer_cache:
        return _transformer_cache[cache_key]

    has_axis_order = True
    try:
        transformer = Transformer.from_crs(
            "epsg:4326", target_crs, always_axis_order=True
        )
    except TypeError:
        # Fallback for older pyproj versions lacking always_axis_order
        transformer = Transformer.from_crs("epsg:4326", target_crs)
        has_axis_order = False

    _transformer_cache[cache_key] = (transformer, has_axis_order)
    return transformer, has_axis_order


def lookup_raster_pixel(lat, lon, r_ds, transformer, has_axis_order):
    """
    Projects WGS84 coordinates and samples the pixel value from an open rasterio dataset.

    Returns:
        The pixel value (int or float) or None if sampling fails.
    """
    if lat is None or lon is None or (lat == 0.0 and lon == 0.0):
        return None
    try:
        if has_axis_order:
            x, y = transformer.transform(lon, lat)
        else:
            # Older pyproj versions expect coordinates in the default EPSG:4326 order: (lat, lon)
            x, y = transformer.transform(lat, lon)

        return list(r_ds.sample([(x, y)]))[0][0]
    except Exception:
        return None


def get_eunis_label(pixel_val, dynamic_mapping=None):
    """
    Maps a raw pixel value from the EUNIS ecosystem raster to its human-readable class label.
    """
    if pixel_val is None:
        return "Unknown"

    mapping = (
        dynamic_mapping if dynamic_mapping is not None else EUNIS_ECOSYSTEM_MAPPING
    )

    # Try direct key lookup
    if pixel_val in mapping:
        return mapping[pixel_val]

    # Try string conversion lookups
    val_str = str(pixel_val).strip()
    if val_str in mapping:
        return mapping[val_str]

    try:
        val_int = int(float(val_str))
        if val_int in mapping:
            return mapping[val_int]
    except ValueError:
        pass

    return "Unknown"


def get_environmental_zone_label(pixel_val):
    """
    Maps a raw pixel value from the Metzger Environmental Zones raster to its class label.
    """
    if pixel_val is None:
        return "Unknown"

    if pixel_val in ENV_ZONES_MAPPING:
        return ENV_ZONES_MAPPING[pixel_val]

    try:
        val_int = int(float(str(pixel_val).strip()))
        if val_int in ENV_ZONES_MAPPING:
            return ENV_ZONES_MAPPING[val_int]
    except ValueError:
        pass

    return "Unknown"


def load_eunis_legend(raster_path):
    """
    Loads EUNIS Level 3 legend mappings from 'eunis_legend_detailed.csv'
    located in the same directory as the raster.

    Returns:
        dict: A mapping from Id (int) -> dict of level names: 'eunis_l1', 'eunis_l2', 'eunis_l3'
    """
    import os

    import pandas as pd

    legend_path = os.path.join(
        os.path.dirname(raster_path), "eunis_legend_detailed.csv"
    )
    if not os.path.exists(legend_path):
        legend_path = os.path.join(
            os.path.dirname(os.path.dirname(raster_path)), "eunis_legend_detailed.csv"
        )

    legend_df = pd.read_csv(legend_path)
    legend_mapping = {}
    for _, row in legend_df.iterrows():
        try:
            rid = int(row["Id"])
            l1_name = (
                str(row["EUNIS1_name"]).strip()
                if pd.notna(row["EUNIS1_name"])
                else str(row["EUNIS1"]).strip()
            )

            l2_name = (
                str(row["EUNIS2_name"]).strip()
                if pd.notna(row["EUNIS2_name"])
                else l1_name
            )
            if not l2_name or l2_name.lower() == "nan":
                l2_name = l1_name

            l3_name = (
                str(row["EUNIS3_name"]).strip()
                if pd.notna(row["EUNIS3_name"])
                else l2_name
            )
            if not l3_name or l3_name.lower() == "nan":
                l3_name = l2_name

            legend_mapping[rid] = {
                "eunis_l1": l1_name,
                "eunis_l2": l2_name,
                "eunis_l3": l3_name,
            }
        except Exception:
            continue
    return legend_mapping


def lookup_environmental_zone(lat, lon, r_ds, transformer, has_axis_order):
    """
    Looks up the Environmental Zone label for a given coordinate.

    Returns:
        str: Environmental Zone name, or "" if not found/invalid.
    """
    pixel_val = lookup_raster_pixel(lat, lon, r_ds, transformer, has_axis_order)
    if pixel_val is None or pixel_val == r_ds.nodata or pixel_val <= 0:
        return ""
    label = get_environmental_zone_label(pixel_val)
    return label if label != "Unknown" else ""


def lookup_eunis_levels(lat, lon, r_ds, transformer, has_axis_order, legend_mapping):
    """
    Looks up the EUNIS Level 1, 2, and 3 classifications for a coordinate.

    Returns:
        dict: {'eunis_l1': ..., 'eunis_l2': ..., 'eunis_l3': ...} or None if not found/invalid.
    """
    pixel_val = lookup_raster_pixel(lat, lon, r_ds, transformer, has_axis_order)
    if (
        pixel_val is None
        or pixel_val == r_ds.nodata
        or pixel_val <= 0
        or pixel_val not in legend_mapping
    ):
        return None
    return legend_mapping[pixel_val]
