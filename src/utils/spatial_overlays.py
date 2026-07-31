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
        transformer = Transformer.from_crs("epsg:4326", target_crs, always_axis_order=True)
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
        
    mapping = dynamic_mapping if dynamic_mapping is not None else EUNIS_ECOSYSTEM_MAPPING

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
