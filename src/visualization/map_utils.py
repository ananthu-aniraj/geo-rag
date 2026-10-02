"""Map utilities for Folium and Leaflet visualizations.

Handles CARTO basemap configurations, automatic API key injection from
environment variables or .env files, and wrappers around Folium maps to
prevent 'API key required' watermarks on Carto basemaps.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

import folium

CARTO_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
)

# Standard CARTO raster basemap styles mapping
CARTO_STYLE_VARIANTS = {
    "cartodb positron": "light_all",
    "cartodbpositron": "light_all",
    "positron": "light_all",
    "light_all": "light_all",
    "cartodb dark_matter": "dark_all",
    "cartodbdark_matter": "dark_all",
    "dark_matter": "dark_all",
    "dark_all": "dark_all",
    "cartodb voyager": "rastertiles/voyager",
    "cartodbvoyager": "rastertiles/voyager",
    "voyager": "rastertiles/voyager",
    "rastertiles/voyager": "rastertiles/voyager",
}


def get_carto_api_key(search_dir: Path | str | None = None) -> str:
    """Retrieve the CARTO API key from environment variables or .env file.

    Args:
        search_dir: Optional directory or file path to search for .env first.

    Returns:
        The CARTO API key as a string, or empty string if not found.
    """
    if search_dir is not None:
        target_env = (
            Path(search_dir) / ".env" if Path(search_dir).is_dir() else Path(search_dir)
        )
        if target_env.is_file():
            try:
                with open(target_env, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        if k.strip() == "CARTO_API_KEY":
                            val = v.strip().strip('"').strip("'")
                            if val:
                                return val
            except Exception:
                pass

    api_key = os.environ.get("CARTO_API_KEY", "").strip().strip('"').strip("'")
    if api_key:
        return api_key

    # Try python-dotenv if available
    try:
        from dotenv import find_dotenv, load_dotenv

        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path)
            api_key = os.environ.get("CARTO_API_KEY", "").strip().strip('"').strip("'")
            if api_key:
                return api_key
    except ImportError:
        pass

    # Fallback: manual walk-up search for .env
    search_dirs = [Path.cwd(), Path(__file__).resolve().parent]
    visited = set()
    for start_dir in search_dirs:
        curr = start_dir
        while curr not in visited and curr != curr.parent:
            visited.add(curr)
            env_file = curr / ".env"
            if env_file.is_file():
                try:
                    with open(env_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("#") or "=" not in line:
                                continue
                            k, v = line.split("=", 1)
                            if k.strip() == "CARTO_API_KEY":
                                val = v.strip().strip('"').strip("'")
                                if val:
                                    os.environ["CARTO_API_KEY"] = val
                                    return val
                except Exception:
                    pass
            curr = curr.parent

    return ""


def get_carto_key_param(api_key: str | None = None) -> str:
    """Return query parameter '?key=<api_key>' if CARTO API key is found, else ''.

    Args:
        api_key: Optional explicit API key. If None, retrieved via get_carto_api_key().

    Returns:
        Query parameter string like '?key=xxx' or '' if no key is present.
    """
    key = api_key if api_key is not None else get_carto_api_key()
    key = key.strip() if key else ""
    return f"?key={key}" if key else ""


def get_carto_tile_config(
    variant: str = "light_all",
    api_key: str | None = None,
) -> Tuple[str, str]:
    """Return the tile URL template and attribution for CARTO raster basemaps.

    Args:
        variant: Style name or variant (e.g. 'CartoDB Positron', 'light_all',
                 'CartoDB Dark_Matter', 'voyager').
        api_key: Optional explicit API key. If not provided, retrieved from env.

    Returns:
        Tuple of (tile_url, attribution).
    """
    normalized = variant.strip().lower()
    path_variant = CARTO_STYLE_VARIANTS.get(normalized, variant)

    key_param = get_carto_key_param(api_key)
    tile_url = f"https://{{s}}.basemaps.cartocdn.com/{path_variant}/{{z}}/{{x}}/{{y}}{{r}}.png{key_param}"
    return tile_url, CARTO_ATTRIBUTION


def is_carto_tiles(tiles: Any) -> bool:
    """Check if the given tile specification corresponds to CARTO basemaps."""
    if not isinstance(tiles, str):
        return False
    tiles_lower = tiles.strip().lower()
    return (
        tiles_lower in CARTO_STYLE_VARIANTS
        or "cartodb" in tiles_lower
        or "cartocdn" in tiles_lower
    )


def create_folium_map(
    location: Any = None,
    zoom_start: int = 2,
    tiles: Any = "CartoDB Positron",
    **kwargs: Any,
) -> folium.Map:
    """Create a Folium Map with CARTO basemap API key injection.

    If CARTO tiles are specified (default: 'CartoDB Positron'), this ensures
    the tile URL contains the CARTO API key (if available) and the required
    CARTO / OSM attribution to remove the 'API key required' watermark.

    If tiles is not a CARTO style, standard Folium Map initialization is used.
    """
    api_key = kwargs.pop("carto_api_key", None)
    if is_carto_tiles(tiles):
        key = api_key if api_key is not None else get_carto_api_key()
        if key:
            tile_url, default_attr = get_carto_tile_config(variant=tiles, api_key=key)
            kwargs.setdefault("attr", default_attr)
            return folium.Map(
                location=location, zoom_start=zoom_start, tiles=tile_url, **kwargs
            )
        else:
            try:
                return folium.Map(
                    location=location, zoom_start=zoom_start, tiles=tiles, **kwargs
                )
            except ValueError:
                tile_url, default_attr = get_carto_tile_config(
                    variant=tiles, api_key=None
                )
                kwargs.setdefault("attr", default_attr)
                return folium.Map(
                    location=location, zoom_start=zoom_start, tiles=tile_url, **kwargs
                )

    return folium.Map(location=location, zoom_start=zoom_start, tiles=tiles, **kwargs)


def create_carto_tile_layer(
    variant: str = "light_all",
    api_key: str | None = None,
    **kwargs: Any,
) -> folium.TileLayer:
    """Create a Folium TileLayer configured with CARTO tiles and API key."""
    url, attr = get_carto_tile_config(variant=variant, api_key=api_key)
    kwargs.setdefault("attr", attr)
    name = kwargs.pop("name", f"CARTO {variant}")
    return folium.TileLayer(tiles=url, name=name, **kwargs)
