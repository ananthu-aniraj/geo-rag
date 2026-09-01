# Step 1b: Timestamp Standardization & Climate Mapping

This document describes the design and operation of `standardize_timestamps.py`, which normalizes inconsistent datetimes, performs spatial geographic intersections, and maps Köppen-Geiger climate zones.

---

## ⚙️ Core Operation

The script normalizes inconsistent date/time strings and Unix epochs across diverse platforms into standardized ISO 8601 strings (`YYYY-MM-DDTHH:MM:SSZ`).

---

## 🌎 Geographical Region & Climate Mapping

To prepare coordinates for downstream analysis, visualization, and VLM prompts, the script maps spatial characteristics:

### 1. Köppen-Geiger Climate Mapping
If a Köppen-Geiger climate classification TIF map is provided (via `--koppen_tif` or parsed from `params.yaml`), coordinates are queried to assign each image a climate code (e.g. `BWh` for Hot Desert, `Csa` for Hot-Summer Mediterranean). The script utilizes `rasterio` and `pyproj` to dynamically handle raster projections.

### 2. Boundary Snapping (Country & Continent)
If an administrative land boundary shapefile is provided (via `--land_shp`), the script performs a spatial polygon intersection join on the coordinates.

To optimize performance and handle boundary limitations, the script implements:

* **H3-Cell Centroid Aggregation**: Coordinates are grouped into H3 Resolution 8 parent cells prior to joining. This reduces the number of geometric boundary tests by **$1000\times$**, mapping thousands of coordinates in a single check.
* **Nearest-Land Snapping (88.8 km Buffer)**: Coordinates located slightly offshore (e.g., beaches, piers, bridges) or on coastal H3 cells whose centroids lie in the water often miss land polygons. The script projects coordinate coordinates into a Web Mercator projection system (`EPSG:3857`) and snaps cells to the nearest country within an **$88,800$ meter** (approx. $0.8^{\circ}$) radius. Points beyond this buffer are classified as `"Ocean / Unknown"`.

---

## 📊 Derived Categorizations

### 1. Time of Day
Captured hours are grouped into standard blocks:

* **Dawn**: 05:00 – 08:00
* **Morning**: 08:00 – 12:00
* **Afternoon**: 12:00 – 17:00
* **Dusk**: 17:00 – 20:00
* **Night**: 20:00 – 05:00

### 2. Seasons (Climate-Aware Zoning)
Rather than using standard temperate astronomical calendars everywhere, seasons are mapped based on climate characteristics:

* **Desert Climates (BWh, BWk)**: Fixed to `"Dry Season"` year-round.
* **Tropical Savanna & Monsoon (Aw, Am)**: Wet season is defined as June–September in the Northern Hemisphere and November–April in the Southern Hemisphere.
* **Mediterranean (Csa, Csb)**: Wet season is December–February in the Northern Hemisphere and June–August in the Southern Hemisphere.
* **Temperate & Polar Fallbacks**: Mapped to standard astronomical seasons (Spring, Summer, Autumn, Winter) based on hemisphere and month spans.
