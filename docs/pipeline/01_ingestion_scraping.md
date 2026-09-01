# Ingestion & Scraping Utilities

This document details the first phase of the Geo-RAG pipeline: raw data ingestion, scraping, and spatial target filtering.

---

## 📊 Ingested Data Sources

The pipeline processes diverse geotagged image sources, unifies their schemas, and resolves timezone inconsistencies:

| Data Source | Content Type | Location / Spatial Metadata | Timestamp & Timezone Characteristics |
|---|---|---|---|
| **Flickr** | Geotagged outdoor/scenic photographs | Latitude / Longitude coordinates | Naive local datetimes (date-taken) |
| **Mapillary** | Sequential street-level imagery | Resolution 11 H3 cells & coordinates | Naive local datetimes (`datetime_local`) or UTC epoch |
| **KartaView** | Crowdsourced street-level imagery | Coordinates & H3 cells (Global Streetscapes) | Standardized local datetimes (`datetime_local`) |
| **Wikimedia Commons** | Educational/illustrative media (landscapes, flora/fauna) | Latitude / Longitude coordinates | Naive or standardized timestamps (`extmetadata`) |
| **iWildCam** | Camera trap wildlife observations | Station coordinates | Naive local timestamps |
| **iNaturalist** | Species observations | Coordinates (filtered for sky/macro) | Standardized observation datetimes |

> [!NOTE]
> **Timezone Standardization**: Because naive local time strings are parsed with `utc=True` during ingestion, the numeric hour digit is preserved exactly as it was captured (e.g. 10:00 PM local is stored as `22:00:00Z`). This allows accurate local Time of Day profiling without external timezone database queries.

---

## 📡 Scraping Utilities

Prior to entering the main data engineering pipeline, raw data is harvested using targeted scraper tools.

### 1. Flickr Scrapers
* **`src/scrapers/flickr_density_profiler.py`**: Scrapes Flickr outdoor photos for targeted cities/landmarks. Supports geocoding location names via the Nominatim API, automatically padding landmark coordinate boxes, and generating 5km grids with stratified image limits per box.
* **`src/scrapers/flickr_5km_grid_search.py`**: Performs a global grid search to gather representative outdoor photos across the globe. Features a land mask to skip ocean areas.
* **`scripts/scrapers/run_flickr_scraper.sh`**: A pipeline runner that loops through chunks sequentially to execute the global grid search, loading configuration parameters from `config/scrapers/flickr_scraper.yaml` and API credentials from `.env`.
* **`scripts/scrapers/run_flickr_density_profiler.sh`**: Helper shell wrapper to launch `flickr_density_profiler.py` for specific location queries, loading defaults from `config/scrapers/flickr_profiler.yaml` and credentials from `.env`.

### 2. Mapillary Scrapers
* **`src/scrapers/mapillary_density_profiler.py`**: Scrapes street-level coordinate tracks for targeted cities/landmarks. Supports geocoding, landmark padding, and stratified grid-box limits.
* **`src/scrapers/mapillary_scraper.py`**: Standard batch grid scraper for Mapillary.
* **`scripts/scrapers/run_mapillary_scraper.sh`**: Orchestrates sequential batch street view scraping over grid chunks, loading configuration from `config/scrapers/mapillary_scraper.yaml` and credentials from `.env`.
* **`scripts/scrapers/run_mapillary_density_profiler.sh`**: Helper shell wrapper to launch `mapillary_density_profiler.py` for specific location queries, loading defaults from `config/scrapers/mapillary_profiler.yaml` and credentials from `.env`.

### 3. iNaturalist Scrapers
* **`src/scrapers/fetch_inaturalist_data.py`**: Connects to the iNaturalist API to download species observations.
* **`scripts/scrapers/run_inaturalist_scrapers.sh`**: A batch script that loops over specific countries/regions, loading execution settings and the region list from `config/scrapers/inaturalist_scraper.yaml`.
* **`scripts/scrapers/run_inaturalist_presets.sh`**: Ingests species observations using predefined biome presets, loading parameters from `config/scrapers/inaturalist_presets.yaml`.

### 4. OpenStreetMap Boundary Scrapers
* **`src/scrapers/osm_polygon_scraper.py`**: Scrapes geotagged files inside defined boundaries exclusively from **KartaView** (extracting timestamps, track parameters, and licenses, defaulting to CC BY-SA 4.0). Partitioning uses an optimized **3.5km x 3.5km grid** to stay within KartaView's strict server-enforced $0.04^{\circ}$ bounding box limit per request without recursive splitting.
* **`scripts/scrapers/run_osm_scraper.sh`**: Orchestrates sequential batch scraping of the boundary grid chunks, loading configuration from `config/scrapers/osm_scraper.yaml`.

---

## ⚙️ Configuration & Secrets Management

To keep credentials secure and make runs reproducible, configurations are decoupled from code and wrapper runner files:

### 1. API Credentials (`.env`)
All sensitive credentials must be set in a `.env` file in the repository root (copied from `.env.template`):

* `FLICKR_API_KEY`: Sourced by Flickr scrapers and profilers.
* `MAPILLARY_TOKEN`: Sourced by Mapillary scrapers and profilers.

The scraper shell scripts automatically source `.env` at startup to export these variables to the runtime environment.

### 2. Scraper Parameters (`config/scrapers/`)
Scraping parameters (limits, chunks, directories, bounding boxes, target regions) are defined in dedicated YAML files under `config/scrapers/`:

* **`flickr_scraper.yaml` / `mapillary_scraper.yaml`**: Grid search settings.
* **`flickr_profiler.yaml` / `mapillary_profiler.yaml`**: Density profiling boundaries and locations.
* **`inaturalist_scraper.yaml` / `inaturalist_presets.yaml`**: Regions, biome presets, and observations quotas.
* **`osm_scraper.yaml`**: Boundary scraping modes and OSM relations.

These configurations are read dynamically at run-time, and can still be overridden using command-line arguments.

---

## 🗺️ Uncovered Area Detection (Spatial Filtering)

To optimize global random search and avoid querying coordinates that already contain dense image coverage, the pipeline uses a spatial difference mask:

* **`src/utils/create_uncovered_land_areas_shp.py`**: Reads existing image CSV datasets, aggregates coordinate points into H3 cells at a user-defined resolution (default: resolution 5), and flags cells as "covered" if they exceed an image count threshold (default: 0). It then converts the covered H3 cells to polygon geometry and subtracts them from a standard global land mass shapefile (e.g., Natural Earth admin borders).
* **Usage in Search**: The resulting output shapefile (`shapefiles/uncovered_land_areas_test.shp`) represents land areas that are still poorly mapped. The global grid searchers (like `src/scrapers/flickr_5km_grid_search.py`) load this shapefile at startup and perform a fast spatial R-tree index check. They skip querying any grid box that does not intersect an uncovered land polygon, which reduces API requests and concentrates scraping efforts on data-poor zones.
