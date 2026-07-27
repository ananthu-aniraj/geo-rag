#  Data Engineering Documentation

This document provides a comprehensive guide to the data processing, clustering, and spatial-semantic mapping pipeline.

---

## 🗺️ 1. Pipeline Overview & Architecture

The pipeline is designed to ingest raw street-level and outdoor image databases (Flickr, Mapillary, camera traps), deduplicate them spatially, cluster them semantically using image embeddings, auto-label clusters using Multi-Modal LLMs (MLLMs), and build interactive web visualizations.

![Pipeline Flowchart](pipeline_flowchart_clean.jpg)

---

## 📊 2. Ingested Data Sources

The pipeline processes diverse geotagged image sources, unifies their schemas, and resolves timezone inconsistencies:

| Data Source | Content Type | Location / Spatial Metadata | Timestamp & Timezone Characteristics |
|---|---|---|---|
| **Flickr** | Geotagged outdoor/scenic photographs | Latitude / Longitude coordinates | Naive local datetimes (date-taken) |
| **Mapillary** | Sequential street-level imagery | Resolution 11 H3 cells & coordinates | Naive local datetimes (`datetime_local`) or UTC epoch |
| **KartaView** | Crowdsourced street-level imagery | Coordinates & H3 cells (Global Streetscapes) | Standardized local datetimes (`datetime_local`) |
| **iWildCam** | Camera trap wildlife observations | Station coordinates | Naive local timestamps |
| **iNaturalist** | Species observations | Coordinates (filtered for sky/macro) | Standardized observation datetimes |

> [!NOTE]
> **Timezone Standardization**: Because naive local time strings are parsed with `utc=True` during ingestion, the numeric hour digit is preserved exactly as it was captured (e.g. 10:00 PM local is stored as `22:00:00Z`). This allows accurate local Time of Day profiling without external timezone database queries.

### 📡 2.1 Ingestion & Scraping Utilities

Before raw data enters the main pipeline, a set of specialized scrapers are used to harvest geotagged imagery.

#### 1. Flickr Scrapers
* **`src/scrapers/flickr_density_profiler.py`**: Scrapes Flickr outdoor photos for targeted cities/landmarks. Supports geocoding location names via the Nominatim API, automatically padding landmark coordinate boxes, and generating 5km grids with stratified image limits per box.
* **`src/scrapers/flickr_5km_grid_search.py`**: Performs a global grid search to gather representative outdoor photos across the globe. Features a land mask to skip ocean areas.
* **`run_flickr_scraper.sh`**: A pipeline runner that loops through 10,000 randomized chunks sequentially to execute the global grid search, with crash-halting checks to ensure continuity.

#### 2. Mapillary Scrapers
* **`src/scrapers/mapillary_density_profiler.py`**: Scrapes street-level coordinate tracks for targeted cities/landmarks. Supports geocoding, landmark padding, and stratified grid-box limits.
* **`src/scrapers/mapillary_scraper.py`**: Standard batch grid scraper for Mapillary.
* **`run_mapillary_scraper.sh`**: Orchestrates sequential batch street view scraping over 10,000 randomized grid chunks with crash-halting checks.

#### 3. iNaturalist Scrapers
* **`src/scrapers/fetch_inaturalist_data.py`**: Connects to the iNaturalist API to download species observations.
* **`run_inaturalist_scrapers.sh`**: A batch script that loops over specific countries/regions (e.g., Angola, Alaska, Algeria), downloading balanced species distributions and optionally excluding flying fauna.
* **`run_inaturalist_presets.sh`**: Ingests species observations using predefined biome presets (e.g. `desert`, `tundra`, `wetland`, `boreal`, `rainforest`, `polar`).

#### 4. Uncovered Area Detection (Spatial Filtering)
To optimize global random search and avoid querying coordinates that already contain dense image coverage, the pipeline uses a spatial difference mask:
* **`src/utils/create_uncovered_land_areas_shp.py`**: Reads existing image CSV datasets, aggregates coordinate points into H3 cells at a user-defined resolution (default: resolution 5), and flags cells as "covered" if they exceed an image count threshold (default: 0). It then converts the covered H3 cells to polygon geometry and subtracts them from a standard global land mass shapefile (e.g., Natural Earth admin borders).
* **Usage in Search**: The resulting output shapefile (`shapefiles/uncovered_land_areas_test.shp`) represents land areas that are still poorly mapped. The global grid searchers (like `src/scrapers/flickr_5km_grid_search.py`) load this shapefile at startup and perform a fast spatial R-tree index check. They skip querying any grid box that does not intersect an uncovered land polygon, which reduces API requests and concentrates scraping efforts on data-poor zones.

---

## ⚙️ 3. Pipeline Walkthrough

### Step 1: Spatial Deduplication & Filtering
* **Script**: `src/processing/process_scraped_data.py`
* **Operation**: Groups coordinates into H3 Resolution 11 parent cells [Brodsky, 2018]. Performs spatial-temporal deduplication using TIPSv2 image embeddings [Cao, 2026] to ensure uniform geographic coverage.
* **Million-Row Streaming Optimization & Parallel Network Engine**: 
  To support processing datasets in the millions without Out-of-Memory (OOM) errors, the script dynamically identifies **active H3 cells** (cells containing new scraped images) and loads only their existing embeddings from the Parquet database using `pyarrow.dataset` (bypassing the other 99% in-memory). It then writes updates atomically using a custom `stream_update_parquet` streaming engine that filters and appends chunk-by-chunk. 
  To optimize ingestion speed and minimize network bottlenecks:
  * **HTTP Connection Pooling**: A thread-safe global `requests.Session` with a connection adapter (128 maximum connections) keeps sockets alive across download threads, eliminating TCP/SSL handshake latency.
  * **On-The-Fly Background Resizing**: Images are resized to `448x448` immediately inside background download threads before passing them to the main thread. This reduces the RAM footprint per image to ~602 KB, allowing you to safely scale the `--cell_chunk_size` (e.g. to `256` or `512` images in parallel per cell block) without OOM risks.
  * **Thread Concurrency & Batching**: Uses 64 concurrent download threads. The GPU batch size can be set via `--batch_size` (e.g., to `128` or `256` for modern GPUs) to maximize inference throughput.
  * **Streaming Output**: CSV metadata is exported using a streaming row-group writer to avoid buffering embeddings in memory.
* **Zero-Shot Noise Filters**:
  * **Flickr Indoor/Outdoor Filter**: By default, the script filters out indoor photos using zero-shot text-image classification with TIPSv2. Images are compared against the prompts *"An indoor scene"* and *"An outdoor landscape or street view"*. If an image matches the indoor class, it is discarded. This filter applies **only to Flickr images** (since street-view platforms like Mapillary/KartaView are intrinsically outdoor, and iNaturalist observations are filtered by macro characteristics). Can be bypassed with the `--no_filter` flag.
  * `--filter_macro`: Filters out macro close-up flora/fauna images (typically for iNaturalist data) using a zero-shot *"A macro/close-up photo of flowers, leaves, bark, or insects"* classifier.
  * `--filter_sky`: Filters out empty sky views (typically for iNaturalist data) using a zero-shot *"A view of empty sky, clouds, or flying objects"* classifier.

### Step 1b: Timestamp Standardization, Climate Zoning, and Geographical Region Mapping
* **Script**: `src/processing/standardize_timestamps.py`
* **Operation**: Normalizes inconsistent date/time strings and Unix epochs into ISO 8601 strings (`YYYY-MM-DDTHH:MM:SSZ`).
* **Köppen-Geiger Climate Mapping**: If a Köppen-Geiger climate classification [Beck et al., 2023] TIF map is provided (using the `--koppen_tif` argument), coordinates are queried to assign each image a climate code (e.g., `BWh`, `Aw`, `Csa`). The dataset can be downloaded from [here](https://www.gloh2o.org/koppen/). The script uses `rasterio` to read the TIF and `pyproj` to convert coordinates to the TIF's projection system.
* **Geographical Region Mapping (Country & Continent)**: If a land boundaries shapefile (e.g. Natural Earth administrative boundaries) is provided via the `--land_shp` parameter, the script performs a spatial polygon intersection join on the coordinates. Resolving these boundaries early ensures geographic metadata is preserved and available for downstream analysis, visualizations, and MLLM auto-labeling prompts. To optimize performance and ensure high-accuracy assignments:
  * **H3-Cell Aggregation**: Coordinates are grouped into H3 cell centroids (default: H3 resolution 8 parent cells) prior to the spatial join. This reduces the number of expensive geometry checks by $1000\times$, mapping thousands of images in a single call.
  * **Nearest-Land Snapping Fallback (88 km / 88.8k m Buffer)**: Coordinates located slightly offshore (e.g., harbors, beaches, piers, or bridges) often miss strict land shapefile polygons due to finite boundary resolution. Additionally, H3 centroid offsets can place land-based cells slightly inside the ocean. To prevent these coastal points from being misclassified as open ocean, the script uses a projected Web Mercator coordinate system (`EPSG:3857`) to perform a nearest-neighbor spatial join. If a land boundary lies within an $88.8$ km radius ($88,800$ meters or $\sim 0.8^{\circ}$), the cell is snapped to that host country. Points beyond this threshold are classified as `"Ocean / Unknown"`.
* **Categorizations Added**:
  * **Time of Day**: *Dawn* (05-08), *Morning* (08-12), *Afternoon* (12-17), *Dusk* (17-20), *Night* (20-05).
  * **Seasons (Climate-Aware Zoning)**:
    * **Desert Climates (BWh, BWk)**: Fixed to *"Dry Season"* year-round.
    * **Tropical Wet/Dry Savanna & Monsoon (Aw, Am)**: Wet season is defined as June–September in the Northern Hemisphere and November–April in the Southern Hemisphere.
    * **Mediterranean (Csa, Csb)**: Dry summer / wet winter. Wet season is Dec–Feb in the Northern Hemisphere and June–August in the Southern Hemisphere.
    * **Temperate & Polar Fallbacks**: Classified into standard astronomical seasons (*Spring*, *Summer*, *Autumn*, *Winter*) based on standard hemispheric month spans.

### Step 1c: Coordinate Anomaly Cleanup (GPS Glitch Removal)
* **Script**: `src/processing/cleanup_coordinate_anomalies.py`
* **Operation**: Scans the deduplicated database to safely purge locked-latitude coordinate lines caused by faulty contributor GPS units at source. Writes the clean data to independent output files (`geo_space_cleaned.parquet` / `geo_space_cleaned.csv`), leaving the raw deduplicated database untouched.
* **Type Resilience**: Both Parquet loading and CSV chunk writing perform defensive `pd.to_numeric` conversions on the coordinate columns before executing high-precision rounding operations, preventing float-string mixed schema exceptions.
* **Safety Criteria**: A rounded latitude parallel *L* (rounded to 5 decimal places, representing ~1.1 meters precision) is flagged and purged only if:

$$
\text{Count}(L) \gt 10 \quad \text{and} \quad \text{Longitude Span}(L) \gt 1.0^{\circ}
$$

*(A longitude span of > 1.0° is ~111 km, which ensures that dense cities—which naturally occupy tiny bounding boxes—are completely preserved, while global coordinate-locked lines spanning multiple countries are cleanly discarded).*

### Step 1d: Automatically Finding Optimal k (Spatial Validation)
* **Script**: `src/utils/validate_cluster_count.py`
* **Operation**: If `auto_find_k: true` is configured in `params.yaml`, the pipeline runs a spatial block validation search before clustering:
  * **Dynamic Parameter Update**: The script evaluates reconstruction losses across a search range (defined by `k_min`, `k_max`, and `k_step` in `params.yaml`), identifies the optimal cluster count $k$, and writes the value back to the `k_clusters` parameter in `params.yaml` to dynamically size the downstream clustering execution.
  * **Mathematical Methodology**: To prevent spatial data leakage, the script implements geographical block hold-outs and selects $k$ using the Elbow method. For the full mathematical details on Tobler's First Law, reconstruction loss, and splitting, see [Section 7: Determining the Optimal Cluster Count (*k*)](#-7-determining-the-optimal-cluster-count-k).

### Step 2: Global FAISS GPU Clustering & Dynamic Mode Selection
* **Scripts**: `src/indexing/cluster_images_global.py` & `src/utils/check_semantic_drift.py`
* **Automated Decision Heuristic**:
  The pipeline dynamically determines whether to perform full re-clustering and MLLM re-labeling (`fit` mode) or simply map new images to existing centroids (`assign` mode) using a scale-proof semantic drift detector:
  1. **Dynamic Drift Analysis (`check_semantic_drift.py`)**: If a clustered parquet database file already exists on disk for the current $k$, the pipeline samples 10,000 embeddings from the new dataset. It queries them against the old centroids using FAISS and measures their cosine similarities.
  2. **`fit` Mode (Re-cluster and Label)**: Triggered if no pre-existing clustered parquet database file exists, if the target $k$ has changed, or if **significant semantic drift is detected** (meaning more than 3% of the new images are classified as outliers, having a cosine similarity of $< 0.70$ with all existing centroids). It runs full FAISS Spherical K-Means and triggers VLM auto-labeling.
  3. **`assign` Mode (Map to Existing Centroids)**: Triggered if the pre-existing database file exists and the semantic distribution is stable (outliers $\le 3\%$).
     * **Zero VLM Cost:** Rather than re-clustering all data and re-running expensive MLLMs, it dynamically calculates the centroid coordinates from the existing Parquet database in memory, maps the new images to their nearest centroids using FAISS nearest-neighbor search, and maps existing VLM labels and descriptions to the new images. This executes in seconds.
* **Operation details**:
  * **Type Resilience**: Coordinate columns are defensively cast to numeric float64 right after loading to preserve schema alignment during final Parquet export.
  * **Fine-Grained Child Clustering**: Runs Spherical K-Means on GPU (`faiss.Kmeans(d, k, niter=20, spherical=True, gpu=True)`) on raw image embeddings to partition the data into *k* fine-grained child clusters.
  * **Hierarchical Parent Clustering**: Runs Spherical K-Means on GPU on normalized child centroids to group them into *k_parents* broader parent clusters (where *k_parents* = max(2, *k* / 80)).
  * **Immediate Persistence & RAM Release**: Saves cluster assignments and centroids directly to the clustered parquet database and immediately releases heavy embedding matrices from RAM to avoid CPU memory bottlenecks.

### Step 2b: Multi-Modal LLM Cluster Auto-Labeling
* **Script**: `src/indexing/label_clusters_mllm.py`
* **Operation**: Reads the clustered Parquet database (`geo_space_clustered.parquet`) and performs automated semantic labeling:
  * **Automated SGLang Server Lifecycle**: The master script `run_full_pipeline.sh` automatically manages the SGLang container (`sglang-server`) using Docker with NVIDIA runtime. It performs pre-launch cleanup to avoid port binding conflicts, executes health checks using `/health` and monitors for crashes, and automatically shuts down at completion to release GPU resources.
  * **MLLM / Zero-Shot Labeling**: Sends representative centroid samples for both child and parent clusters to a Multi-Modal LLM (e.g., Gemma-2 via SGLang or OpenAI API) to automatically generate descriptive semantic labels and structured LULC descriptions.
  * **Decoupled Execution**: Can be run independently, re-run with different MLLM models, or executed on a separate GPU/CPU instance without having to re-cluster image embeddings.

### Step 2c: Fallback Safety Check (Re-labeling Failed Clusters)
* **Script**: `src/indexing/relabel_failed_clusters.py`
* **Operation**: Scans the clustered dataset for any clusters whose initial MLLM labeling failed or timed out (labeled as `"Error Labeling"` or `"Unlabeled"`) and performs fallback depth retries up to `--fallback_depth 10` to guarantee 100% cluster label coverage.

### Step 2d: H3 Spatial-Semantic Indexing
* **Script**: `src/indexing/build_spatial_semantic_index.py`
* **Operation**: Aggregates the clustered dataset into a multi-resolution H3 spatial index [Brodsky, 2018], linking cells to dominant cluster categories, seasons, and times of day.

---

## 🤖 4. Multi-Modal LLM (MLLM) Auto-Labeling Prompt Strategy

To automatically label image clusters, the pipeline uses a decoupled **Two-Step Prompting Strategy** inside `src/indexing/label_clusters_mllm.py`. This split architecture enhances classification accuracy, minimizes category hallucination, and avoids processing heavy image tokens twice.

---

### 📷 Step 1: Multimodal Visual Description
* **Input**: The representative cluster image + `prompts/shared/prompt_step1.txt`
* **Output**: A detailed, objective paragraph describing the physical scene features (no LULC labels).

**Prompt Template (`prompts/shared/prompt_step1.txt`):**
```text
You are analyzing a single ground-level photograph from a global land-survey campaign; the sample point may be anywhere on Earth (tropical, arid, temperate, boreal, alpine, wetland, or coastal). The camera stands at a sample point and looks outward; the land cover AT THE FOREGROUND/CENTER of the frame is the DOMINANT cover that must be characterized. Background elements (distant trees, buildings, hills) are CONTEXT only and must never override the dominant foreground cover.

Carefully observe and describe the following components in a single, detailed, and cohesive paragraph:
1. image_medium: State clearly whether this is a real-world photograph, a painting/artwork, a map, an abstract drawing/illustration, a text graphic, or a close-up indoor sign/object.
2. dominant_cover: The single land-cover type occupying most of the foreground/center (e.g. cereal crop, grassland, bare soil, broadleaved trees, built surface, sand, snow/ice, water, wetland).
3. cover_fraction_estimate: Share of the visible ground area that is vegetation vs bare soil/rock vs sealed/built surface vs water/ice.
4. visible_evidence & salient_objects: Primary foreground objects, structures, architectural elements, vehicles, or natural formations visible. Describe them naturally as a human observer would.
5. vegetation_detail: Height, leaf/frond shape, seed-head/flower form, rows or not (cereals, broadleaf/industrial crops, grassland, woody cover broadleaved vs coniferous vs palm).
6. vegetation_condition & phenological_stage: Vigor (vigorous-green, senescent, dead/brown) and growth stage (emerging, full-canopy, harvested/stubble).
7. canopy_structure: Canopy closure (closed, open, scattered) and height (low shrub <2m, small trees 2-5m, mature trees >5m).
8. soil_surface_state & lithology: Exposed ground sealing (permeable, unsealed, sealed-impervious) and visibly exposed soil/rock texture and color.
9. structure_and_pattern: Terrain slope, parcel boundaries, linear features (crop rows, tramlines, ditches, tracks), or recent disturbances.
10. context_background: Elements visible in the distance or periphery (buildings, treelines, mountains, water bodies) that are not the sample point cover.
11. human_evidence & activities: Visible signs of human use (buildings, machinery, fences, planted rows, paths, irrigation, or a selfie where a person is visiting a landmark or hiking).

Do not assign a LULC category code or label. Focus purely on describing what is visually supported.
```

---

### 🗺️ Step 2: Geographical LULC Classification (Text-Only)
* **Input**: The visual description output from Step 1 + geographic/climatic metadata + the LULC category list + `prompts/shared/prompt_step2.txt`
* **Output**: The final structured classification label and a consolidated description paragraph.

**Prompt Template (`prompts/shared/prompt_step2.txt`):**
```text
You are a geographical and ecological classifier. Your task is to classify the land cover/land use of a location based on a visual description of the scene and geographic/climatic metadata.

[METADATA CONTEXT]
Location coordinates/bounding box: {location}
Region/Country: {country}
Continent: {continent}
Time of Day: {time_of_day}
Season: {season}
Köppen-Geiger Climate Code: {koppen_code} ({koppen_desc})

[VISUAL DESCRIPTION]
{visual_description}

[CLASSIFICATION RULES]
1. Use the climate code and season to resolve ambiguities (e.g. distinguishing tropical savanna from temperate grassland, or identifying agricultural crops vs natural vegetation).
2. If the image medium is noted as a painting, artwork, illustration, drawing, sketch, map, diagram, close-up text graphic, or indoor sign, classify it as "None of the above / Noise".
3. Selfies or tourist photos of people are fully acceptable as long as the background is visible; classify the scene based on the background cover and context.
4. If any metadata attributes (such as Season, Time of Day, or Country/Continent) are specified as "Unknown", rely primarily on the visual description and other available metadata elements.

Based on the metadata and visual description, classify this environment into EXACTLY one of the following LULC categories:
{lulc_list}

Format your output EXACTLY as follows:
LABEL: <Insert EXACTLY one category from the list above>
DESCRIPTION: <A detailed, cohesive paragraph in fluent natural language describing the visual evidence, human activities, land cover, and vegetation. This paragraph must integrate all of the observed components and metadata naturally and concisely.>
```

### 🗂️ Land Use / Land Cover (LULC) Classification Vocabulary

#### Natural Land Use / Land Cover Categories
* **Broadleaved forest**: Deciduous or evergreen broad-leaf trees (oak, beech, maple, birch).
* **Coniferous forest**: Evergreen needle-leaf trees (pine, spruce, fir, larch).
* **Mixed forest**: Co-dominant broadleaved and coniferous trees.
* **Tropical forest**: Equatorial rainforests, mangroves, or tropical dry forest.
* **Sparsely wooded / Savanna**: Grassland with scattered trees (10-30% canopy cover).
* **Natural grassland**: Meadows, wild steppes, alpine grasslands, or prairies.
* **Temperate shrubland / Scrub**: Low woody scrub (heather, gorse, bramble).
* **Arid shrubland**: Desert scrub, sagebrush, or dry savanna bushland.
* **Tundra**: Low-growing polar vegetation (mosses, lichens, dwarf shrubs).
* **Sandy desert / Dunes**: Sand sheets, active dunes, or sandy flats.
* **Rocky desert / Gravel plains**: Stony hamadas, gravel plains, or barren volcanic ash fields.
* **Barren soil / Badlands**: Highly eroded clay hills, bare dry earth, or dry salt flats.
* **Bare rock / Cliffs**: Exposed bedrock, cliffs, or scree slopes.
* **Mountain peak / Alpine ridge**: High-altitude rocky ridges, jagged summits, or mountain peaks.
* **Waterfall / Cascade**: Water flowing vertically over a cliff or steep drop.
* **Volcanic terrain / Lava flow**: Barren basalt fields, volcanic craters, or active geothermal areas.
* **Salt flat / Playa**: Dry lake beds covered with salt crusts or mineral deposits.
* **Coastal beach / Spit**: Sandy or pebbly sea coast.
* **Wetland / Marsh / Bog**: Marshes, peat bogs, fens, reed beds, or swamps.
* **River / Stream**: Flowing freshwater channels, creeks, or canals.
* **Lake / Pond**: Standing inland water bodies or reservoirs.
* **Marine / Estuary**: Coastal saltwater, ocean surf, bays, or intertidal flats.
* **Glacier / Permanent ice**: Glaciers, ice caps, or permanent snowfields.
* **Other natural land cover**: Any other natural land cover or landscape.

#### Man-made Land Use / Land Cover Categories
* **Forest plantation**: Evenly spaced rows of planted timber trees.
* **Managed pasture**: Fenced grazing pastures or paddocks.
* **Herbaceous cropland**: Annual cultivated field crops (cereal, corn, wheat, canola).
* **Orchards & Vineyards**: Woody perennial row crops (vineyards, fruit/olive orchards, plantations).
* **Rice paddies / Flooded crops**: Water-flooded agricultural basins.
* **Covered agriculture**: Greenhouses, polytunnels, or nurseries.
* **High-density built-up**: Skyscrapers, high-rise blocks, and dense urban centers.
* **Suburban / Low-density residential**: Single-family houses, villas, private gardens, and streets.
* **Industrial / Commercial zone**: Factories, warehouses, refineries, shopping centers, or office parks.
* **Active construction site**: Earthworks, building foundations, cranes, and scaffolding.
* **Transportation network**: Highways, railways, runways, or shipping ports.
* **Mine / Quarry / Landfill**: Open-pit mines, gravel quarries, or landfill sites.
* **Urban green space**: City parks, golf courses, botanical gardens, or sports fields.
* **Historical / Cultural monument / Archaeological site**: Ancient ruins, historic temples, or archaeological landmarks.
* **Other man-made surface**: Any other artificial or managed land cover or surface.

---

## 🗺️ 5. Spatial Analytics & Interactive Mapping

The pipeline outputs five primary interactive maps, dashboards, and projections to inspect data density, cluster qualities, and land usage:

### 1. H3 Occupancy Map (`src/visualization/generate_h3_occupancy_map.py`)
* **Purpose**: Heatmap displaying image density across the globe.
* **Toggles**: Single density layer.

### 2. Spatial-Semantic Index Map (`src/visualization/generate_h3_semantic_map.py`)
* **Purpose**: Displays the dominant land use/land cover categories on a world map.
* **Hover Details**: Tooltip displays the percentage breakdown of different semantic clusters within that cell (e.g., `Residential: 60%`, `Urban: 40%`).

### 3. Dynamic Statistics Map (`src/utils/dataset_statistics.py`)
* **Purpose**: Generates analytics, visual plots, and layered maps globally or for a specific location.
* **Dynamic Resolution**: Adapts from global views (H3 res 4) to local city views (H3 res 8).
* **Interactive Map Elements**:
  * **Global views** are compiled into a lightweight **11MB** single-layer map showing all 9,888 H3 cells, with Platform, Time of Day, Season, and Köppen Climate breakdowns compiled directly inside the hover tooltips.
  * **Regional views** generate multi-layered maps with toggleable layers for visual comparisons of times, seasons, and Köppen climate zones.
* **Visual Plots**: Generates multi-panel summary plots (.png). Dynamically switches from a standard **2x2** grid to a **3x2** layout if climate or semantic categorization columns are present, adding plots for Köppen climate distribution and top 10 semantic parent category distribution.

### 4. Dynamic Cluster Dashboard (`src/visualization/visualize_cluster_samples.py`)
* **Purpose**: Generates an interactive web dashboard (`cluster_samples.html`) showing representative images and outliers for each cluster.
* **Key Features**:
  * **Two-Step VLM Descriptions**: Displays both the Step 1 visual description (objective catalog of visual details, styled in a warm-yellow box with an amber border) and the Step 2 ecological classification description (styled in a clean grey box with an indigo border).
  * **Image Sample Metadata**: Each representative sample displays its Photo ID, cosine similarity to the cluster centroid, Latitude/Longitude coordinates, collection season, time of day, Köppen-Geiger climate code, and its country/continent region metadata.
  * **Interactive Geographic Map**: Embeds a Leaflet-based geographic spread map for each cluster card showing the cluster's geographic center, density markers for H3 cell centroids, and pins for the representative and outlier image coordinate points.
  * **Offline-First Region Search**: Implements local, offline country and continent search filters. If you type a region that exists in the database, it filters the clusters instantly in-memory, completely bypassing external OSM Nominatim geocoding requests.

### 5. Centroid Semantic UMAP Projection (`src/visualization/visualize_cluster_scatter.py`)
* **Purpose**: Computes a 2D UMAP projection on the average embeddings (centroids) of all 50,000 fine-grained visual clusters to visualize the global semantic manifold of the dataset.
* **Dual Output**:
  * **Static Plot (`cluster_scatter.png`)**: High-resolution image showing the semantic distribution of centroids colored by their parent land-use category.
  * **Interactive WebGL Dashboard (`cluster_scatter.html`)**: A lightweight Plotly WebGL-accelerated interactive scatter plot.
* **Interactive Elements**:
  * **WebGL Rendering**: Employs GPU-accelerated WebGL (`Scattergl`) to run smoothly (60 FPS) when panning, zooming, or box-selecting across tens of thousands of clusters in any browser.
  * **Density Sizing**: Node marker size scales dynamically based on the number of images assigned to that cluster (larger dots = higher density clusters).
  * **Hover Metadata**: Hovering over any cluster centroid displays its unique `Cluster ID`, `Cluster Label`, `Parent Category`, exact image count, and its VLM visual description.
  * **Filter Toggles**: Double-clicking or clicking parent categories in the legend instantly filters and highlights specific semantic families (e.g. isolating all forestry subclasses).

---

## 💾 6. Data Versioning (DVC)

To handle heavy files (Parquet databases, HTML maps, images), `run_full_pipeline.sh` implements autonomous DVC standalone tracking:
1. **HDD Storage**: Outputs are written to a fast SSD, then backed up to a high-capacity HDD directory tracked by DVC.
2. **Push**: Pushes heavy data to remote storage using `dvc push`.
3. **Git Sync**: Copies the updated `.dvc` tracking files back to the SSD Git repository, automatically commits them, and pushes them to track repository state changes.

---

## ⚖️ 7. Determining the Optimal Cluster Count (*k*)

As the dataset grows (e.g., from 3.3M to 5.5M+ images), determining the optimal cluster count *k* is essential. Standard random cross-validation fails due to **spatial autocorrelation** (spatial data leakage). 

The `validate_cluster_count.py` script implements **Spatial Block Hold-Out validation** using the GPU (FAISS [Johnson et al., 2019]) to systematically find the optimal *k*.

### 🔬 Methodology: Spatial Block Hold-Out

#### 1. Spatial Autocorrelation & The Generalization Gap
In spatial datasets, adjacent data points are highly correlated due to **Tobler's First Law of Geography** [Tobler, 1970]: *"Everything is related to everything else, but near things are more related than distant things."* 

If we partition the training and validation sets randomly, nearby images (e.g., sequential streetscapes or photos of the same landmark) will appear in both sets. This causes **spatial data leakage**, artificially deflating the validation loss and hiding overfitting. 

To measure true generalization, we must partition the dataset geographically. We downscale the fine-grained H3 cell *c* (resolution 11) to its coarse parent block *b*:

$$
b = \text{parent}(c, R_p)
$$

where *R_p* = 4 (coarse blocks of ~11,000 km²). We randomly split the set of unique parent blocks *B* into disjoint training and validation blocks:

$$
\mathcal{B}_{\text{train}} \cap \mathcal{B}_{\text{val}} = \emptyset
$$

$$
\mathcal{B}_{\text{train}} \cup \mathcal{B}_{\text{val}} = \mathcal{B}
$$

#### 2. Optimization Objective & Reconstruction Loss
Let *X_train* be the set of image embeddings belonging to *B_train*, and *X_val* be the embeddings belonging to *B_val*.

For a given number of clusters *k*, K-Means learns a set of centroids *C\** = {*c₁*, ..., *c_k*} by minimizing the Within-Cluster Sum of Squares (WCSS) on the training set:

$$
\mathcal{L}_{\text{train}}(C) = \sum_{x \in X_{\text{train}}} \min_{c \in C} \| x - c \|^2
$$

The **Validation Reconstruction Loss (Mean Squared Error)** is then evaluated by measuring how well the centroids *C\** represent the unseen validation blocks:

$$
\text{MSE}_{\text{val}}(k) = \frac{1}{|X_{\text{val}}|} \sum_{y \in X_{\text{val}}} \min_{c \in C^*} \| y - c \|^2
$$

#### 3. Optimal *k* Selection via the Elbow Method
As *k* → *N*, the training loss `MSE_train(k)` → 0. However, on the validation set, if *k* is too high, the centroids will overfit to the specific geographic configurations of the training blocks. The optimal *k\** is determined using the **Elbow Method** [Thorndike, 1953] on `MSE_val(k)`—the point at which the rate of decrease in validation error slows down significantly, representing the maximum compression with optimal generalization:

$$
k^* = \arg\max_k \left( \frac{\partial^2 \text{MSE}_{\text{val}}}{\partial k^2} \right)
$$

### 💻 Execution Example
Run the validation script across a range of *k* values (*k* ∈ [10,000, 50,000]):
```bash
python3 -m src.utils.validate_cluster_count \
  --input "full_pipeline_output/geo_space_deduplicated.parquet" \
  --k_min 10000 \
  --k_max 50000 \
  --k_step 10000 \
  --sample_limit 0 \
  --output_plot "cluster_count_validation.png"
```
*(Setting `--sample_limit 0` ensures the script uses the entire dataset for partitioning rather than downsampling).*

---

## 📚 8. Literature & Software References

* Brodsky, A. (2018). *H3: Uber's Hexagonal Hierarchical Spatial Index*. Uber Engineering. [https://h3geo.org](https://h3geo.org)
* Dhillon, I. S., & Modha, D. S. (2001). Concept decompositions for large sparse text document collections with applications to high-dimensional clustering. *Machine Learning*, 42(1), 143–175.
* Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.
* McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint arXiv:1802.03426*.
* Cao, B., Chen, K., Maninis, K. K., Chen, K., Karpur, A., Xia, Y., ... & Araujo, A. (2026). Tipsv2: Advancing vision-language pretraining with enhanced patch-text alignment. *CVPR 2026*.
* Sculley, D. (2010). Web-scale k-means clustering. In *Proceedings of the 19th International Conference on World Wide Web* (pp. 1177–1178).
* Thorndike, R. L. (1953). Who belongs in the family? *Psychometrika*, 18(4), 267–276.
* Tobler, W. R. (1970). A computer movie simulating urban growth in the Detroit region. *Economic Geography*, 46(sup1), 234–240.
* Beck, H. E., T. R. McVicar, N. Vergopolan, A. Berg, N. J. Lutsko, A. Dufour, Z. Zeng, X. Jiang, A. I. J. M. van Dijk, and D. G. Miralles. High-resolution (1 km) Köppen-Geiger maps for 1901–2099 based on constrained CMIP6 projections. Scientific Data 10, 724 (2023).
