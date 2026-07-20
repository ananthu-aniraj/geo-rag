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
* **`src/visualization/create_uncovered_land_areas_shp.py`**: Reads existing image CSV datasets, aggregates coordinate points into H3 cells at a user-defined resolution (default: resolution 5), and flags cells as "covered" if they exceed an image count threshold (default: 0). It then converts the covered H3 cells to polygon geometry and subtracts them from a standard global land mass shapefile (e.g., Natural Earth admin borders).
* **Usage in Search**: The resulting output shapefile (`shapefiles/uncovered_land_areas_test.shp`) represents land areas that are still poorly mapped. The global grid searchers (like `src/scrapers/flickr_5km_grid_search.py`) load this shapefile at startup and perform a fast spatial R-tree index check. They skip querying any grid box that does not intersect an uncovered land polygon, which reduces API requests and concentrates scraping efforts on data-poor zones.

---

## ⚙️ 3. Pipeline Walkthrough

### Step 1: Spatial Deduplication & Filtering
* **Script**: `src/processing/process_scraped_data.py`
* **Operation**: Groups coordinates into H3 Resolution 11 parent cells [Brodsky, 2018]. Performs spatial-temporal deduplication using TIPSv2 image embeddings [Cao, 2026] to ensure uniform geographic coverage.
* **Zero-Shot Noise Filters**:
  * **Flickr Indoor/Outdoor Filter**: By default, the script filters out indoor photos using zero-shot text-image classification with TIPSv2. Images are compared against the prompts *"An indoor scene"* and *"An outdoor landscape or street view"*. If an image matches the indoor class, it is discarded. This filter applies **only to Flickr images** (since street-view platforms like Mapillary/KartaView are intrinsically outdoor, and iNaturalist observations are filtered by macro characteristics). Can be bypassed with the `--no_filter` flag.
  * `--filter_macro`: Filters out macro close-up flora/fauna images (typically for iNaturalist data) using a zero-shot *"A macro/close-up photo of flowers, leaves, bark, or insects"* classifier.
  * `--filter_sky`: Filters out empty sky views (typically for iNaturalist data) using a zero-shot *"A view of empty sky, clouds, or flying objects"* classifier.

### Step 1b: Timestamp Standardization
* **Script**: `src/processing/standardize_timestamps.py`
* **Operation**: Normalizes inconsistent date/time strings and Unix epochs into ISO 8601 strings (`YYYY-MM-DDTHH:MM:SSZ`).
* **Categorizations Added**:
  * **Time of Day**: *Dawn* (05-08), *Morning* (08-12), *Afternoon* (12-17), *Dusk* (17-20), *Night* (20-05).
  * **Seasons**: Latitude-aware seasons (Spring/Summer/Autumn/Winter for temperate/polar regions; Wet/Dry seasons for tropical regions).

### Step 1d: Coordinate Anomaly Cleanup (GPS Glitch Removal)
* **Script**: `src/processing/cleanup_coordinate_anomalies.py`
* **Operation**: Scans the deduplicated database to safely purge locked-latitude coordinate lines caused by faulty contributor GPS units at source. Writes the clean data to independent output files (`geo_space_cleaned.parquet` / `geo_space_cleaned.csv`), leaving the raw deduplicated database untouched.
* **Safety Criteria**: A rounded latitude parallel $L$ (rounded to 5 decimal places, representing $\approx 1.1\text{ meters}$ precision) is flagged and purged only if:
  $$\text{Count}(L) > 10 \quad \text{and} \quad \text{Longitude Span}(L) > 1.0^{\circ}$$
  *(A longitude span of $> 1.0^{\circ}$ is $\approx 111\text{ km}$, which ensures that dense cities—which naturally occupy tiny bounding boxes—are completely preserved, while global coordinate-locked lines spanning multiple countries are cleanly discarded).*

### Step 2: Global Clustering & MLLM Auto-Labeling
* **Script**: `src/indexing/cluster_images_global.py` & `src/indexing/relabel_failed_clusters.py`
* **Operation**: Performs hierarchical two-level clustering on image embeddings:
  1. **Fine-Grained Child Clustering**: Runs Mini-Batch K-Means [Sculley, 2010] on the raw image embeddings (e.g., $N=3.37\text{M}$ vectors) to partition the data into $k$ fine-grained child clusters. Each cluster captures highly specific visual/geographical concepts.
  2. **Hierarchical Parent Clustering**: Runs Spherical K-Means [Dhillon & Modha, 2001] on the normalized child centroids to group them into $k_{\text{parents}}$ broader parent clusters (where $k_{\text{parents}} = \max(2, k / 80)$). This groups similar child clusters into high-level visual/semantic classes.
  3. **Multi-Modal LLM Labeling**: Sends representative centroid samples for both child and parent clusters to a Multi-Modal LLM (e.g., Gemma-2 via SGLang) to automatically generate descriptive semantic labels and descriptions. Script `src/indexing/relabel_failed_clusters.py` acts as a fallback for download/API timeout failures.

### Step 3: H3 Spatial-Semantic Indexing
* **Script**: `src/indexing/build_spatial_semantic_index.py`
* **Operation**: Aggregates the clustered dataset into a multi-resolution H3 spatial index [Brodsky, 2018], linking cells to dominant cluster categories, seasons, and times of day.

---

## 🤖 4. Multi-Modal LLM (MLLM) Auto-Labeling Prompt

To automatically label image clusters, the pipeline uses a Multi-Modal LLM (such as Gemma-2 via SGLang) to analyze representative cluster images. The MLLM is provided with the following detailed prompt template and vocabulary.

### 📝 MLLM Prompt Template
```text
You are analyzing a single ground-level photograph from a global land-survey campaign; the sample point may be anywhere on Earth (tropical, arid, temperate, boreal, alpine, wetland, or coastal). The camera stands at a sample point and looks outward; the land cover AT THE FOREGROUND/CENTER of the frame is the DOMINANT cover that must be characterized. Background elements (distant trees, buildings, hills) are CONTEXT only and must never override the dominant foreground cover.

Report only what is visually supported. When a cue is ambiguous, describe the cue precisely rather than guessing a label, and commit to the single most likely identity while noting the discriminating feature you used. Do not infer management status, permanence, ownership, subsurface geology, or history that is not visually evident (e.g. do not call grass "temporary" unless mowing/tillage/rows are visible, and do not name a bedrock formation that is not exposed).

Several fields below describe properties that are also detectable from satellite or airborne sensors (vegetation greenness, growth stage, canopy height, surface sealing, soil and surface-rock colour). Fill these from visual evidence wherever the cue is present, as they will be linked to remote-sensing data.

Carefully observe and describe the following components:
1. dominant_cover: The single land-cover type occupying most of the foreground/center (e.g. cereal crop, grassland, bare soil, broadleaved trees, built surface, sand, snow/ice, water, wetland).
2. cover_fraction_estimate: Share of the visible ground area that is vegetation vs bare soil/rock vs sealed/built surface vs water/ice.
3. visible_evidence & salient_objects: Primary foreground objects, structures, architectural elements, vehicles, or natural formations visible. Describe them naturally as a human observer would.
4. vegetation_detail: Height, leaf/frond shape, seed-head/flower form, rows or not (cereals, broadleaf/industrial crops, grassland, woody cover broadleaved vs coniferous vs palm).
5. vegetation_condition & phenological_stage: Vigor (vigorous-green, senescent, dead/brown) and growth stage (emerging, full-canopy, harvested/stubble).
6. canopy_structure: Canopy closure (closed, open, scattered) and height (low shrub <2m, small trees 2-5m, mature trees >5m).
7. soil_surface_state & lithology: Exposed ground sealing (permeable, unsealed, sealed-impervious) and visibly exposed soil/rock texture and color.
8. structure_and_pattern: Terrain slope, parcel boundaries, linear features (crop rows, tramlines, ditches, tracks), or recent disturbances.
9. context_background: Elements visible in the distance or periphery (buildings, treelines, mountains, water bodies) that are not the sample point cover.
10. human_evidence & activities: Visible signs of human use (buildings, machinery, fences, planted rows, paths, irrigation) and the activities they support.

Based ONLY on the visual evidence described above, classify this environment into EXACTLY one of the following Land Use / Land Cover (LULC) categories:
[LULC_LIST]

Format your output EXACTLY as follows:
LABEL: <Insert EXACTLY one category from the list above>
DESCRIPTION: <A detailed, cohesive paragraph in fluent natural language describing the visual evidence, human activities, land cover, and vegetation. This paragraph must integrate all of the observed components naturally and concisely. To avoid repetition, do not repeat the land cover class or vegetation type multiple times across different sentences; instead, weave them together into a single, cohesive narrative description of the scene.>
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

The pipeline outputs three primary Leaflet-based interactive maps to inspect data density and land usage:

### 1. H3 Occupancy Map (`src/visualization/generate_h3_occupancy_map.py`)
* **Purpose**: Heatmap displaying image density across the globe.
* **Toggles**: Single density layer.

### 2. Spatial-Semantic Index Map (`src/visualization/generate_h3_semantic_map.py`)
* **Purpose**: Displays the dominant land use/land cover categories on a world map.
* **Hover Details**: Tooltip displays the percentage breakdown of different semantic clusters within that cell (e.g., `Residential: 60%`, `Urban: 40%`).

### 3. Dynamic Statistics Map (`src/utils/dataset_statistics.py`)
* **Purpose**: Generates analytics and plots globally or for a specific location.
* **Dynamic Resolution**: Adapts from global views (H3 res 4) to local city views (H3 res 8).
* **Smart Performance**: 
  * **Global views** are compiled into a lightweight **11MB** single-layer map showing all 9,888 H3 cells, with Platform, Time of Day, and Season breakdowns compiled directly inside the hover tooltips.
  * **Regional views** generate multi-layered maps with toggleable layers for visual comparisons of times/seasons.

---

## 💾 6. Data Versioning (DVC)

To handle heavy files (Parquet databases, HTML maps, images), `run_full_pipeline.sh` implements autonomous DVC standalone tracking:
1. **HDD Storage**: Outputs are written to a fast SSD, then backed up to a high-capacity HDD directory tracked by DVC.
2. **Push**: Pushes heavy data to remote storage using `dvc push`.
3. **Git Sync**: Copies the updated `.dvc` tracking files back to the SSD Git repository, automatically commits them, and pushes them to track repository state changes.

---

## ⚖️ 7. Determining the Optimal Cluster Count (k)

As the dataset grows (e.g., from 3.3M to 5.5M+ images), determining the optimal cluster count $k$ is essential. Standard random cross-validation fails due to **spatial autocorrelation** (spatial data leakage). 

The `validate_cluster_count.py` script implements **Spatial Block Hold-Out validation** using the GPU (FAISS [Johnson et al., 2019]) to systematically find the optimal $k$.

### 🔬 Methodology: Spatial Block Hold-Out

#### 1. Spatial Autocorrelation & The Generalization Gap
In spatial datasets, adjacent data points are highly correlated due to **Tobler's First Law of Geography** [Tobler, 1970]: *"Everything is related to everything else, but near things are more related than distant things."* 

If we partition the training and validation sets randomly, nearby images (e.g., sequential streetscapes or photos of the same landmark) will appear in both sets. This causes **spatial data leakage**, artificially deflating the validation loss and hiding overfitting. 

To measure true generalization, we must partition the dataset geographically. We downscale the fine-grained H3 cell $c$ (resolution 11) to its coarse parent block $b$:
$$b = \text{parent}(c, R_p)$$
where $R_p = 4$ (coarse blocks of $\approx 11,000\text{ km}^2$). We randomly split the set of unique parent blocks $\mathcal{B}$ into disjoint training and validation blocks:
$$\mathcal{B}_{\text{train}} \cap \mathcal{B}_{\text{val}} = \emptyset$$
$$\mathcal{B}_{\text{train}} \cup \mathcal{B}_{\text{val}} = \mathcal{B}$$

#### 2. Optimization Objective & Reconstruction Loss
Let $X_{\text{train}}$ be the set of image embeddings belonging to $\mathcal{B}_{\text{train}}$, and $X_{\text{val}}$ be the embeddings belonging to $\mathcal{B}_{\text{val}}$.

For a given number of clusters $k$, K-Means learns a set of centroids $C^* = \{c_1, \dots, c_k\}$ by minimizing the Within-Cluster Sum of Squares (WCSS) on the training set:
$$\mathcal{L}_{\text{train}}(C) = \sum_{x \in X_{\text{train}}} \min_{c \in C} \| x - c \|^2$$

The **Validation Reconstruction Loss (Mean Squared Error)** is then evaluated by measuring how well the centroids $C^*$ represent the unseen validation blocks:
$$\text{MSE}_{\text{val}}(k) = \frac{1}{|X_{\text{val}}|} \sum_{y \in X_{\text{val}}} \min_{c \in C^*} \| y - c \|^2$$

#### 3. Optimal $k$ Selection via the Elbow Method
As $k \to N$, the training loss $\text{MSE}_{\text{train}}(k) \to 0$. However, on the validation set, if $k$ is too high, the centroids will overfit to the specific geographic configurations of the training blocks. The optimal $k^*$ is determined using the **Elbow Method** [Thorndike, 1953] on $\text{MSE}_{\text{val}}(k)$—the point at which the rate of decrease in validation error slows down significantly, representing the maximum compression with optimal generalization:
$$k^* = \arg\max_k \left( \frac{\partial^2 \text{MSE}_{\text{val}}}{\partial k^2} \right)$$

### 💻 Execution Example
Run the validation script across a range of $k$ values ($k \in [10000, 50000]$):
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
