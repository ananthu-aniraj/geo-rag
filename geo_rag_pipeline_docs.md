# Geo-RAG Pipeline & Data Architecture Documentation

This document provides a comprehensive guide to the **Geo-RAG (Geographic Retrieval-Augmented Generation)** data processing, clustering, and spatial-semantic mapping pipeline.

---

## 🗺️ 1. Pipeline Overview & Architecture

The Geo-RAG pipeline is designed to ingest raw street-level and outdoor image databases (Flickr, Mapillary, camera traps), deduplicate them spatially, cluster them semantically using image embeddings, auto-label clusters using Multi-Modal LLMs (MLLMs), and build interactive web visualizations.

![Pipeline Flowchart](pipeline_flowchart_clean.jpg)

---

## 📊 2. Ingested Data Sources

The pipeline processes diverse geotagged image sources, unifies their schemas, and resolves timezone inconsistencies:

| Data Source | Content Type | Location / Spatial Metadata | Timestamp & Timezone Characteristics |
|---|---|---|---|
| **Flickr** | Geotagged outdoor/scenic photographs | Latitude / Longitude coordinates | Naive local datetimes (date-taken) |
| **Mapillary** | Sequential street-level imagery | Resolution 11 H3 cells & coordinates | Naive local datetimes (`datetime_local`) or UTC epoch |
| **iWildCam** | Camera trap wildlife observations | Station coordinates | Naive local timestamps |
| **iNaturalist** | Species observations | Coordinates (filtered for sky/macro) | Standardized observation datetimes |

> [!NOTE]
> **Timezone Standardization**: Because naive local time strings are parsed with `utc=True` during ingestion, the numeric hour digit is preserved exactly as it was captured (e.g. 10:00 PM local is stored as `22:00:00Z`). This allows accurate local Time of Day profiling without external timezone database queries.

---

## ⚙️ 3. Pipeline Walkthrough

### Step 1: Spatial Deduplication & Filtering
* **Script**: `process_scraped_data.py`
* **Operation**: Groups coordinates into H3 Resolution 11 parent cells. Performs spatial-temporal deduplication using TIPSv2 image embeddings to ensure uniform geographic coverage.
* **Optional Filters**: 
  * `--filter_macro`: Filters out macro close-up flora/fauna images.
  * `--filter_sky`: Filters out empty sky views.

### Step 1b: Timestamp Standardization
* **Script**: `standardize_timestamps.py`
* **Operation**: Normalizes inconsistent date/time strings and Unix epochs into ISO 8601 strings (`YYYY-MM-DDTHH:MM:SSZ`).
* **Categorizations Added**:
  * **Time of Day**: *Dawn* (05-08), *Morning* (08-12), *Afternoon* (12-17), *Dusk* (17-20), *Night* (20-05).
  * **Seasons**: Latitude-aware seasons (Spring/Summer/Autumn/Winter for temperate/polar regions; Wet/Dry seasons for tropical regions).

### Step 2: Global Clustering & MLLM Auto-Labeling
* **Script**: `cluster_images_global.py` & `relabel_failed_clusters.py`
* **Operation**: Computes Mini-Batch K-Means on image embeddings. Sends cluster centroid samples to a Multi-Modal LLM (e.g., Gemma-2 via SGLang) to automatically generate descriptive semantic labels and parent categories (e.g., "Forest road", "Residential street").

### Step 3: H3 Spatial-Semantic Indexing
* **Script**: `build_spatial_semantic_index.py`
* **Operation**: Aggregates the clustered dataset into a multi-resolution H3 spatial index, linking cells to dominant cluster categories, seasons, and times of day.

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

### 1. H3 Occupancy Map (`generate_h3_occupancy_map.py`)
* **Purpose**: Heatmap displaying image density across the globe.
* **Toggles**: Single density layer.

### 2. Spatial-Semantic Index Map (`generate_h3_semantic_map.py`)
* **Purpose**: Displays the dominant land use/land cover categories on a world map.
* **Hover Details**: Tooltip displays the percentage breakdown of different semantic clusters within that cell (e.g., `Residential: 60%`, `Urban: 40%`).

### 3. Dynamic Statistics Map (`dataset_statistics.py`)
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

The `validate_cluster_count.py` script implements **Spatial Block Hold-Out validation** using the GPU (FAISS) to systematically find the optimal $k$.

### 🔬 Methodology: Spatial Block Hold-Out
1. **Parent Block Downscaling**: Downscales the fine-grained `H3_Cell` (resolution 11) to coarse geographic blocks at **Resolution 4** (representing areas of ~11,000 sq km).
2. **Block-Level Partitioning**: Randomly holds out 10% of these coarse H3 blocks as a validation set, keeping the remaining 90% as the training set. This ensures the model is evaluated on "unseen regions" rather than geographically adjacent coordinates.
3. **FAISS GPU Acceleration**: Trains K-Means on the training blocks and computes the average reconstruction loss on the validation blocks entirely on the GPU.

### 💻 Execution Example
Run the validation script across a range of $k$ values ($k \in [10000, 50000]$):
```bash
python3 validate_cluster_count.py \
  --input "full_pipeline_output/geo_space_deduplicated.parquet" \
  --k_min 10000 \
  --k_max 50000 \
  --k_step 10000 \
  --sample_limit 0 \
  --output_plot "cluster_count_validation.png"
```
*(Setting `--sample_limit 0` ensures the script uses the entire dataset for partitioning rather than downsampling).*
