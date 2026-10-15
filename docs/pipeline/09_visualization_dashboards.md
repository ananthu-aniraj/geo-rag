# Spatial Analytics & Interactive Mapping

This document describes the visualization suite and dashboard mapping utilities in the Geo-RAG pipeline.

---

## 🗺️ Mapping & Visualization Suite

The pipeline outputs five interactive maps, dashboards, and projections to inspect data density, cluster qualities, and land usage:

### 1. H3 Occupancy Map (`generate_h3_occupancy_map.py`)

* **Purpose**: Heatmap displaying image density across the globe.
* **Toggles**: Single density layer.

### 2. Spatial-Semantic Index Map (`generate_h3_semantic_map.py`)

* **Purpose**: Displays the dominant land use/land cover categories on a world map.
* **Hover Details**: Tooltip displays the percentage breakdown of different semantic clusters within that cell (e.g., `Residential: 60%`, `Urban: 40%`).

### 3. Dynamic Statistics Map (`dataset_statistics.py`)

* **Purpose**: Generates analytics, visual plots, and layered maps globally or for a specific location.
* **Dynamic Resolution**: Adapts from global views (H3 res 4) to local city views (H3 res 8).
* **Interactive Map Elements**:
  * **Global views** are compiled into a lightweight **11MB** single-layer map showing all 9,888 H3 cells, with Platform, Time of Day, Season, and Köppen Climate breakdowns compiled directly inside the hover tooltips.
  * **Regional views** generate multi-layered maps with toggleable layers for visual comparisons of times, seasons, and Köppen climate zones.
* **Visual Plots**: Generates multi-panel summary plots (.png). Dynamically switches from a standard **2x2** grid to a **3x2** layout if climate or semantic categorization columns are present, adding plots for Köppen climate distribution and top 10 semantic parent category distribution.

### 4. Dynamic Cluster Dashboard (`visualize_cluster_samples.py`)

* **Purpose**: Generates an interactive web dashboard (`cluster_samples.html`) showing representative images and outliers for each cluster.
* **Key Features**:
  * **Two-Step VLM Descriptions**: Displays both the Step 1 visual description (objective catalog of visual details, styled in a warm-yellow box with an amber border) and the Step 2 ecological classification description (styled in a clean grey box with an indigo border).
  * **Image Sample Metadata**: Each representative sample displays its Photo ID, cosine similarity to the cluster centroid, Latitude/Longitude coordinates, collection season, time of day, Köppen-Geiger climate code, and its country/continent region metadata.
  * **Interactive Geographic Map**: Embeds a Leaflet-based geographic spread map for each cluster card showing the cluster's geographic center, density markers for H3 cell centroids, and pins for the representative and outlier image coordinate points.
  * **Offline-First Region Search**: Implements local, offline country and continent search filters. If you type a region that exists in the database, it filters the clusters instantly in-memory, completely bypassing external OSM Nominatim geocoding requests.
  * **Remote Image Viewing over SSH Tunneling**:
    If the dashboard is generated on a remote execution machine and you want to view it on a local PC (e.g. your laptop), you can securely stream the offline images without copying them:

    1. **Start Server on Remote PC**: Start a lightweight python file server pointing to your image directory:

       ```bash
       python3 -m http.server 8000 --directory /user/aaniraj/home/Documents/Projects/data/
       ```

    2. **Establish Tunnel on Local Laptop**: Run an SSH session with local port forwarding enabled:

       ```bash
       ssh -L 8000:localhost:8000 user@execution_pc_ip
       ```

    3. **Map Path in Browser UI**: Copy the `cluster_samples.html` and `cluster_samples_data.js` to your laptop and open the HTML file. In the **Offline Image Server Mapping** panel at the top, paste `http://localhost:8000/` in the **Replace** box. The browser will dynamically stream the images from the remote disk over your SSH tunnel in real-time.

### 5. Centroid Semantic UMAP Projection (`visualize_cluster_scatter.py`)

* **Purpose**: Computes a 2D UMAP projection on the average embeddings (centroids) of all 50,000 fine-grained visual clusters to visualize the global semantic manifold of the dataset.
* **Dual Output**:
  * **Static Plot (`cluster_scatter.png`)**: High-resolution image showing the semantic distribution of centroids colored by their parent land-use category.
  * **Interactive WebGL Dashboard (`cluster_scatter.html`)**: A lightweight Plotly WebGL-accelerated interactive scatter plot.
* **Interactive Elements**:
  * **WebGL Rendering**: Employs GPU-accelerated WebGL (`Scattergl`) to run smoothly (60 FPS) when panning, zooming, or box-selecting across tens of thousands of clusters in any browser.
  * **Density Sizing**: Node marker size scales dynamically based on the number of images assigned to that cluster (larger dots = higher density clusters).
  * **Interactive Photo Hover Cards (Cave Umwelten inspired)**: Hovering over any cluster dot triggers a floating glassmorphism card displaying real representative photo thumbnails, the cluster's unique ID, dominant ecological label, parent category badge, image count, and full VLM description.
  * **Platform Filtering**: `--platforms` argument (e.g. `--platforms flickr inaturalist`) to filter thumbnail previews to persistent platforms, or leave unset to sample across all platforms (with automatic prioritization of permanent URLs).
  * **SSH Tunneling & Offline Image Server Mapping**: Includes `--image_root_dir`, `--find_prefix`, and `--replace_prefix` CLI options, auto-detecting base image directories and embedding an interactive top-right **"Offline Image Server Mapping"** control with `localStorage` memory. This allows viewing offline images streamed over an SSH tunnel (`http://localhost:8000/`) directly in the UMAP hover cards.
  * **Coordinate Caching**: Automatically caches 2D UMAP projections to `_umap_coords.npy` for sub-second re-generation. Use `--recompute_umap` to force recalculation.
  * **Filter Toggles**: Double-clicking or clicking parent categories in the legend instantly filters and highlights specific semantic families (e.g. isolating all forestry subclasses).

### 6. Interactive Anomaly Cleaner Dashboard (`interactive_anomaly_cleaner.py`)

* **Purpose**: Provides a live interactive web dashboard to manually inspect, flag, and purge spatial anomalies (specifically Mapillary sensor glitches or unwanted tracks) on a continent-by-continent basis.
* **Interactive Elements**:
  * **Continent-Scoped Hexagonal Grid**: High-performance Leaflet canvas layer (`preferCanvas: true`) rendering thousands of H3 cells at 60 FPS without DOM overhead.
  * **Sub-50ms On-Click Image Inspection**: Clicking any hexagon opens a side drawer displaying cell-level generic metadata (countries, Köppen climate classes, centroid coordinates) and representative photo thumbnails via direct Mapillary CDN URLs.
  * **2-Phase Staged Curation**: Flagging cells updates an in-memory "Staged Removals" cart (with 0 ms latency); hitting **"🚀 Execute Purge"** runs a single streaming row-group pass over the Parquet dataset using PyArrow.
  * **Reproducible Rules Specification**: Automatically persists and loads decisions from `manual_h3_removals.json`, enabling headless batch execution in CI/CD via `--batch-purge`.

### 7. Continent H3 Spatial-Semantic Map (`generate_continent_h3_semantic_map.py`)

* **Purpose**: Generates an interactive continent-scoped Folium map at H3 resolution 4 (configurable) that pairs semantic land-cover clusters with real representative photo thumbnails for every polygon.
* **Explicit User Inputs**:
  * `--continent <Name>`: Target continent (e.g. `Africa`, `Europe`, `Asia`).
  * `--index <path>`: Path to the pre-aggregated H3 spatial-semantic index (`geo_space_...h3_semantic_index.parquet`), supplying the hexagon cells, density colormap, and dominant LULC percentage breakdowns.
  * `--dataset <path>`: Path to the clustered Parquet dataset (e.g. `geo_space_...clustered_k_40000.parquet` or merged Parquet) or base cleaned dataset. If a clustered sidecar is passed, companion coordinates from `geo_space_cleaned.parquet` are auto-linked via high-speed zero-copy append in <2 seconds.
  * `--cleaned_dataset <path>`: *(Optional)* Explicit path to base cleaned dataset if the clustered sidecar is stored in a separate directory.
  * `--platforms <Platform ...>`: *(Optional)* Filter representative preview photos to specific platforms (e.g. `--platforms flickr inaturalist` guarantees persistent static image rendering for public embeds, default: all platforms).
* **Interactive Elements**:
  * **Stratified Cluster Photo Sampling**: When a clustered dataset is loaded, representative preview photos are stratified across the distinct dominant clusters present within each H3 cell (e.g. 1 photo from Rainforest, 1 from Savanna, 1 from Wetlands). Each thumbnail displays its true cluster category badge.
  * **Continent Scoping & High Performance**: By focusing on a single continent (`--continent Africa`, `Europe`, `Asia`, etc.), active H3 cells remain under a few thousand, rendering a fast (~2–3 MB) standalone HTML map at 60 FPS without DOM or browser lag.
  * **Hover Tooltip**: Displays the cell ID, image count, and dominant LULC classes with percentages.
  * **On-Click Image Gallery**: Clicking any hexagon opens a popup card detailing dominant categories and a 2×2 responsive thumbnail grid of photos from that polygon (with Mapillary web links).
  * **Multi-Basemap Selection**: Features CartoDB Positron (Light), CartoDB Dark_Matter (Dark), CartoDB Voyager (Color OpenStreetMap), and Esri World Imagery (Satellite) without tile blocking on local `file:///` URLs.
  * **SSH Tunneling & Offline Dataset Mapping**: Integrates the floating "Offline Image Server Mapping" control with local path auto-detection and `--replace_prefix` / `localStorage` persistence, allowing users to view remote server images over an SSH tunnel (`http://localhost:8000/`).

---

## 📂 Experiment Output Organization & Namespacing

To facilitate model comparison sweeps and avoid clobbering visualization files across experiments (e.g. comparing TIPSv2 vs DINOv2 vs SigLIP, or varying $k$ from 20,000 to 50,000), all visual reports and maps are routed into dedicated folders:

```text
vis_${num_clusters}_${model_name}_${representation_type}_${precision}/
├── global_cluster_map.html                  # Interactive Folium map
├── cluster_samples_k_${k}.html              # Multi-image sample grid
├── cluster_semantic_scatter_k_${k}.png      # UMAP 2D scatter plot
├── global_h3_occupancy_map.html             # Spatial density map
├── global_h3_semantic_map.html              # Spatial-semantic map
├── africa_h3_semantic_map.html              # Continent H3 map with photo galleries
├── europe_h3_semantic_map.html              # Continent H3 map with photo galleries
├── asia_h3_semantic_map.html                # Continent H3 map with photo galleries
├── north_america_h3_semantic_map.html       # Continent H3 map with photo galleries
├── south_america_h3_semantic_map.html       # Continent H3 map with photo galleries
├── oceania_h3_semantic_map.html             # Continent H3 map with photo galleries
├── global_dataset_stats.png                 # Summary distribution plot
├── global_dataset_stats.txt                 # Textual statistics report
├── global_dataset_map.html                  # Layered coverage map
└── cluster_count_validation.png             # (If auto_find_k is enabled)
```

The heavy dataset and companion embedding binaries remain stored once in the root dataset directory, keeping visual experiment folders lightweight, modular, and easy to share or archive.
