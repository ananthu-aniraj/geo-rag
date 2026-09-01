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
  * **Hover Metadata**: Hovering over any cluster centroid displays its unique `Cluster ID`, `Cluster Label`, `Parent Category`, exact image count, and its VLM visual description.
  * **Filter Toggles**: Double-clicking or clicking parent categories in the legend instantly filters and highlights specific semantic families (e.g. isolating all forestry subclasses).
