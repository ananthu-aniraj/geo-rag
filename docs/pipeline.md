# Geo-RAG Pipeline Guide & Architecture

This document serves as the master overview and index for the Geo-RAG spatial-semantic data engineering, clustering, and mapping pipeline.

> [!TIP] New to Geo-RAG?
> If you are new to the project or looking for high-level concepts and a glossary, check out the **[Preliminaries & Core Concepts Guide](preliminaries.md)** first!

---

## 🗺️ 1. Pipeline Overview & Architecture

The pipeline takes raw street-level and outdoor image databases, cleans up bad GPS data, extracts AI visual fingerprints, clusters similar landscapes globally, generates natural-language descriptions with Vision-Language Models (VLMs), and compiles interactive web maps.

![Pipeline Flowchart](pipeline_flowchart_clean.jpg)

---

## ⚙️ 2. Pipeline Walkthrough & Stages

Detailed documentation for each stage of the pipeline can be found in the sub-guides below:

| Stage | Plain English Goal | Core Responsibility & Mechanics | Guide |
| :--- | :--- | :--- | :--- |
| **01. Ingestion & Scraping** | Download photo records from public platforms | Flickr, Mapillary, KartaView, and iNaturalist API queries with spatial difference masking. | [Guide ➡️](pipeline/01_ingestion_scraping.md) |
| **02. Spatial Deduplication** | Prune redundant shots taken at the exact same tree/corner | H3 Resolution 11 spatial grouping, visual cosine similarity filtering, and decoupled `.npy` vector caching. | [Guide ➡️](pipeline/02_spatial_deduplication.md) |
| **03. Timestamp Standardization** | Fix capture timezones and validate country boundaries | Datetime UTC normalization, climate zone attribution, and EPSG:3857 coastal buffer validation. | [Guide ➡️](pipeline/03_timestamp_standardization.md) |
| **04. Coordinate Anomaly Cleanup** | Remove GPS tracks corrupted by receiver glitches | Detects frozen-latitude tracks and purges faulty Mapillary vehicle sequences. | [Guide ➡️](pipeline/04_coordinate_cleanup.md) |
| **05. Optimal $k$ Estimation** | Estimate how many distinct visual landscapes exist | Spatial Block Hold-Out cross-validation and reconstruction loss Elbow curve heuristics. | [Guide ➡️](pipeline/05_optimal_k_estimation.md) |
| **06. Global GPU Clustering** | Group millions of photos by visual similarity | FAISS GPU spherical $k$-means clustering, semantic drift outlier tracking, and hierarchical partitioning. | [Guide ➡️](pipeline/06_clustering_dynamic_drift.md) |
| **07. MLLM Cluster Labeling** | Ask an AI to write descriptive captions for each cluster | Generates film-strip collages of cluster medoids and prompts VLMs (via SGLang Docker) for ecological labels. | [Guide ➡️](pipeline/07_mllm_labeling.md) |
| **08. Spatial-Semantic Indexing** | Build a sub-millisecond global search index | Multi-resolution H3 hexagonal aggregation mapping geography to visual categories. | [Guide ➡️](pipeline/08_spatial_semantic_indexing.md) |
| **09. Visualization Dashboards** | Render interactive maps and exploration charts | Leaflet density/semantic maps, WebGL 2D UMAP projections, and HTML cluster galleries. | [Guide ➡️](pipeline/09_visualization_dashboards.md) |

---

## ⚡ 3. Offline Pipeline & Model Representation Sweeps (`scripts/pipeline/run_offline_pipeline.sh`)

When working with pre-downloaded offline datasets (e.g. `geo_space_cleaned_offline.parquet`, camera trap subsets, or benchmark extracts), the scraping, cell-chunking, and network downloading stages can be skipped entirely.

The offline pipeline (`scripts/pipeline/run_offline_pipeline.sh`, configured via `config/pipeline/params_offline.yaml`) allows rapid evaluation across different vision backbones (`google/tipsv2-b14`, `facebook/dinov2-base`, SigLIP, etc.), pooling strategies (`cls`, `avg_patch`, `cls_avg_patch`), and cluster counts ($k$).

### Two-Dataset Architecture & Automated Image Synchronization

Geo-RAG maintains a clean separation between remote and local datasets:

1. **Master Dataset (`full_dataset_parquet`)**: The canonical dataset preserving all original remote URLs (`Image_URL = https://...`).
2. **Offline Dataset (`input_parquet`)**: The self-contained operational dataset with local relative paths (`Image_Location = ./images/<platform>/<photo_id>.jpg`) and companion embeddings (`.npy` + `.keys.parquet`).

When new images are appended to the master dataset, pass `--run_download_images` to sync the offline dataset automatically:

```bash
./scripts/pipeline/run_offline_pipeline.sh \
  --input /path/to/geo_space_cleaned_offline.parquet \
  --full_dataset_parquet /path/to/geo_space_cleaned.parquet \
  --run_download_images \
  --download_output_dir /path/to/images \
  --image_root_dirs /path/to/images /path/to/other_cam_traps \
  --output_dir /path/to/output \
  --model_name "google/tipsv2-b14" \
  --representation_type "cls" \
  --precision "float16" \
  --run_backfill_embeddings \
  --k_clusters 40000
```

* **Step 0 (Download / Sync)**: Reads `--full_dataset_parquet`, searches existing offline files across `--image_root_dirs` (storage locations on local PC), copies discovered images into `--download_output_dir`, downloads missing deltas with streaming atomic checkpoints into `--download_output_dir`, and appends them to `--input` (metadata-only via `--skip_embeddings`, strictly decoupling image retrieval from model representations).
* **Step 0.5 (Backfill Embeddings)**: Reads `--input` with `--resume`, skips already embedded images, loads images directly from `--download_output_dir`, computes embeddings only for the newly downloaded deltas (for the exact `--model_name`, `--representation_type`, and `--precision` configured), and updates companion `.npy` matrices.
* **Steps 1–5**: Performs clustering, spatial indexing, medoid labeling, and interactive visualization generation seamlessly, reading images directly from `--download_output_dir`.

### Zero-Inference Embedding Derivation & CNN Aggregation (`backfill_embeddings.py`)

When running representation sweeps across different pooling variants (`cls`, `avg_patch`, `cls_avg_patch`), `backfill_embeddings.py` automatically detects existing companion files and derives representations mathematically without GPU model inference:

* **ViT Decomposition (`cls_avg_patch` $\to$ `cls` / `avg_patch`)**:
  * In Vision Transformers (ViT, TIPSv2, CLIP, DINOv2, SigLIP), `cls_avg_patch` is stored as direct concatenation $[\mathbf{cls} \parallel \mathbf{avg\_patch}] \in \mathbb{R}^{2D}$.
  * If a sweep requests `cls`, `backfill_embeddings.py` automatically slices the first half (`[:, :D]`).
  * If a sweep requests `avg_patch`, it automatically slices the second half (`[:, D:]`).
  * Slicing operates directly via zero-copy NumPy memory maps (`mmap_mode="r"`), completing in seconds across millions of vectors without downloading images or loading PyTorch models into GPU VRAM.
* **ViT Concatenation (`cls` + `avg_patch` $\to$ `cls_avg_patch`)**:
  * If `cls_avg_patch` is requested and separate `_cls_embeddings.npy` and `_avg_patch_embeddings.npy` files exist, they are automatically concatenated along axis 1 with key alignment.
* **Dynamic Architecture Aggregation & Zero-Inference Aliasing**:
  * Models without discrete `[CLS]` tokens—including CNNs (ResNet, ConvNeXt, EfficientNet) and CLS-less Vision Transformers (Swin Transformer, GAP-pooled ViTs, PVT)—reduce spatial tokens via Global Average Pooling (GAP) or patch averaging, making `cls` and `avg_patch` mathematically identical.
  * Rather than relying on fragile hard-coded architecture lists, `backfill_embeddings.py` dynamically inspects `cls_token` and `num_prefix_tokens` via `model_has_cls_token` using PyTorch's `meta` device (allocating **0 bytes of RAM/VRAM** and **0 weight downloads** in ~2ms).
  * Existing spatial pooled embeddings are automatically reused across `cls` and `avg_patch` interchangeably without recomputation.
* **Full Override Control**: Pass `--no_resume` or `--force` to bypass automatic derivation and force recomputation from scratch.

### Model-Agnostic Offline Dataset & Multi-Model Embeddings Synchronization (`sync_offline_dataset.py`)

When upstream coordinate anomaly cleanup (`cleanup_coordinate_anomalies.py`), duplicate pruning, or sequence purging removes corrupted records from the master online dataset (`geo_space_cleaned.parquet`), pre-existing offline datasets (`geo_space_cleaned_offline.parquet`) and their companion embeddings files retain those deleted records unless explicitly synchronized.

Because the offline pipeline may explore different model architectures (e.g. DINOv2, SigLIP, Swin) or multiple backfilled representations than the online pipeline, synchronization is **completely model-agnostic**:

1. **Automatic Multi-Model & Clustered Sidecar Discovery**:
   Scans the offline directory (and optional `--extra_dirs`, e.g. pipeline output directories) for all companion files matching `{base_name}_*_embeddings.keys.parquet` / `{base_name}_*_embeddings.npy` as well as companion clustered sidecars matching `*clustered_k_*.parquet`. It automatically detects companion matrices and cluster assignments for **every model and sweep variant** present on disk.
2. **Metadata Pruning**:
   Extracts `photo_key` sets from the master online dataset and filters `input_parquet` in-place using atomic temporary files, removing any rows that were purged upstream.
3. **Zero-Inference Embedding Slicing**:
   For each discovered companion pair, slices the `.npy` matrix via memory mapping (`mmap_mode="r"`) to match the surviving keys, and updates `.keys.parquet` atomically. This takes $<1$s per model without GPU forward passes or model weight loading.
4. **Companion Clustered Sidecar Reconciliation**:
   Prunes pre-existing clustered sidecar Parquet files (`*_clustered_k_*.parquet`) in-place so cluster assignments stay 100% synchronized with the base dataset, preventing stale cluster mappings or missing key warnings during downstream indexing.
5. **Physical Image Deletion (Optional)**:
   When `--delete_images` (or `--delete_orphaned_images`) is enabled, identifies local image files (`Image_Location`) for purged records and removes them from disk to reclaim storage space.

**Running as a Standalone Command**:

```bash
python -m src.utils.sync_offline_dataset \
  --online full_pipeline_output/geo_space_cleaned.parquet \
  --offline full_pipeline_output/geo_space_cleaned_offline.parquet \
  --image_dir /path/to/images \
  --delete_images
```

**Running within the Offline Pipeline**:
Pass `--sync_offline` to `scripts/pipeline/run_offline_pipeline.sh` (or set `sync_offline_dataset: true` in `config/pipeline/params_offline.yaml`):

```bash
./scripts/pipeline/run_offline_pipeline.sh \
  --input /path/to/geo_space_cleaned_offline.parquet \
  --full_dataset_parquet /path/to/geo_space_cleaned.parquet \
  --sync_offline \
  --delete_orphaned_images
```

If the offline dataset is already 100% in sync, this pre-flight step exits in $<1$s with a confirmation checkmark and proceeds.

### Visualizations & Sidecar Layout

To prevent destructive file overwrites across different model sweeps:

* **Visualizations Directory**: All HTML maps, sample grids, UMAP scatter plots, and summary statistics are routed into a dedicated subfolder:
  `vis_${num_clusters}_${model_name}_${representation_type}_${precision}/`
* **Model-Namespaced Sidecar Parquets**: Clustered sidecar files and spatial indexes record the model provenance in their metadata and filenames:
  `${BASE_NAME}_${model_slug}_${representation_type}_${precision}_clustered_k_${k}.parquet`
* **Smart Parent-Directory Resolution**: Downstream loaders (`load_dataset_with_clusters` and `load_embeddings`) automatically traverse parent and sibling directories to discover base metadata and companion `.npy` embeddings matrices without duplicating storage.

---

## 💾 4. Data Versioning (DVC)

To handle heavy files (Parquet databases, HTML maps, images), `scripts/pipeline/run_full_pipeline.sh` implements autonomous DVC standalone tracking:

1. **HDD Storage**: Outputs are written to a fast SSD, then backed up to a high-capacity HDD directory tracked by DVC.
2. **Push**: Pushes heavy data to remote storage using `dvc push`.
3. **Git Sync**: Copies the updated `.dvc` tracking files back to the SSD Git repository, automatically commits them, and pushes them to track repository state changes.

---

## 📚 5. Literature & Software References

* Brodsky, A. (2018). *H3: Uber's Hexagonal Hierarchical Spatial Index*. Uber Engineering. [https://h3geo.org](https://h3geo.org)
* Dhillon, I. S., & Modha, D. S. (2001). Concept decompositions for large sparse text document collections with applications to high-dimensional clustering. *Machine Learning*, 42(1), 143–175.
* Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.
* McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint arXiv:1802.03426*.
* Cao, B., Chen, K., Maninis, K. K., Chen, K., Karpur, A., Xia, Y., ... & Araujo, A. (2026). Tipsv2: Advancing vision-language pretraining with enhanced patch-text alignment. *CVPR 2026*.
* Sculley, D. (2010). Web-scale k-means clustering. In *Proceedings of the 19th International Conference on World Wide Web* (pp. 1177–1178).
* Thorndike, R. L. (1953). Who belongs in the family? *Psychometrika*, 18(4), 267–276.
* Tobler, W. R. (1970). A computer movie simulating urban growth in the Detroit region. *Economic Geography*, 46(sup1), 234–240.
* Beck, H. E., T. R. McVicar, N. Vergopolan, A. Berg, N. J. Lutsko, A. Dufour, Z. Zeng, X. Jiang, A. I. J. M. van Dijk, and D. G. Miralles. High-resolution (1 km) Köppen-Geiger maps for 1901–2099 based on constrained CMIP6 projections. Scientific Data 10, 724 (2023).
* Vo, Huy V., et al. ‘Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach’. *Transactions on Machine Learning Research*, 2024.
