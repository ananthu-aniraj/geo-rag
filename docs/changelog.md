# Changelog

All notable changes and updates to the Geo-RAG codebase are documented here.

---

## [Unreleased]

### Fixed
- **H3 Index Missing Points Groupby Fix**: Resolved a major bug in `build_spatial_semantic_index.py` where rows with `NaN` in `parent_cluster_label` (unlabeled parent clusters) were silently discarded by Pandas `groupby` during index aggregation. This caused 98% of points to be missing from the generated spatial-semantic index. Fixed by filling NaN values in all label and description columns with default fallback values (e.g. `Parent Cluster <ID>`) and adding `dropna=False` to the groupby.
- **MLLM Parent Label Streaming Mapping Key Fix**: Fixed a critical bug in the output Parquet streaming writer of `label_clusters_mllm.py` where parent labels and descriptions were being mapped against `cluster_id` instead of `parent_cluster_id`. Since parent labels are keyed 0–499 while child IDs go up to 39,999, this caused all parent columns to be written as `NaN` for 98.7% of the dataset.
- **VLM Relabeler NaN Label Check Fix**: Resolved a bug in `relabel_failed_clusters.py` where pandas `NaN` values loaded from Parquet (representing missing/failed labels) were not identified as needing relabeling because they evaluate as `float('nan')` instead of `None` or empty strings. Fixed by adding `pd.isna()` checks to the cluster and parent labeling verification loops.
- **PyArrow Push-Down Filter Optimization**: Optimized loading of the massive 46.8-million-row index file (`geo_space_h3_semantic_index.parquet`) in `generate_h3_semantic_map.py`, `query_geo_space.py`, `visualize_cluster_samples.py`, and `dataset_statistics.py`. The scripts now use PyArrow's push-down filters to load only the target resolution data, reducing memory usage by 90% (to ~3.5 GB) and speeding up loading times from minutes to under 2 seconds.
- **MLLM Labeling Array Bound Crash Fix**: Resolved a traceback crash (`IndexError: boolean index did not match indexed array...`) in `label_clusters_mllm.py` during MLLM cluster labeling by allocating arrays based on `max(parent_ids.values()) + 1` to accommodate missing parent IDs from the resampling majority vote.
- **Duplicate Key Reindexing Protection**: Gracefully handle duplicate photo keys in `load_embeddings` (`io.py`) and streaming indexers (`process_scraped_data.py`) by resolving indexing to their first unique occurrence, preventing `InvalidIndexError` when reindexing large datasets.
- **Lowercase Platform and photo_key Standardization**: Standardized database and codebase casings for `Platform` columns and `photo_key` identifiers to lowercase globally (in `process_scraped_data.py` and `io.py`). This prevents duplicate entries when resuming scraping runs or merging datasets with mixed platform casings (e.g. 'Flickr' vs 'flickr').
- **EUNIS 2024 Code-Based Legend Resolution**: Keyed the EUNIS 2024 legend mapping using the classification `Code` attribute rather than the row `Id` index in `spatial_overlays.py`. This resolves coordinate query errors where pixel codes (e.g. `8000`, `2113`) failed to map, fully restoring Level 1, 2, and 3 geobotanical retrieval evaluations.
- **Complete Checkpoint Purge**: Updated the cleanup task handler at the end of `process_scraped_data.py` to completely purge companion `*.keys.parquet`, `.npy` files, and lingering `.tmp` files left over from checkpoints upon successful completion.

### Planned Security Refactoring
- **Pipeline & MLLM Configuration Decoupling**:
  - Restructure core pipeline configurations (`params.yaml` -> `config/pipeline/params.yaml`).
  - Update `run_full_pipeline.sh` to source the `.env` variables and pass them natively to the Docker container environment (e.g. via `--env-file .env`).

## [1.1.2] - 2026-08-20

### Added
- **Hierarchical EUNIS Level 1, 2, and 3 Spatial Retrieval**: Updated the EUNIS Ecosystem Map benchmark (`benchmark_eunis.py`) and the LUCAS overlay benchmark (`benchmark_lucas.py`) to run retrieval precision evaluations across Level 1 (Macro), Level 2 (Meso), and Level 3 (Exact) habitat definitions.
- **Unified EUNIS 2024 Level 3 Dataset Integration**: Swapped out the legacy 2012 ecosystem map for the new 2024 Level 3 EUNIS dominant habitat raster map (`eunis_dominant.tif` and `eunis_legend_detailed.csv`).
- **Dynamic Capping & Uniform Spatial Curation (`sample_by_h3.py`)**: Added `--target_size` CLI argument to automatically calculate the optimal geographic cap per H3 cell using binary search to achieve a target size with maximum spatial uniformity.
- **Unified Spatial Query Helpers**: Created unified coordinates lookup functions (`lookup_environmental_zone` and `lookup_eunis_levels`) inside `src/utils/spatial_overlays.py` to eliminate coordinate query code duplication across `benchmark_environmental_zones.py`, `benchmark_eunis.py`, and `benchmark_lucas.py`.
- **Automated Model Benchmarking Comparison**: Created `compare_models.py` to automate running the benchmarking pipeline across multiple models sequentially, parsing the metrics, and collating them into a single markdown comparison report.
- **Interactive Project Wiki & MathJax Support**: Added a Material for MkDocs configuration (`config/wiki/mkdocs.yml`) with support for Mermaid diagrams and MathJax LaTeX equation rendering (delimiters: `$ ... $`, `$$ ... $$`). Relocated the changelog inside the `docs/` folder and created a site homepage.
- **Configurable CLIP Comparison**: Added the `compare_clip` parameter to the Places365 config, allowing users to toggle CLIP baselines directly from the YAML.
- **Sanitized Model-Specific Output Reports**: Appended clean model names to benchmark report text and CSV files (e.g. `lucas_report_google_tipsv2-b14.txt`), preventing concurrent runs from overwriting each other.

### Changed
- **Consolidated EUNIS Mappings**: Replaced the complex and multi-level fallback DBF mapping logic with direct CSV legend parsing from `load_eunis_legend()`.
- **Streamlined Model Inference**: Refactored the vision model loading and inference layer to focus strictly on `timm` and `TIPSv2` models, making the code much easier to maintain.
- **Lazy Class Token Verification**: Implemented lazy checks to query and cache class token presence (`cls_token` or `num_prefix_tokens > 0`) on model instances during the first forward pass, enabling support for models without a standard CLS token.
- **Robust Boolean Parsing**: Standardized bash wrappers to support both lowercase (`false`) and title-case (`False`) outputs from PyYAML parser lookups, correcting issues where flags like `use_segformer: false` were ignored.
- **Dynamic Mapillary API Keying in Evaluations**: Replaced hardcoded Mapillary credential variables in the spatial evaluation scripts with dynamic `--mapillary_token` parameters populated from the `.env` environment.

## [1.1.1] - 2026-08-18

### Added
- **Centralized Secrets Environment**: Created a gitignored `.env` file in the project root to store Flickr, Mapillary, and Hugging Face tokens, along with a committed `.env.template` setup template.
- **Scraper Configuration Decoupling**: Created a new `config/scrapers/` folder containing separate YAML parameters for all 7 scraping utilities.
- **Refactored Scraper Scripts**: Updated all 7 scraper shell scripts in `scripts/scrapers/` to load their parameters from `config/scrapers/*.yaml` and credentials from `.env`, removing all fallback references to the main `params.yaml` file.
- **Evaluation Configuration Decoupling**: Relocated root evaluation configs (`eval_params_offline.yaml` and `eval_params_online.yaml`) into a new `config/evaluation/` directory as `params_offline.yaml` and `params_online.yaml`, and created dedicated YAML parameter configurations for the LUCAS and Places365 visual model evaluation runners.
- **Refactored Evaluation Scripts**: Updated all 4 evaluation shell runners in `scripts/evaluation/` to dynamically load parameters from the new `config/evaluation/` YAML files, fixing relative path bugs and imports.
- **Multi-Model Benchmarking Support**: Added a unified `load_vision_model` helper supporting `timm` models and custom local/HF `TIPSv2` models, dynamically resolving input resolution, normalization, and transforms.
- **Dynamic Prefix and Patch Extraction**: Updated the feature extraction pipelines in all 4 python benchmarks to automatically respect prefix tokens (using `model.num_prefix_tokens`) and dynamically compute grid sizes and patch boundaries at run-time (removing hardcoded patch-size expectations).
- **Configurable YAML Model Names**: Exposed `model_name` attributes in the offline and online evaluation configs, enabling seamless comparisons across diverse representation models (like DINOv2, SigLIP, or CLIP via timm) by updating the YAML parameters.
- **Optional SegFormer Speedup Toggle**: Added `--no_segformer` CLI argument and `use_segformer` YAML configuration support, enabling users to completely bypass SegFormer model loading and background segmentation checks for a massive (several times) performance speedup.

## [1.1.0] - 2026-08-17

### Added
- **Directory Refactoring (Shell Scripts)**: Consolidated 11 of the 12 shell scripts from the repository root into functional subdirectories:
  - `/scripts/scrapers/` for scraping loops and density profilers.
  - `/scripts/evaluation/` for Places365, LUCAS, and spatial benchmarks.
- **Relocatable Path Safety**: Prepended auto-resolution headers to all migrated scripts (`SCRIPT_DIR` + `PROJECT_ROOT`) so they dynamically resolve imports and config relative to the project root directory, allowing safe execution from anywhere.
- **Unified Format Ingestion (CSV & Parquet)**: Updated `process_scraped_data.py` to accept both `.csv` and `.parquet` files as raw input scraping sources (found via `--dirs` and `--offline_dataset_dirs`). Included safety filtering to ignore keys, checkpoints, and output databases during ingestion.
- **Precomputed Embedding Bypass**: If an ingested Parquet file has precomputed embeddings (e.g., from `backfill_embeddings.py`), `process_cell` dynamically loads them and bypasses both the image downloading and the GPU-based TIPSv2 model forward pass.
- **Strict Representation Type Checking**: Added verification of filename representation suffixes (e.g. `_cls_embeddings.npy` vs `_avg_patch_`) in `load_embeddings()` to prevent loading mismatched vector configurations.
- **First-Match Column Fallbacks**: Implemented first-match fallback column mapping (`local_path`, `Image_Location`, `file_name`, `path`) to prevent duplicate column collisions (e.g. when a dataset contains both `Image_Location` and `file_name`).
- **Configurable API Arguments**: Added command-line arguments (`--api_key` and `--token`) to scrapers and density profilers to allow running them as standalone tools with explicit keys without modifying files.
- **Profiler Shell Wrappers**: Added `run_flickr_density_profiler.sh` and `run_mapillary_density_profiler.sh` wrappers to easily configure and run density profiling scrapers from the command line with token fallbacks and command line location overrides.


### Changed
- **Decoupled API Token Infrastructure**: Moved Mapillary token and Flickr API key out of source code files and into the centralized config file `params.yaml`. Updated shell wrappers (`run_flickr_scraper.sh`, `run_mapillary_scraper.sh`, and `run_full_pipeline.sh`) to automatically parse and export them.
- **Key-Based Embedding Matching**: Completely eliminated legacy `embedding_idx` integers and migrated loaders/writers to match embeddings via the stable `photo_key` string column. Row dropping, filtering, and joining are now 100% safe from alignment drift.
- **Enforced Parquet backfilling**: Refactored `backfill_embeddings.py` to save data utilizing `save_dataframe()`, automatically enforcing `.parquet` output extension, saving companion `.npy` matrices, and generating `.keys.parquet` index files.

### Fixed
- **VLM Parent Cluster Path Resolution**: Fixed a bug in `label_clusters_mllm.py` where representative image tasks for parent clusters omitted `photo_id` and `platform` keys, which broke offline path resolution for parent cluster labeling.
