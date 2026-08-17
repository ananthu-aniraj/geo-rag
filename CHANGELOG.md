# Changelog

All notable changes and updates to the Geo-RAG codebase are documented here.

---

## [Unreleased]

### Planned Security Refactoring
- **Credential Decoupling & Ignored Configuration**:
  - Untrack and add local `params.yaml` to `.gitignore` to prevent committing active keys.
  - Create a `params.template.yaml` template file for tracking configuration parameters.
  - Clear hardcoded API keys and tokens from all shell scripts, letting them fall back to local `params.yaml` or env variables.

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
