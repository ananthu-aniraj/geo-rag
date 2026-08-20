# Changelog

All notable changes and updates to the Geo-RAG codebase are documented here.

---

## [Unreleased]

### Planned Security Refactoring
- **Pipeline & MLLM Configuration Decoupling**:
  - Restructure core pipeline configurations (`params.yaml` -> `config/pipeline/params.yaml`).
  - Update `run_full_pipeline.sh` to source the `.env` variables and pass them natively to the Docker container environment (e.g. via `--env-file .env`).

## [1.1.2] - 2026-08-20

### Added
- **Automated Model Benchmarking Comparison**: Created `compare_models.py` to automate running the benchmarking pipeline across multiple models sequentially, parsing the metrics, and collating them into a single markdown comparison report.
- **Interactive Project Wiki & MathJax Support**: Added a Material for MkDocs configuration (`config/wiki/mkdocs.yml`) with support for Mermaid diagrams and MathJax LaTeX equation rendering (delimiters: `$ ... $`, `$$ ... $$`). Relocated the changelog inside the `docs/` folder and created a site homepage.
- **Configurable CLIP Comparison**: Added the `compare_clip` parameter to the Places365 config, allowing users to toggle CLIP baselines directly from the YAML.
- **Sanitized Model-Specific Output Reports**: Appended clean model names to benchmark report text and CSV files (e.g. `lucas_report_google_tipsv2-b14.txt`), preventing concurrent runs from overwriting each other.

### Changed
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
