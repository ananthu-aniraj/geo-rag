# Step 1: Spatial Deduplication & Filtering

This document describes the design and operation of `process_scraped_data.py`, which performs image processing, vision model feature extraction, spatial deduplication, and zero-shot filtering.

---

## ⚙️ Core Operation

The script takes a raw scraped database, partitions coordinates into **H3 Resolution 11 parent cells** (each spanning ~2,000 m²), and performs spatial-temporal deduplication using deep vision model embeddings to ensure uniform geographic coverage.

### 1. Multi-Model Support

The pipeline is fully decoupled from a single vision architecture and supports arbitrary vision encoders via `--model_name`:

* **Supported Architectures**:
  * **TIPSv2 (Default: `google/tipsv2-b14`)**: Hugging Face remote-code wrapper or local checkpoints (`--tips_model_path`, `--tips_model_variant`, `--tips_low_res`).
  * **`timm` Models**: Vision Transformers, Swin, ConvNeXt, ResNet (e.g. `vit_base_patch16_224`, `vit_large_patch14_clip_224.openai`).
  * **Hugging Face Models**: AutoModel encoders (e.g. `facebook/dinov2-base`, CLIP).
* **Dynamic Resolution & Transforms**: Input resolution and preprocessing transforms are resolved automatically via `load_vision_model` (e.g. $448 \times 448$ for TIPSv2, $224 \times 224$ for standard `timm` models). Background download workers resize incoming imagery to the model's exact target resolution on the fly.

### 2. Multi-Representation & Precision Customization

Through command-line options and the master `params.yaml`, the pipeline supports customizable representations and storage layouts:

* **`--representation_type`**:
  * `cls`: Standard CLS token (or global average pooled features for models lacking a class token).
  * `avg_patch`: Average of spatial patch tokens.
  * `cls_avg_patch`: Concatenated CLS + Average Patch tokens.
* **`--precision`**:
  * `float32`: High-precision float storage.
  * `float16`: Half-precision float storage. This downcasts the final matrix right before writing to disk, reducing SSD storage footprint by **50%** (saving ~8 GB on a 16 GB database). Slicing/loading routines automatically upcast the segments back to `float32` in RAM for downstream model compatibility.

### 3. Fast Regular Inference Engine (`extract_regular_embeddings`)

Rather than relying on complex, hook-based benchmark extraction loops, the pipeline uses native forward inference:
$$\text{image} \longrightarrow \text{transform} \longrightarrow \text{model} \longrightarrow (\text{CLS token}, \text{patch tokens})$$

* **GPU Tensor Acceleration**: All pooling (`avg_patch`), selection (`cls`), and concatenation (`cls_avg_patch`) operations are performed as tensor operations on the GPU prior to host memory transfer.
* **Zero PCIe Bottleneck**: Eliminates host-transfer of bulky raw patch token sequences across the PCIe bus, saving significant memory bandwidth and avoiding CPU-side reshaping overhead.

---

## 📐 Decoupled Storage Layout & Model Disambiguation

To prevent Parquet file bloating and RAM starvation, the database uses a decoupled storage architecture:

* **Lightweight Parquet File**: The `.parquet` output holds only metadata columns (Photo ID, Platform, coordinates, H3 cell) and a stable unique `photo_key` column (formatted as `{Platform}_{Photo_ID}`). No heavy vectors are stored inside the Parquet format.
* **Companion NumPy Binary File**: The embedding vectors are stacked in a dense NumPy matrix and saved to an independent file named:
  * For TIPSv2 (`google/tipsv2-b14`): `{core_name}_{representation_type}_embeddings.npy` (e.g., `geo_space_cls_avg_patch_embeddings.npy`). Files without a model name suffix represent TIPSv2 by convention for backward compatibility.
  * For other models: `{core_name}_{model_name}_{representation_type}_embeddings.npy` (e.g., `geo_space_timm_vit_base_patch16_224_cls_embeddings.npy`).
* **Keys Index File**: A companion index file named `[embedding_filename].keys.parquet` stores the ordered list of `photo_key` values matching the rows of the `.npy` matrix.
* **Missing Embedding Validation**: If an embedding matrix for a non-TIPSv2 model or a non-CLS representation is not found on disk when resuming, the loader raises a descriptive `FileNotFoundError` providing the exact `python -m src.processing.backfill_embeddings` command required to backfill representations.

---

## ⚙️ Unified Ingestion & Offline Dataset Support

The deduplication pipeline has been upgraded to support seamless ingestion of pre-computed offline datasets:

### 1. Unified Ingestion (CSV & Parquet)

* The pipeline accepts both **`.csv`** and **`.parquet`** files in the input arguments (`--dirs` and `--offline_dataset_dirs`), supporting both **directory paths** (which glob all valid `.csv`/`.parquet` files) and **direct file paths** (allowing ingestion of a specific target file).
* Helper index files (`*.keys.parquet`), checkpoint files (`*_checkpoint.parquet`), and the output database itself (`{output_name}.parquet`) are automatically ignored during directory scanning to prevent circular ingestion.

### 2. Precomputed Embeddings Bypass

* When loading input `.parquet` files, the loader automatically resolves and maps their companion `.npy` embeddings.
* During cell-by-cell deduplication, images with matching precomputed embeddings **completely bypass image downloading and TIPSv2 GPU forward passes**, saving massive network bandwidth and compute resources.

### 3. Strict Representation Type Checking

* Precomputed embeddings are validated against the requested `representation_type` (e.g. `cls`, `avg_patch`, `cls_avg_patch`).
* Suffix verification (e.g., checking for `_cls_embeddings.npy` in the filename) prevents loading mismatched vector configurations. Mismatched or missing representation vectors are scheduled for re-inference.

### 4. Schema Normalization & Path Resolution

* Columns from diverse datasets (such as lowercase/camelCase fields like `photo_id`, `Captured_At`, `latitude`) are normalized into the canonical PascalCase schema.
* To prevent duplicate column name collisions (e.g. if a dataset has both `Image_Location` and `file_name` columns), a first-match fallback strategy is used for locating image URLs.

### 5. Camera Trap & Platform Licensing Fallbacks

* **Camera Trap Support**: Camera trap platforms (`SnapshotUSA`, `wildlife_insights`, `iWildCam`) are natively recognized and preserved as outdoor scene sensors (bypassing indoor Flickr filters and iNaturalist macro/sky filters).
* **Automatic License Population**: Missing platform licenses are automatically populated with standard defaults if absent (`CC0` for Snapshot USA, `CDLA-Permissive-1.0` for iWildCam, `CC BY-SA 4.0` for Mapillary/KartaView, and `CC BY-NC 4.0` for iNaturalist).

---

## 🏎️ Million-Row Streaming Optimization

To process datasets in the millions without Out-of-Memory (OOM) errors, the script dynamically identifies **active H3 cells** (cells containing new scraped images) and loads only their existing embeddings from the Parquet database using `pyarrow.dataset` (bypassing the other 99% in-memory). It then writes updates atomically using a custom `stream_update_parquet` streaming engine that filters, matches, and appends chunk-by-chunk. This streaming process is now **100% key-driven**, utilizing the unique `photo_key` strings and C-accelerated hash lookups to dynamically resolve embeddings. This makes updates completely immune to index-shift corruptions.

### Parallel Network Engine

To optimize ingestion speed and minimize network bottlenecks:

* **HTTP Connection Pooling**: A thread-safe global `requests.Session` with a connection adapter (128 maximum connections) keeps sockets alive across download threads, eliminating TCP/SSL handshake latency.
* **On-The-Fly Background Resizing**: Images are resized to `448x448` immediately inside background download threads before passing them to the main thread. This reduces the RAM footprint per image to ~602 KB, allowing you to safely scale the `--cell_chunk_size` (e.g. to `256` or `512` images in parallel per cell block) without OOM risks.
* **Thread Concurrency & Batching**: Uses 64 concurrent download threads. The GPU batch size can be set via `--batch_size` (e.g., to `128` or `256` for modern GPUs) to maximize inference throughput.
* **Streaming Output**: CSV metadata is exported using a streaming row-group writer to avoid buffering embeddings in memory.

---

## 🤖 Zero-Shot Noise Filters

> [!NOTE]
> Zero-shot noise filtering requires a multimodal vision-language model equipped with text encoding capabilities (such as TIPSv2 via `encode_text`). For pure vision backbones (e.g. `timm` models or standard DINOv2), text filtering is automatically and gracefully bypassed while spatial and cosine deduplication continue normally.

### 1. Flickr Indoor/Outdoor Filter

Filters out indoor photos using zero-shot text-image classification with TIPSv2 (or any model supporting `encode_text`). Images are compared against the prompts:

* *"An indoor scene"*
* *"An outdoor landscape or street view"*

If an image matches the indoor class, it is discarded. This filter applies **only to Flickr images** (since street-view platforms like Mapillary/KartaView are intrinsically outdoor, and iNaturalist observations are filtered by macro characteristics). Can be bypassed with the `--no_filter` flag.

### 2. Macro Filter (`--filter_macro`)

Filters out macro close-up flora/fauna images (typically for iNaturalist data) using a zero-shot *"A macro/close-up photo of flowers, leaves, bark, or insects"* classifier.

### 3. Sky Filter (`--filter_sky`)

Filters out empty sky views (typically for iNaturalist observations) using a zero-shot *"A view of empty sky, clouds, or flying objects"* classifier.
