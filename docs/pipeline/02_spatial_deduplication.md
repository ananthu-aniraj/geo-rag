# Step 1: Spatial Deduplication & Filtering

This document describes the design and operation of `process_scraped_data.py`, which performs image processing, vision model feature extraction, spatial deduplication, and zero-shot filtering.

---

## ⚙️ Core Operation

The script takes a raw scraped database, partitions coordinates into **H3 Resolution 11 parent cells** (each spanning ~2,000 m²), and performs spatial-temporal deduplication using TIPSv2 image embeddings to ensure uniform geographic coverage.

### 1. Multi-Representation & Precision Customization
Through command-line options and the master `params.yaml`, the pipeline supports customizable representations and storage layouts:
* **`--representation_type`**:
  * `cls`: Standard CLS token ($768$ dimensions).
  * `avg_patch`: Attention-weighted average of vision patch tokens ($768$ dimensions).
  * `cls_avg_patch`: Concatenated CLS + Average Patch tokens ($1536$ dimensions).
* **`--precision`**:
  * `float32`: High-precision float storage.
  * `float16`: Half-precision float storage. This downcasts the final matrix right before writing to disk, reducing SSD storage footprint by **50%** (saving ~8 GB on a 16 GB database). Slicing/loading routines automatically upcast the segments back to `float32` in RAM for downstream model compatibility.

### 2. Single-Pass Feature Extraction
When computing concatenated representations (`cls_avg_patch`), rather than running two separate forward passes, the script uses `extract_model_embeddings` from `src.models.vision_model_inference`. It executes transformer blocks $1$ through $11$ once and branches only at the $12\text{th}$ block, yielding both CLS and value attention patch projections in practically the cost of a single standard forward pass.

---

## 📐 Decoupled Storage Layout

To prevent Parquet file bloating and RAM starvation, the database uses a decoupled storage architecture:
* **Lightweight Parquet File**: The `.parquet` output holds only metadata columns (Photo ID, Platform, coordinates, H3 cell) and a stable unique `photo_key` column (formatted as `{Platform}_{Photo_ID}`). No heavy vectors are stored inside the Parquet format.
* **Companion NumPy Binary File**: The embedding vectors are stacked in a dense NumPy matrix and saved to an independent file named `{core_name}_{representation_type}_embeddings.npy` (e.g., `geo_space_cls_avg_patch_embeddings.npy`).
* **Keys Index File**: A companion index file named `{core_name}_{representation_type}_embeddings.keys.parquet` stores the ordered list of `photo_key` values matching the rows of the `.npy` matrix.
* **Alignment**: The loader dynamically resolves alignment by matching the metadata's `photo_key` against the companion `keys.parquet` index.

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

### 1. Flickr Indoor/Outdoor Filter
Filters out indoor photos using zero-shot text-image classification with TIPSv2. Images are compared against the prompts:
* *"An indoor scene"*
* *"An outdoor landscape or street view"*

If an image matches the indoor class, it is discarded. This filter applies **only to Flickr images** (since street-view platforms like Mapillary/KartaView are intrinsically outdoor, and iNaturalist observations are filtered by macro characteristics). Can be bypassed with the `--no_filter` flag.

### 2. Macro Filter (`--filter_macro`)
Filters out macro close-up flora/fauna images (typically for iNaturalist data) using a zero-shot *"A macro/close-up photo of flowers, leaves, bark, or insects"* classifier.

### 3. Sky Filter (`--filter_sky`)
Filters out empty sky views (typically for iNaturalist observations) using a zero-shot *"A view of empty sky, clouds, or flying objects"* classifier.
