# Bulk Image Downloader & Offline Migration Guide

The `src/utils/download_images.py` utility transforms online, URL-referenced datasets into verified, self-contained offline image archives. It is designed to handle multi-million-row datasets with multi-threaded downloads, streaming atomic checkpoints, companion embedding synchronization, and local directory consolidation.

---

## 🚀 Key Features

* **Multi-Platform Support**: Downloads and normalizes imagery from Flickr, Mapillary, iNaturalist, Wildlife Insights, WildObs, KartaView, and direct HTTP/HTTPS URLs.
* **Fault-Tolerant Resuming (`--resume`, `--resume_from`)**: Automatically detects existing output records on disk, skips verified images, and downloads only missing candidates from the input dataset.
* **Streaming Atomic Checkpoints (`--checkpoint_interval`)**: Periodically flushes new image records and companion embeddings to disk every $N$ seconds (default: 60s) via atomic temporary file renaming, preventing data corruption upon process interruption.
* **Offline Directory Migration (`--copy_offline_images`)**: Scans existing local folders (`--image_root_dirs`) for images, copies them into a standardized `<output_dir>/<platform>/<photo_id>.jpg` structure, and rewrites dataset paths accordingly.
* **Companion Embedding Synchronization**: Slices and writes matching companion NumPy matrices (`.npy`) and keys indices (`.keys.parquet`), ensuring representations remain aligned row-for-row with downloaded metadata.
* **Bulk Mapillary Graph API Caching**: Resolves virtual `mapillary://` URIs in batches of 250–500 IDs per request using the Graph API multi-ID endpoint, caching signed 30-day CDN URLs to disk (`.mapillary_url_cache.parquet`).
* **Wildlife Insights GraphQL Authentication**: Resolves signed Google Cloud Storage asset URLs dynamically using session cookies or JWT tokens.
* **Binary Integrity & Placeholder Detection**: Inspects magic bytes to discard corrupted files or HTML login walls, and verifies MD5 hashes against known placeholder images (such as Flickr's "photo unavailable" GIF).

---

## 📋 Common Usage Examples

### 1. Basic Multi-Threaded Download

Download images for any dataset (Parquet or CSV) into an offline directory:

```bash
PYTHONPATH=. python3 src/utils/download_images.py \
  --input data/scraped/flickr_sample.parquet \
  --output_dir data/offline/images \
  --output data/offline/flickr_sample_offline.parquet \
  --threads 32
```

The output Parquet file will update `Image_Location`, `file_name`, and `Image_URL` to point to the local file path (e.g. `./data/offline/images/flickr/12345.jpg`).

### 2. Resuming an Interrupted Download

If a download is cancelled or loses network connectivity, resume it safely with `--resume`:

```bash
PYTHONPATH=. python3 src/utils/download_images.py \
  --input data/scraped/large_dataset.parquet \
  --output_dir data/offline/images \
  --output data/offline/large_dataset_offline.parquet \
  --resume \
  --checkpoint_interval 60 \
  --threads 32
```

* **Candidate Filtering**: Automatically checks `large_dataset_offline.parquet` and skips all records where local image files already exist and pass binary validation.
* **Streaming Updates**: Saves incremental progress every 60 seconds.

### 3. Migrating & Consolidating Existing Local Images

When you already have offline images scattered across local folders (e.g. from camera trap exports, prior scraping runs, or external hard drives), use `--copy_offline_images`:

```bash
PYTHONPATH=. python3 src/utils/download_images.py \
  --input data/scraped/camera_traps.parquet \
  --output_dir data/offline/images \
  --output data/offline/camera_traps_offline.parquet \
  --image_root_dirs /mnt/storage/raw_cam_traps /data/local_cache \
  --copy_offline_images \
  --resume
```

* Existing files matching `{Photo_ID}.jpg` or `{platform}/{Photo_ID}.jpg` inside `--image_root_dirs` are copied to `--output_dir/<platform>/<photo_id>.jpg`.
* Only images missing from local directories are downloaded from the web.

### 4. Downloading Datasets with Companion Embeddings

If the input dataset has precomputed embeddings (e.g. `geo_space_cls_embeddings.npy` and `geo_space_cls_embeddings.keys.parquet`):

```bash
PYTHONPATH=. python3 src/utils/download_images.py \
  --input data/pipeline/geo_space_deduplicated.parquet \
  --output_dir data/offline/images \
  --output data/offline/geo_space_offline.parquet \
  --representation_type cls \
  --precision float32 \
  --resume
```

* The downloader loads the companion embeddings matrix using `src.utils.io.load_embeddings`.
* As new images are downloaded and saved, the matching embedding vectors are sliced and written to companion files (`geo_space_offline_cls_embeddings.npy` and `geo_space_offline_cls_embeddings.keys.parquet`).

### 5. Authenticated Downloads (Mapillary & Wildlife Insights)

Credentials can be stored in `.env` or passed via command-line arguments:

```bash
PYTHONPATH=. python3 src/utils/download_images.py \
  --input data/scraped/snapshot_usa_filtered.parquet \
  --output_dir data/offline/images \
  --output data/offline/snapshot_usa_offline.parquet \
  --wildlife_cookie "connect.sid=s%3A..." \
  --mapillary_token "MLY|..." \
  --mapillary_batch_size 250 \
  --threads 24
```

---

## ⚙️ CLI Parameter Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `str` | *(Required)* | Path to input dataset (`.parquet` or `.csv`). |
| `--output_dir` | `str` | `"./downloaded_images"` | Directory where downloaded/copied image files are saved. |
| `--output` | `str` | `None` | Path for updated output metadata file (defaults to `[input_base]_offline.parquet`). |
| `--threads` | `int` | `16` | Number of concurrent download worker threads. |
| `--resume` | `flag` | `False` | Resume downloading by skipping images already present in the output Parquet file. |
| `--resume_from` | `str` | `None` | Explicit path to existing output Parquet file to resume from. |
| `--checkpoint_interval`| `int` | `60` | Interval in seconds for atomic streaming checkpoints (set to `0` to disable). |
| `--copy_offline_images`| `flag` | `False` | Copy existing offline images found in `--image_root_dirs` into `--output_dir`. |
| `--image_root_dirs` | `str` (list) | `None` | Local directories to search for existing offline images. |
| `--representation_type`| `str` | `"cls"` | Representation type for companion embeddings (`cls`, `avg_patch`, `cls_avg_patch`). |
| `--precision` | `str` | `"float32"` | Storage precision for companion embeddings (`float32` or `float16`). |
| `--timeout` | `int` | `20` | Request timeout in seconds per image download. |
| `--max_retries` | `int` | `3` | Maximum retry attempts for failed downloads with exponential backoff. |
| `--mapillary_token` | `str` | `None` | Mapillary access token (overrides `MAPILLARY_TOKEN` in `.env`). |
| `--mapillary_batch_size` | `int` | `250` | Number of photo IDs per Mapillary Graph API request. |
| `--mapillary_resolver_threads` | `int` | `4` | Number of concurrent threads for bulk Mapillary URL resolution. |
| `--wildlife_cookie` | `str` | `None` | Wildlife Insights session cookie (`connect.sid=...`). |
| `--wildlife_token` | `str` | `None` | Wildlife Insights JWT authorization token. |

---

## 🔄 Integration with the Geo-RAG Pipeline

Once an offline dataset is generated via `download_images.py`, it can be ingested directly into the spatial deduplication pipeline:

```bash
PYTHONPATH=. python3 src/processing/process_scraped_data.py \
  --dirs data/offline/flickr_sample_offline.parquet \
  --offline_dataset_dirs data/offline/images \
  --output_dir full_pipeline_output
```

Because `Image_Location` points to verified local image files and companion embeddings are synchronized, `process_scraped_data.py` reads them with zero network overhead and bypasses redundant GPU forward passes.
