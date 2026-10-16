# Step 1c: Coordinate & Remote Biome Anomaly Cleanup

This document describes the design and operation of `cleanup_coordinate_anomalies.py`, which filters out coordinate-locking anomalies, Null-Island/diagonal GPS parser glitches, and misplaced images in remote biomes (such as Antarctica and hyper-arid deserts like the Sahara) caused by faulty contributor GPS units or EXIF corruption.

---

## ⚙️ Core Operation

During crowdsourced scraping, contributor hardware or EXIF parsers can introduce three distinct classes of geographic corruption:

1. **Locked-Latitude Parallels**: Faulty GPS tracking equipment locks onto a single latitude coordinate (parallel) while longitude continues to update, creating straight lines stretching across multiple regions.
2. **Null-Island & Diagonal Coordinate Glitches**: Uninitialized GPS receivers default to `(0, 0)` (`|Lat| < 0.05` and `|Lon| < 0.05`) or copy-paste parser bugs duplicate a single value into both latitude and longitude (`|Lat - Lon| < 1e-5` for `|Lat| > 1.0°`, e.g. `25.761680, 25.761680` in the Sahara).
3. **Remote-Biome GPS Relocation Glitches**: Hemisphere sign flips (`-Lat` vs. `+Lat`), coordinate swaps, or NMEA offset bugs place temperate highways, forests, or indoor graphics into extreme remote regions such as **Antarctica (`EF` / `ET`)** or **remote hyper-arid deserts (`BWh` / `BWk`)**.

The script scans coordinate distributions and companion visual embeddings (from Step 1a), flags anomalies within the configured target platform(s), and writes an independent cleaned metadata file (`geo_space_cleaned.parquet`), leaving the raw deduplicated database untouched.

---

## 📐 Detector 1: Coordinate Geometry & Parallel Safety Criteria

### 1a. Locked-Latitude Parallels

To prevent false positives, **parallel checking is grouped by platform and continent**:

A rounded latitude parallel $L$ (rounded to 5 decimal places, representing ~1.1 meters precision) is flagged and purged within a specific platform/continent grouping only if:

$$
\text{Count}_{\text{Platform}}(L) \gt 10 \quad \text{and} \quad \text{Longitude Span}_{\text{Platform}}(L) \gt 1.0^{\circ}
$$

* A longitude span threshold of $> 1.0^{\circ}$ (~111 km) ensures that dense city streets or landmarks—which naturally accumulate thousands of images inside small bounding boxes—are completely preserved.

### 1b. Null-Island & Diagonal Locks

Within the configured target platforms and continent scope, points satisfying either of the following deterministic glitch conditions are flagged:

* **Null Island**: $|\text{Latitude}| < 0.05^{\circ}$ and $|\text{Longitude}| < 0.05^{\circ}$.
* **Diagonal Copy-Paste Lock**: $|\text{Latitude} - \text{Longitude}| < 10^{-5}$ with $|\text{Latitude}| > 1.0^{\circ}$.

---

## 🌍 Detector 2: Leave-One-Cell-Out Visual $k$-NN Biome Consistency

Because Step 1c runs **before** global clustering and MLLM labeling, it relies solely on the **Stage 02 companion visual embeddings** (`*_embeddings.npy` + `.keys.parquet`) and **Stage 03 Köppen-Geiger climate codes** (`Koppen_Code`).

### 1. Non-Circular Clean Reference Bank

To avoid circular voting (e.g. a displaced 200-frame Mapillary sequence in the Sahara voting for itself), the detector builds a clean reference bank of up to 15,000 vectors per major Köppen climate group (`A` Tropical, `B` Arid, `C` Temperate, `D` Continental, `E` Polar):

* **Trusted Anchor Prioritization**: Platforms not listed in `--platform` (such as curated scientific/ecological repositories `inaturalist`, `wikimedia`, and camera-trap datasets) are never purged and are prioritized first when sampling reference anchors, supplemented by multi-platform H3 Resolution 3 cells ($\ge 2$ platforms and $\ge 20$ photos).
* **10% Outlier Tail Trimming**: Within each major Köppen group, the bottom 10% of vectors furthest from the group mean direction (by cosine similarity) are trimmed prior to indexing, guaranteeing a pure visual core per biome.

### 2. Leave-One-Cell-Out Cross-Region Voting ($K = 25$)

For candidate images belonging to the target `--platform` list inside extreme biomes:

* **Spatial Exclusion Radius**: When querying the top-$K$ ($K=25$) visual nearest neighbors for candidate image $i$, all reference images inside the **same H3 Resolution 3 cell** (~400 km) or within $\pm 3.0^{\circ}$ latitude/longitude are masked out.
* **Polar / Antarctica (`EF`, `ET`, or `continent == "Antarctica"`)**:
  * Flagged if $\ge 84\%$ (`--incompat_threshold 0.84`) of cross-region visual neighbors originate from incompatible warm/temperate/tropical/arid climates (`A*`, `B*`, `Cfa`, `Cfb`, `Csa`, `Csb`, `Cwa`, `Cwb`, `Dfa`, `Dfb`).
* **Remote Arid Deserts (`BWh`, `BWk` in non-metropolitan H3 Res 4 cells)**:
  * Dense multi-platform desert metropolises ($\ge 500$ images across $\ge 3$ platforms in an H3 Res 4 cell, such as Cairo, Phoenix, Dubai, or Las Vegas) are excluded from candidate scanning to preserve genuine urban oases.
  * In sparse/remote desert cells (e.g. the Sahara interior), an image is flagged if $\ge 84\%$ of its cross-region visual neighbors originate from incompatible humid forest, temperate/continental, or polar climates (`Af`, `Am`, `Cfa`, `Cfb`, `Cfc`, `Dfa`, `Dfb`, `Dfc`, `Dfd`, `EF`, `ET`).

---

## 🗺️ Automated Mapillary Sequence Expansion

Street-level imagery in Mapillary is captured in continuous temporal driving sessions termed **sequences**. When a contributor experiences a hardware lockup or NMEA hemisphere/offset glitch that shifts a European highway drive into the Sahara desert, the entire driving session is contaminated.

To eliminate GPS pollution comprehensively:

1. When any Mapillary photo is flagged by **Detector 1** (locked latitude / coordinate glitch) or **Detector 2** (Visual $k$-NN Biome mismatch), the script loads `MAPILLARY_TOKEN` (from the environment or `.env`).
2. It batch-queries Mapillary Graph API v4 (`https://graph.mapillary.com/?ids=...&fields=id,sequence`) in chunks of 100 to map all flagged points to their parent `sequence_id`s.
3. For every violating sequence, it queries `https://graph.mapillary.com/image_ids?sequence_id={seq_id}` to retrieve all image IDs belonging to that contaminated drive.
4. During streaming Parquet and CSV writes, **the entire sequence** (all member image IDs) is purged from the database alongside the flagged points.
5. If `MAPILLARY_TOKEN` is not present, the script gracefully falls back to point-level purging with an informative notice.

> [!TIP] Synchronizing Existing Offline Datasets
> When coordinate cleaning purges new anomalies or sequences from `geo_space_cleaned.parquet`, any existing offline dataset (`geo_space_cleaned_offline.parquet`), companion embedding matrices (`.npy` and `.keys.parquet`), and clustered sidecars (`*_clustered_k_*.parquet`) can be brought into 100% alignment using `src/utils/sync_offline_dataset.py` (or `./scripts/pipeline/run_offline_pipeline.sh --sync_offline`).

---

## ⚙️ Options & Parameters

You can configure the target platforms and optional continent restriction in `config/pipeline/params.yaml` (and `config/pipeline/params_offline.yaml`):

```yaml
pipeline:
  cleanup_anomalies: true  # Enable coordinate & remote-biome anomaly cleaning
  cleanup_platform:        # Target platforms to clean (other platforms are preserved & used as trusted reference anchors)
    - flickr
    - mapillary
    - kartaview
  cleanup_continent: ""    # (Optional) Restrict cleanup to a specific continent (empty = global)
```

### Command Line Arguments

Alternatively, you can run the script manually with targeted flags:

* `--platform`: Target platform(s) for detection and cleaning (accepts space-separated values or a comma-separated string, e.g. `--platform flickr mapillary kartaview`).
* `--continent`: Restricts detection and cleaning to a specific continent (e.g. `--continent Antarctica` or `--continent Africa`).
* `--model_name` & `--representation_type`: Resolves the matching companion embeddings file (`*_embeddings.npy`) for Detector 2.
* `--disable_biome_knn`: Disables Detector 2 (runs only coordinate geometry checks).
* `--incompat_threshold`: Fraction of incompatible cross-region visual neighbors required to flag a remote-biome anomaly (default: `0.84`).

---

## 🏎️ Performance & Type Resilience

Both Parquet loading and CSV chunk writing perform defensive `pd.to_numeric` conversions on coordinates at startup to prevent float-string mixed schema exceptions during high-precision rounding operations.

---

## 🖥️ Interactive Manual Anomaly Cleaner (`src/processing/interactive_anomaly_cleaner.py`)

For visual inspection and targeted curation of geographic anomalies—specifically Mapillary tracks, sensor glitches, or unwanted platform-specific clusters—Geo-RAG provides a dedicated, lightweight interactive web dashboard.

### Key Architecture & Features

1. **Continent-Restricted & Fast Loading**:
   * Scopes analysis to a single continent (e.g., `--continent Africa` or `--continent Europe`), reducing memory footprint and keeping startup under 8 seconds even for Europe (2.5M+ records).
   * Automatically centers and fits the Leaflet map bounds to the selected continent.

2. **Mapillary Priority & Multi-Platform Filtering**:
   * Color-codes H3 hexagonal cells (e.g. resolution 4, ~177km edge) by Mapillary image density using log scaling.
   * Provides fine-grained removal controls:
     * **Purge Mapillary Only**: Removes erroneous Mapillary tracks while preserving genuine user photos from Flickr, Wikimedia, or iNaturalist in that cell.
     * **Purge Entire Cell**: Discards all photos within the spatial boundary regardless of platform.

3. **Sub-50ms On-Click Image Inspection**:
   * Clicking any hexagon opens a side drawer in under 50 ms.
   * Displays cell-level generic metadata: top countries, Köppen climate codes & descriptions, and centroid coordinates.
   * Renders representative photo samples in a thumbnail grid (querying direct CDN thumbnails via `MAPILLARY_TOKEN`) with clickable links to the Mapillary web app (`https://www.mapillary.com/app/?pKey=...`).

4. **2-Phase Staging & Single-Step Streaming Purge**:
   * **Phase 1 (Instant In-Memory Staging)**: Marking cells updates your "Staged Removals" list in memory with 0 ms latency; flagged hexagons display a prominent red dashed border on the map.
   * **Phase 2 (Single-Step Purge)**: Clicking **"🚀 Execute Purge"** runs a single streaming row-group filter over the Parquet file, filtering rows in C++ memory via PyArrow and atomically writing the cleaned dataset without high RAM consumption.
   * **Reproducible Rules JSON**: Staged decisions are automatically persisted to `manual_h3_removals.json` (and can be exported/imported), making cleanup actions version-controlled and auditable.

### Usage

#### 1. Interactive Web Dashboard

Launch the server pointing to your Parquet dataset and target continent:

```bash
python3 -m src.processing.interactive_anomaly_cleaner \
    --input full_pipeline_output/geo_space_deduplicated.parquet \
    --output full_pipeline_output/geo_space_deduplicated_cleaned.parquet \
    --continent Africa \
    --res 4 \
    --port 8080
```

Then open `http://localhost:8080` in your browser.

#### 2. Headless Batch Execution

To re-apply previously saved removal rules headlessly (e.g. in CI/CD or batch pipelines):

```bash
python3 -m src.processing.interactive_anomaly_cleaner \
    --input full_pipeline_output/geo_space_deduplicated.parquet \
    --output full_pipeline_output/geo_space_deduplicated_cleaned.parquet \
    --rules-file manual_h3_removals.json \
    --batch-purge
```

#### CLI Parameters

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `str` | `.../geo_space_deduplicated.parquet` | Path to input Parquet dataset. |
| `--output` | `str` | `None` (in-place) | Path to cleaned output Parquet dataset. |
| `--continent` | `str` | `"Africa"` | Target continent to inspect (`Africa`, `Europe`, `Asia`, `North America`, etc.). |
| `--res` | `int` | `4` | H3 aggregation grid resolution (`3`, `4`, or `5`). |
| `--port` | `int` | `8080` | Local port to serve the interactive web dashboard. |
| `--host` | `str` | `127.0.0.1` | Host address to bind. |
| `--rules-file` | `str` | `manual_h3_removals.json` | Path to JSON file storing staged removal rules. |
| `--batch-purge` | `flag` | `False` | Apply rules directly from `--rules-file` without starting the server. |

---

## 🧭 Mapillary Trajectory & Kinematic Validation (`src/utils/mapillary_trajectory_validator.py`)

In crowdsourced street imagery (particularly noticeable in regions like Africa), contributors occasionally upload sequences captured from **stationary cameras** (e.g., phones placed indoors, in courtyards, or on tables). Because of poor sky visibility and multipath GNSS interference, recorded coordinates wander erratically in random walks across dozens or hundreds of meters.

Downstream spatial deduplication thins out consecutive frames, leaving sparse remnants scattered across different H3 cells that masquerade as moving captures.

### Kinematic & Geometric Evaluation Rules

When evaluating a Mapillary sequence track $(t_i, \text{lat}_i, \text{lon}_i)_{i=1}^N$, the validator executes three strict tests:

1. **Stationary Camera GPS Jitter**:
   * Condition: Frame count $N \ge 15$, bounding radius $\le 35\text{ m}$, cumulative path length $\ge 100\text{ m}$, and tortuosity $T \ge 15.0$.
   * Identifies cameras that never actually moved while GPS jittered back and forth.
2. **Severe Random-Walk Tortuosity**:
   * Condition: Path length $\ge 250\text{ m}$, net displacement $\le 25\text{ m}$, and tortuosity $T = \frac{L_{\text{total}}}{\max(D_{\text{net}}, 1\text{m})} \ge 20.0$.
   * Filters out chaotic ping-pong wanderings.
3. **Impossible Instantaneous Speed Spikes**:
   * Condition: Max speed $v_{\max} \ge 160\text{ km/h}$ over jumps $> 100\text{ m}$ with $\Delta t \ge 0.5\text{ s}$.
   * Flags teleportation anomalies and corrupted timestamps.

### Persistent SQLite Caching

To guarantee zero redundant API calls across pipeline runs, chunks, or interactive sessions, the validator persists:

* `photo_sequences`: Mapping of `photo_id -> sequence_id`.
* `sequences`: Evaluated sequence metrics, validity verdicts, reasons, and GeoJSON tracks in `data/cache/mapillary_sequences.db`.

---

## 🚜 Batch Continent Sequence Auditor (`src/processing/audit_mapillary_sequences.py`)

To audit an entire continent and purge all corrupted Mapillary images in a single streaming pass:

```bash
# 1. Dry run: scan Africa, inspect violations and reason breakdown without modifying files
python3 -m src.processing.audit_mapillary_sequences \
    --input full_pipeline_output/geo_space_cleaned.parquet \
    --continent Africa \
    --dry_run \
    --report_json full_pipeline_output/africa_mapillary_audit.json

# 2. Execute purge: stream-remove invalid sequence photos from primary and sibling datasets
python3 -m src.processing.audit_mapillary_sequences \
    --input full_pipeline_output/geo_space_deduplicated.parquet \
    --extra_parquets full_pipeline_output/geo_space_cleaned.parquet full_pipeline_output/geo_space_cleaned_hf.parquet \
    --continent Africa
```

> [!TIP]
> **Zero-Redundancy Multi-Dataset Synchronization**:
>
> * **Primary Dataset**: Purged row-group by row-group in streaming mode with minimal RAM overhead.
> * **Additional Parquets (`--extra_parquets`)**: Pass sibling datasets (e.g. `geo_space_cleaned.parquet`) or pass `auto` to automatically discover and purge matching sibling datasets in the same run.
> * **Companion Embeddings**: Discovers companion embeddings (`*_embeddings.npy` and `*.keys.parquet`) sharing the dataset prefix and prunes matching vectors atomically via zero-GPU memory-mapped slicing (`np.memmap`).
> * **Clustered Sidecars**: Discovers and filters companion clustered sidecars (`*_clustered_k_*.parquet`) in-place.

### Scraper Chunk-Level Validation (`src/scrapers/mapillary_scraper.py`)

For future scraping runs, [`mapillary_scraper.py`](file:///home/aaniraj/Documents/Projects/code/geo-rag/src/scrapers/mapillary_scraper.py) captures `sequence` during initial photo discovery and automatically audits unique sequences at the end of each chunk. Corrupted stationary tracks are purged before saving, guaranteeing that downstream metadata parsing and deduplication receive 100% clean data with the standard 6-column schema.
