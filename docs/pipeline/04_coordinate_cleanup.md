# Step 1c: Coordinate Anomaly Cleanup

This document describes the design and operation of `cleanup_coordinate_anomalies.py`, which filters out coordinate-locking anomalies caused by faulty contributor GPS units.

---

## ⚙️ Core Operation

During scraping, certain contributors may have faulty GPS tracking equipment that locks onto a single latitude coordinate (parallel) while longitude continues to update. This creates straight lines of incorrect coordinates stretching across multiple regions.

The script scans coordinate distributions, flags locked parallels, and deletes the anomalies to write an independent cleaned metadata file (`geo_space_cleaned.parquet`), leaving the raw deduplicated database untouched.

To prevent false positives, **anomaly checking is grouped by platform and continent**:

1. It reads `Latitude`, `Longitude`, `Platform`, and `continent` (if present) from the Parquet dataset.
2. It aggregates statistics within individual platforms (and optionally continents) rather than globally.
3. This ensures that a normal dense urban center captured by one platform is not flagged as an anomaly because of a GPS glitch present on a different platform.

---

## 📐 Safety Criteria

A rounded latitude parallel $L$ (rounded to 5 decimal places, representing ~1.1 meters precision) is flagged and purged within a specific platform/continent grouping only if:

$$
\text{Count}_{\text{Platform}}(L) \gt 10 \quad \text{and} \quad \text{Longitude Span}_{\text{Platform}}(L) \gt 1.0^{\circ}
$$

* A longitude span threshold of $> 1.0^{\circ}$ (~111 km) ensures that dense city streets or landmarks—which naturally accumulate thousands of images inside extremely small bounding boxes—are completely preserved.
* Platform-specific locked parallel lines spanning across multiple countries or provinces are safely purged.

---

## 🗺️ Automated Mapillary Sequence Expansion

Street-level imagery in Mapillary is captured in continuous temporal driving sessions termed **sequences**. When a contributor experiences a hardware or GPS lockup (e.g. frozen latitude sensor across hundreds of kilometers), the entire driving session is contaminated.

While points directly on the parallel are flagged by the `round(5)` check, other frames in that faulty sequence may drift or jitter slightly (e.g. `±0.00001` or GPS noise) and escape point-level detection.

To eliminate GPS pollution comprehensively:

1. When Mapillary locked-latitude anomalies are detected within the filtered platform/continent scope, the script automatically loads `MAPILLARY_TOKEN` (from the environment or `.env`).
2. It batch-queries Mapillary Graph API v4 (`https://graph.mapillary.com/?ids=...&fields=id,sequence`) in chunks of 100 to map all flagged points to their parent `sequence_id`s.
3. For every violating sequence, it queries `https://graph.mapillary.com/image_ids?sequence_id={seq_id}` to retrieve all image IDs belonging to that contaminated drive.
4. During streaming Parquet and CSV writes, **the entire sequence** (all member image IDs) is purged from the database alongside the locked parallel points.
5. If `MAPILLARY_TOKEN` is not present, the script gracefully falls back to point-level parallel purging with an informative notice.

> [!TIP] Synchronizing Existing Offline Datasets
> When coordinate cleaning purges new anomalies or sequences from `geo_space_cleaned.parquet`, any existing offline dataset (`geo_space_cleaned_offline.parquet`), companion embedding matrices (`.npy` and `.keys.parquet`), and clustered sidecars (`*_clustered_k_*.parquet`) can be brought into 100% alignment using `src/utils/sync_offline_dataset.py` (or `./scripts/pipeline/run_offline_pipeline.sh --sync_offline`).

---

## ⚙️ Options & Parameters

You can restrict the scope of coordinate cleaning to a specific platform or region to target known glitches (e.g. Mapillary uploads in Africa) by editing the following keys in `params.yaml`:

```yaml
pipeline:
  cleanup_anomalies: true  # Enable or disable coordinate parallel cleaning
  cleanup_platform: "mapillary" # (Optional) Target platform for coordinate anomaly cleanup (e.g. "mapillary")
  cleanup_continent: "africa"  # (Optional) Target continent for coordinate anomaly cleanup (e.g. "africa")
```

### Command Line Arguments

Alternatively, you can run the script manually with targeted flags:

* `--platform`: Restricts detection and cleaning to a specific platform (e.g. `--platform mapillary`).
* `--continent`: Restricts detection and cleaning to a specific continent (e.g. `--continent africa`).

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

# 2. Execute purge: stream-remove all invalid sequence photos from Parquet and companion CSV
python3 -m src.processing.audit_mapillary_sequences \
    --input full_pipeline_output/geo_space_cleaned.parquet \
    --output full_pipeline_output/geo_space_cleaned.parquet \
    --continent Africa \
    --csv full_pipeline_output/geo_space_cleaned.csv
```

### Scraper Chunk-Level Validation (`src/scrapers/mapillary_scraper.py`)

For future scraping runs, [`mapillary_scraper.py`](file:///home/aaniraj/Documents/Projects/code/geo-rag/src/scrapers/mapillary_scraper.py) captures `sequence` during initial photo discovery and automatically audits unique sequences at the end of each chunk. Corrupted stationary tracks are purged before saving, guaranteeing that downstream metadata parsing and deduplication receive 100% clean data with the standard 6-column schema.
