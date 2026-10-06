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
