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
