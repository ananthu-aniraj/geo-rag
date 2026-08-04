# Step 1c: Coordinate Anomaly Cleanup

This document describes the design and operation of `cleanup_coordinate_anomalies.py`, which filters out coordinate-locking anomalies caused by faulty contributor GPS units.

---

## ⚙️ Core Operation

During scraping, certain contributors may have faulty GPS tracking equipment that locks onto a single latitude coordinate (parallel) while longitude continues to update. This creates straight lines of incorrect coordinates stretching across multiple regions.

The script scans coordinate coordinate distributions, flags locked parallels, and deletes the anomalies to write an independent cleaned metadata file (`geo_space_cleaned.parquet`), leaving the raw deduplicated database untouched.

---

## 📐 Safety Criteria

A rounded latitude parallel $L$ (rounded to 5 decimal places, representing ~1.1 meters precision) is flagged and purged only if:

$$
\text{Count}(L) \gt 10 \quad \text{and} \quad \text{Longitude Span}(L) \gt 1.0^{\circ}
$$

* A longitude span threshold of $> 1.0^{\circ}$ (~111 km) ensures that dense city streets or landmarks—which naturally accumulate thousands of images inside extremely small bounding boxes—are completely preserved.
* Global coordinate-locked anomaly lines spanning multiple cities or countries are safely purged.

---

## 🏎️ Performance & Type Resilience
Both Parquet loading and CSV chunk writing perform defensive `pd.to_numeric` conversions on coordinates at startup to prevent float-string mixed schema exceptions during high-precision rounding operations.
