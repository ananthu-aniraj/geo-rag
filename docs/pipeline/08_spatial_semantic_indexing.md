# Step 2d: H3 Spatial-Semantic Indexing

This document describes the design and operation of `build_spatial_semantic_index.py`, which aggregates the clustered dataset into a multi-resolution H3 spatial index mapping.

---

## ⚙️ Core Operation

The script aggregates individual image coordinates and cluster assignments into Uber's H3 Hexagonal Hierarchical Spatial Index across resolutions 1 through 11.

It calculates and records:
1. **Dominant Semantic Class**: The primary land use/land cover category representing the highest volume of images inside the cell.
2. **Category Percentage Breakdowns**: The relative distribution of all categories inside the hex boundary (e.g., `Broadleaved forest: 70%, River: 30%`), allowing downstream map widgets to generate detailed hover tooltips.
3. **Temporal Characteristics**: Dominant season and time of day distributions within each hexagon.
4. **Dominant Cluster ID**: The child cluster ID with the highest density in the cell.

---

## 📐 Index Layout Schema

The output Parquet file (`geo_space_h3_semantic_index.parquet`) contains:
* `resolution`: The target H3 resolution level ($1 \le R \le 11$).
* `query_cell`: The H3 index string (e.g. `8826856235fffff`).
* `dominant_category`: The primary categorical classification.
* `category_breakdown`: JSON string containing the relative percentage of all represented categories in that cell.
* `cluster_id`: The ID of the dominant child cluster.
* `image_count`: Total number of active image coordinates grouped into the cell.

This index forms the spatial backbone for high-performance map visualizations, preventing browsers from having to plot raw coordinate markers in real-time.
