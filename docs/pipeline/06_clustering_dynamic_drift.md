# Step 2: Global FAISS GPU Clustering & Semantic Drift Detection

This document describes the design and operation of `cluster_images_global.py` and `check_semantic_drift.py`, which handle dynamic clustering mode decisions, FAISS-based GPU Spherical K-Means clustering, and hierarchical grouping.

---

## ⚙️ Automated Mode Selection (Fit vs Assign)

To optimize computation time and avoid unnecessary MLLM labeling costs, the pipeline dynamically determines whether to perform full re-clustering (`fit` mode) or map new images to existing centroids (`assign` mode) using a scale-proof semantic drift detector:

### 1. Dynamic Drift Analysis (`check_semantic_drift.py`)
If a clustered Parquet database file already exists on disk for the current $k$, the pipeline extracts embeddings from the new images. It queries these embeddings against the old centroids using FAISS and measures their cosine similarities.

### 2. `fit` Mode (Re-cluster and Label)
Triggered under any of the following conditions:
* No pre-existing clustered database exists for the space.
* The target cluster count $k$ configured in `params.yaml` has changed.
* **Significant semantic drift is detected**: More than 3% of the new images are classified as outliers, meaning they have a cosine similarity of $< 0.70$ with all existing centroids.

In `fit` mode, it runs full FAISS Spherical K-Means and triggers downstream VLM auto-labeling.

### 3. `assign` Mode (Map to Centroids)
Triggered if the pre-existing clustered database exists and the semantic distribution of new images is stable (outliers $\le 3\%$).
* **Zero VLM Cost**: Rather than re-clustering all data and re-running expensive MLLMs, it dynamically calculates the centroid coordinates from the existing Parquet database in memory, maps the new images to their nearest centroids using FAISS nearest-neighbor search, and maps existing VLM labels/descriptions to the new images. This executes in seconds.

---

## 📐 Clustering Methodology (GPU Spherical K-Means)

Clustering is performed inside `cluster_images_global.py` using two levels of grouping:

### 1. Fine-Grained Child Clustering
Runs Spherical K-Means on GPU:
```python
faiss.Kmeans(d, k, niter=20, spherical=True, gpu=True)
```
Using raw normalized image embeddings, this partitions the dataset into $k$ fine-grained child clusters.

### 2. Hierarchical Parent Clustering
Runs Spherical K-Means on GPU on the normalized child centroids to group them into $k_{\text{parents}}$ broader parent clusters (where $k_{\text{parents}} = \max(2, k / 80)$).

### 3. Type Resilience & Immediate Persistence
* Coordinate columns are cast to numeric float64 right after loading to preserve schema alignment during final Parquet export.
* Cluster assignments are written to the database (`geo_space_clustered_k_{num_clusters}.parquet`) and heavy embedding matrices are immediately released from RAM to avoid CPU memory bottlenecks.
