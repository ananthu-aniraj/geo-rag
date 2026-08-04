# Step 1d: Determining the Optimal Cluster Count (k)

This document describes the design and methodology of `validate_cluster_count.py`, which determines the optimal number of clusters ($k$) using Spatial Block Hold-Out validation.

---

## 🔬 Spatial Autocorrelation & The Generalization Gap

In spatial datasets, adjacent data points are highly correlated due to **Tobler's First Law of Geography** [Tobler, 1970]: *"Everything is related to everything else, but near things are more related than distant things."* 

If we partition the training and validation sets randomly, nearby images (e.g., sequential streetscapes or photos of the same landmark) will appear in both sets. This causes **spatial data leakage**, artificially deflating the validation loss and hiding overfitting. 

To measure true generalization, we must partition the dataset geographically. We downscale the fine-grained H3 cell $c$ (resolution 11) to its coarse parent block $b$:

$$
b = \text{parent}(c, R_p)
$$

where $R_p$ = 4 (coarse blocks of ~11,000 km²). We randomly split the set of unique parent blocks $\mathcal{B}$ into disjoint training and validation blocks:

$$
\mathcal{B}_{\text{train}} \cap \mathcal{B}_{\text{val}} = \emptyset
$$

$$
\mathcal{B}_{\text{train}} \cup \mathcal{B}_{\text{val}} = \mathcal{B}
$$

---

## 📐 Optimization Objective & Reconstruction Loss

Let $X_{\text{train}}$ be the set of image embeddings belonging to $\mathcal{B}_{\text{train}}$, and $X_{\text{val}}$ be the embeddings belonging to $\mathcal{B}_{\text{val}}$.

For a given number of clusters $k$, K-Means learns a set of centroids $C^* = \{c_1, ..., c_k\}$ by minimizing the Within-Cluster Sum of Squares (WCSS) on the training set:

$$
\mathcal{L}_{\text{train}}(C) = \sum_{x \in X_{\text{train}}} \min_{c \in C} \| x - c \|^2
$$

The **Validation Reconstruction Loss (Mean Squared Error)** is then evaluated by measuring how well the centroids $C^*$ represent the unseen validation blocks:

$$
\text{MSE}_{\text{val}}(k) = \frac{1}{|X_{\text{val}}|} \sum_{y \in X_{\text{val}}} \min_{c \in C^*} \| y - c \|^2
$$

---

## 📈 Optimal k Selection via the Elbow Method

As $k \to N$, the training loss $\text{MSE}_{\text{train}}(k) \to 0$. However, on the validation set, if $k$ is too high, the centroids will overfit to the specific geographic configurations of the training blocks. 

The optimal $k^*$ is determined using the **Elbow Method** [Thorndike, 1953] on $\text{MSE}_{\text{val}}(k)$—the point at which the rate of decrease in validation error slows down significantly, representing the maximum compression with optimal generalization:

$$
k^* = \arg\max_k \left( \frac{\partial^2 \text{MSE}_{\text{val}}}{\partial k^2} \right)
$$

---

## 💻 Execution Example

Run the validation script across a range of $k$ values ($k \in [10,000, 50,000]$):
```bash
python3 -m src.utils.validate_cluster_count \
  --input "full_pipeline_output/geo_space_deduplicated.parquet" \
  --k_min 10000 \
  --k_max 50000 \
  --k_step 10000 \
  --sample_limit 0 \
  --output_plot "cluster_count_validation.png"
```
*(Setting `--sample_limit 0` ensures the script uses the entire dataset for partitioning rather than downsampling).*
