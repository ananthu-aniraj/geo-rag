# Evaluation & Benchmarking Methodology

This document outlines the evaluation and benchmarking suites in the Geo-RAG pipeline. These scripts are designed to measure:
1. **MLLM/VLM Captioning Quality & Scene Classification (`caption_test.py`)** - Zero-shot scene classification and text-image alignment on Places365.
2. **LUCAS 2018 Classification Accuracy (`evaluate_lucas.py`)** - Zero-shot VLM land cover, land use, and habitat prediction accuracy.
3. **Cross-Modal Retrieval Performance (`evaluate_retrieval.py`)** - Text-to-image and image-to-text retrieval using VLM captions.
4. **LUCAS Semantic Retrieval Benchmarking (`benchmark_lucas.py`)** - Retrieval precision, MRR, and mAP for land cover/use representation alignments.
5. **Places365 Hierarchy Retrieval Benchmarking (`benchmark_places.py`)** - Retrieval precision across exact place, sub-category, and macro-category levels.
6. **EUNIS Ecosystem Map Retrieval Benchmarking (`benchmark_eunis.py`)** - Image representation retrieval alignment with European terrestrial ecosystem classifications.
7. **Environmental Zones of Europe Retrieval Benchmarking (`benchmark_environmental_zones.py`)** - Image representation retrieval alignment with macro-scale European biogeographical climate zones.

> [!NOTE]
> For a detailed conceptual and mathematical explanation of all the evaluation, classification, and retrieval metrics used across these suites, please refer to the **[Evaluation Metrics Guide](metrics.md)**.

---

## 📊 1. VLM Captioning & Places365 Classification (`caption_test.py`)

This script evaluates how accurately a Vision-Language Model (VLM) can describe a scene and classify it relative to the Places365 database categories.

### 🔍 Methodology
1. **Places365 Label Mapping:** Loads Places365 macro-categories (indoor vs. outdoor natural/man-made), sub-categories, and specific class labels from an Excel mapping file.
2. **MLLM Querying:** Queries a local VLM (via Ollama or custom server) in a two-step prompt:
   - **Step 1:** Extracts visual details (visible evidence, human activities, vegetation, land cover).
   - **Step 2:** Predicts specific place classes, macro categories, and sub-categories using a structured JSON schema.
3. **Evaluation Metrics:**
   - **Places365 Class Similarity:** Measures the cosine similarity between the sentence-embedding representation of the VLM prediction and the ground truth Places365 class name.
   - **Macro Accuracy:** Binary accuracy on whether the model correctly identified if the image is Indoor, Outdoor Natural, or Outdoor Man-made.
   - **Sub-Category Similarity:** Cosine similarity of predicted sub-categories against ground-truth classes.
   - **Visual Alignment (CLIP/TIPS):** Computes the cosine similarity of the generated caption text against the raw image embedding.

### 💻 Usage
```bash
python3 -m src.evaluation.caption_test \
  --labels_path path/to/places365_mapping.xlsx \
  --images_dir path/to/test_images/ \
  --model llava:7b \
  --prompt_version v1
```

### 📦 Outputs
- `vlm_evaluation_summary_[model]_[prompt_version].txt`: Overall accuracy and similarity summaries.
- `vlm_evaluation_results_[model]_[prompt_version].csv`: Detailed per-image predictions (excluding large embedding fields).
- `vlm_retrieval_data_[model]_[prompt_version].pkl`: Pickled data containing image filename, VLM caption, and raw image embeddings (used for retrieval tests).

---

## 🌾 2. LUCAS 2018 Classification Benchmark (`evaluate_lucas.py`)

This script benchmarks the VLM's ability to classify land cover, land use, and habitat types using ground-truth points from the **LUCAS 2018 (Land Use and Coverage Area frame Survey)** validation dataset.

### 🔍 Methodology
1. **LUCAS Metadata Ingestion:** Loads LUCAS coordinate metadata, mapping point IDs to ground-truth labels (`lc_label`, `lu_label`, `eunis_class`).
2. **VLM Classification:** Queries the MLLM to classify:
   - **Land Cover (LC):** e.g., Cropland, Woodland, Grassland, Shrubland.
   - **Land Use (LU):** e.g., Agriculture, Forestry, Residential.
   - **EUNIS Habitat Class:** Standard European habitat taxonomy.
3. **Evaluation Metrics:**
   - **Exact Match (EM) Accuracy:** Strict string equality matches.
   - **Semantic Similarity Accuracy:** Computes cosine similarity of the VLM prediction's sentence embeddings against ground-truth class name embeddings to account for synonyms (e.g. "Woodland" matching "Forest").

### 💻 Usage
```bash
python3 -m src.evaluation.evaluate_lucas \
  --csv_path path/to/lucas_metadata.csv \
  --images_dir path/to/lucas_images/ \
  --model llava \
  --output_dir results/
```

### 📦 Outputs
- `lucas_evaluation_summary_[model].txt`: Summarized exact matches and semantic similarity averages for LC, LU, and EUNIS.
- `lucas_evaluation_results_[model].csv`: Row-by-row comparisons of VLM predictions vs. LUCAS ground-truth coordinates.

---

## 🔄 3. Cross-Modal Image-Caption Retrieval (`evaluate_retrieval.py`)

This script performs cross-modal (Image-to-Text and Text-to-Image) retrieval benchmarks to evaluate how well our embedding models (standard **CLIP** vs. **TIPSv2**) align textual descriptions with visual features.

### 🔍 Methodology
1. **Input Payload Ingestion:** Loads the pickled `.pkl` output containing image embeddings and VLM captions generated during the captioning test.
2. **Text Embedding Generation:** Encodes the VLM captions and their individual components (`visible_evidence`, `human_activities`, `land_cover_usage`, `type_of_vegetation`) into a joint embedding space using CLIP or TIPSv2.
3. **Retrieval Evaluations:**
   - Computes a global similarity matrix between all image embeddings and text embeddings.
   - Evaluates **Recall@1 (Top-1)**, **Recall@5 (Top-5)**, and **Mean Reciprocal Rank (MRR)**.
   - Conducts component-level ablation studies to find *which* textual details (e.g. vegetation type vs. human activities) are most aligned with the image's visual features.

### 💻 Usage
```bash
python3 -m src.evaluation.evaluate_retrieval \
  --pkl path/to/vlm_retrieval_data.pkl \
  --use_tips \
  --use_prefix
```

> [!TIP]
> Use the `--use_tips` flag to toggle between evaluation of standard CLIP embeddings and TIPSv2 aligned visual representations.

## 📈 4. LUCAS Semantic Retrieval Benchmarking (`benchmark_lucas.py`)
 
This script benchmarks the semantic retrieval capability of different image representations (TIPSv2 CLS, average patch, and SegFormer-masked embeddings) on the **LUCAS 2018** dataset. It measures how well nearest-neighbor retrieval aligns with ground-truth land cover, land use, and habitat classes, as well as projected spatial-ecological zones.
 
### 🔍 Methodology
1. **LUCAS Metadata Ingestion & Sampling:** Loads the validation CSV coordinates, matches local images using point ID grouping, and shuffles them to build a query set and database pool.
2. **Spatial Raster Overlay:** Extracts the GPS coordinates (`lat`, `lon`) for each LUCAS point. If EUNIS and Metzger Environmental Zones GeoTIFF rasters are provided, it projects the coordinates to EPSG:3035 to query and append projected EUNIS ecosystem categories and biogeographical climate zones.
3. **Batch Feature Extraction:** Computes embeddings in batches on the GPU. If loading a custom checkpoint (via `--tips_model_path`), it evaluates both `TIPSv2 1st CLS` (visual/semantic) and `TIPSv2 2nd CLS` (geographic) tokens against average patch and Seg-Masked embeddings. Otherwise, it defaults to Hugging Face `google/tipsv2-b14`.
4. **Retrieval Metric Computation:** Queries the database using cosine similarity to retrieve the Top-10 nearest neighbors. It reports alignment metrics (P@1, P@5, P@10, mAP@10, and MRR@10) across five hierarchical/ecological levels:
   * **Land Cover** (`lc_label`)
   * **Land Use** (`lu_label`)
   * **EUNIS Class** (`eunis_class` from CSV metadata)
   * **EUNIS Ecosystem** (`eunis_raster_class` from raster overlay, optional)
   * **Environmental Zone** (`env_zone_class` from raster overlay, optional)
 
### 💻 Usage
```bash
python3 -m src.evaluation.benchmark_lucas \
  --csv path/to/Sen4Map_Metadata_test.csv \
  --img_dir path/to/lucas_images/ \
  --eunis_raster path/to/eunis_ecosystem.tif \
  --env_zones_raster path/to/environmental_zones.tif \
  --tips_model_path path/to/checkpoint.npz \
  --tips_model_variant B \
  --num_queries 100 \
  --num_database 500 \
  --output_report benchmark_results/lucas_report.txt \
  --output_csv benchmark_results/lucas_results.csv
```
 
### 📦 Outputs
- `benchmark_results/lucas_report.txt`: A plain text report summary comparing retrieval accuracy metrics across all representations and labels.
- `benchmark_results/lucas_results.csv`: A detailed CSV table containing the exact query images, retrieved top-1 matches, and corresponding P@1, P@5, P@10, AP@10, and RR@10 scores.
 
---
 
## ⛰️ 5. Places365 Hierarchy Retrieval Benchmarking (`benchmark_places.py`)
 
This script benchmarks the semantic retrieval performance of different representations on the **Places365** dataset. It maps images to categories using the hierarchy defined in `Scene_hierarchy.xlsx` to measure how well retrieval preserves visual and scene categories.
 
### 🔍 Methodology
1. **Places Ingestion & Hierarchy Mapping:** Reads the test directory structure, matching each image to its folder name (exact place), mapping it to macro-category (indoor vs. outdoor) and sub-category.
2. **GPU Feature Extraction:** Computes embeddings in batches. Supports loading official checkpoints (via `--tips_model_path`) comparing `TIPSv2 1st CLS` and `TIPSv2 2nd CLS` representations. You can optionally include CLIP (`--compare_clip`) and Hugging Face's `google/tipsv2-b14` model (`--compare_hf_tips`) in the comparison.
3. **Retrieval Evaluations:** Cosine similarity retrieves the Top-10 nearest neighbors from the database pool. Accuracy metrics (P@1, P@5, P@10, mAP@10, and MRR@10) are reported across three hierarchical levels: **Exact Place Category**, **Sub-Category**, and **Macro Category**.
 
### 💻 Usage
```bash
python3 -m src.evaluation.benchmark_places \
  --labels path/to/Scene_hierarchy.xlsx \
  --img_dir path/to/places365_images/ \
  --tips_model_path path/to/checkpoint.npz \
  --tips_model_variant B \
  --compare_clip \
  --compare_hf_tips \
  --num_queries 100 \
  --num_database 500 \
  --output_report benchmark_results/places_report.txt \
  --output_csv benchmark_results/places_results.csv
```
 
### 📦 Outputs
Output filenames are formatted dynamically by appending the seed and query count (`_s[seed]_q[queries]`) to prevent overwriting results across experiments:
- `benchmark_results/places_report_s42_q100.txt`: A plain text report summary comparing retrieval accuracy metrics across all representations and labels.
- `benchmark_results/places_results_s42_q100.csv`: A detailed CSV table containing query images, retrieved top-1 matches, and corresponding metrics.
 
---
 
## 🗺️ 6. EUNIS Ecosystem Map Retrieval Benchmarking (`benchmark_eunis.py`)
  
This script performs geobotanical representation benchmarking on arbitrary geolocated images in Europe (e.g. scraped Flickr/Mapillary datasets). It overlays WGS84 coordinates on a local **EUNIS Ecosystem GeoTIFF raster map** to extract the ecosystem type, evaluating whether nearest-neighbor retrieval aligns with European habitats.
 
### 🔍 Methodology
1. **EUNIS Raster Coordinate Lookup:** Reads your scraped CSV metadata, transforms coordinates into EPSG:3035 using `pyproj`, and queries the local EUNIS GeoTIFF.
2. **Dynamic Label Parsing:** If the `.vat.dbf` database file is found next to the raster, the script parses it dynamically using Geopandas to map raster values to terrestrial labels (e.g., *Woodland*, *Cropland*).
3. **GPU Feature Extraction:** Computes embeddings in batches. Supports loading custom checkpoints via `--tips_model_path` (evaluating both CLS tokens), defaulting to Hugging Face `google/tipsv2-b14` if omitted.
4. **Retrieval Evaluations:** Reports P@1, P@5, P@10, mAP@10, and MRR@10 on geobotanical classifications.
 
### 💻 Usage
```bash
python3 -m src.evaluation.benchmark_eunis \
  --csv_path path/to/scraped_data.csv \
  --tips_model_path path/to/checkpoint.npz \
  --tips_model_variant B \
  --num_queries 100 \
  --num_database 500 \
  --output_report benchmark_results/eunis_report.txt \
  --output_csv benchmark_results/eunis_results.csv
```
 
### 📦 Outputs
- `benchmark_results/eunis_report.txt`: A plain text report summary comparing retrieval accuracy metrics across all representations.
- `benchmark_results/eunis_results.csv`: A detailed CSV table containing query images, retrieved top-1 matches, and corresponding P@1, P@5, P@10, AP@10, and RR@10 scores.
 
---
 
## 🌲 7. Environmental Zones of Europe Retrieval Benchmarking (`benchmark_environmental_zones.py`)
  
This script performs macro-scale biogeographical representation benchmarking on geolocated images in Europe. It overlays WGS84 coordinates on the **Environmental Zones of Europe (Metzger 2025)** GeoTIFF raster map to extract the climate/ecological zone class (1–19), evaluating whether representations capture continental-scale geographical and geobotanical stratification under retrieval.
 
### 🔍 Methodology
1. **Environmental Zone Raster Lookup:** Reads scraped image CSV coordinates, projects them to EPSG:3035, and queries the Metzger 2025 GeoTIFF using `rasterio`.
2. **Category Extraction:** Maps the sampled value to one of the 19 Environmental Zones (e.g. *Boreal*, *Continental*, *Arctic*).
3. **GPU Feature Extraction:** Computes embeddings in batches. Supports loading custom checkpoints via `--tips_model_path` (evaluating both CLS tokens), defaulting to Hugging Face `google/tipsv2-b14` if omitted.
4. **Retrieval Evaluations:** Reports Precision@1, Precision@5, Precision@10, mAP@10, and MRR@10 on European Environmental Zones.
 
### 💻 Usage
```bash
python3 -m src.evaluation.benchmark_environmental_zones \
  --csv_path path/to/scraped_data.csv \
  --tips_model_path path/to/checkpoint.npz \
  --tips_model_variant B \
  --num_queries 100 \
  --num_database 500 \
  --output_report benchmark_results/env_zones_report.txt \
  --output_csv benchmark_results/env_zones_results.csv
```

## 🏃 Running Evaluations using Shell Orchestrators

To make evaluations easily reproducible and readable, the pipeline uses YAML files to manage configuration parameters, and provides two unified shell runners:

### 1. Offline Semantic Evaluation (LUCAS & Places365)
* **Configuration**: `eval_params_offline.yaml`
* **Shell Script**: `./run_offline_eval_semantic.sh`
* **Operation**: Reads parameters from the offline YAML file and runs `benchmark_lucas.py` and `benchmark_places.py` sequentially, saving results into `benchmark_results/`.

### 2. Spatial/Environmental Evaluation (Environmental Zones & EUNIS)
* **Configuration**: `eval_params_online.yaml`
* **Shell Script**: `./run_offline_eval_spatial.sh`
* **Operation**: Reads parameters from the online YAML file and runs `benchmark_environmental_zones.py` and `benchmark_eunis.py` sequentially, saving results into `benchmark_results/`.

---

## 📈 Combined Evaluation Flow

To run an end-to-end evaluation cycle:
1. Populate your dataset paths and parameters in `eval_params_offline.yaml` and `eval_params_online.yaml`.
2. Run `./run_offline_eval_semantic.sh` to benchmark semantic representation retrieval precision on local datasets.
3. Run `./run_offline_eval_spatial.sh` to measure geobotanical and climate alignment retrieval precision on geolocated databases.
4. Run `caption_test.py` to caption a set of validation images and generate embeddings.
5. Run `evaluate_retrieval.py` on the resulting pickle output to benchmark the retrieval metrics.
6. Inspect text reports in the `benchmark_results/` directory (e.g. `lucas_report.txt`, `places_report.txt`, `eunis_report.txt`, `environmental_zones_report.txt`) for overall alignment summaries.

---

## 💾 Memory-Efficient Large-Scale Evaluation

When evaluating large databases (e.g. `num_database` set up to 100,000 or more), holding all images simultaneously in RAM as PIL Image objects will cause system Out-Of-Memory (OOM) crashes. Both `benchmark_environmental_zones.py` and `benchmark_eunis.py` employ a stream-processing and memory-capped architecture:
1. **Query Pre-load & Free:** Queries are downloaded and embedded first. Raw query images are then immediately closed and popped from RAM.
2. **Chunked Database Processing:** Database images are downloaded, embedded, and discarded in sequential chunks of **1,000 images**.
3. **GPU VRAM Cleanup:** Intermediate PyTorch tensors are deleted dynamically and local PIL streams are closed immediately at the end of each chunk.
4. **Index Mismatch Protection:** The scripts dynamically filter the database metadata list to match only successfully downloaded entries, preventing indexing mismatches during batched similarity evaluations.
