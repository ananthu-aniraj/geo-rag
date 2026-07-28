# Geo-RAG Evaluation & Benchmarking Methodology

This document outlines the evaluation and benchmarking suites in the Geo-RAG pipeline. These scripts are designed to measure:
1. **MLLM/VLM Captioning Quality** (Zero-Shot scene classification & text-image alignment).
2. **Geographical Land Use/Cover Classification (LUCAS 2018)**.
3. **Cross-Modal Retrieval Performance** (Image-to-Text and Text-to-Image alignment).

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

---

## 📈 Combined Evaluation Flow

To run an end-to-end evaluation cycle:
1. Run `caption_test.py` to caption a set of validation images and generate embeddings.
2. Run `evaluate_retrieval.py` on the resulting pickle output to benchmark the retrieval metrics.
3. Inspect `vlm_evaluation_summary_*.txt` for overall VLM prediction accuracies.
