# Preprocessing Pipeline

This directory contains the necessary scripts for video feature extraction and robust dataset partitioning.

## Installation

First, create and activate a new Conda environment:

```bash
conda create -n extract_feats_env python=3.10 -y
conda activate extract_feats_env
```

Next, install PyTorch with CUDA support:

```bash
pip3 install torch torchvision --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
```

Finally, install the remaining required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### `extract_features.py`

This script processes `.mp4` videos to extract frame-level embeddings. Before running the script, update the following parameters directly in the code:

* **Hugging Face Token:** Insert your personal token on **line 46**.
* **I/O Directories:** Define your input video folder and the output destination for the embeddings on **lines 188 and 189**.
* **Model Selection:** Specify the model you want to use for extraction on **line 195**.

> **Note on Micro-SAM:** If you intend to use `microsam` as your embedder, you must run it in a separate Conda environment. Follow the [official installation instructions from source](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-source) to set it up.

To execute the extraction pipeline:

```bash
python extract_features.py
```

---

### `custom_stratified_k_folds.py`

This script generates cross-validation partitions that strictly adhere to two critical conditions simultaneously:

1. **Patient Independence (Group-Aware):** Forces all samples from a single patient into the same fold. This completely prevents data leakage, ensuring no patient's data overlaps between training and validation sets.
2. **Stratification:** Maintains a balanced class distribution (e.g., "Healthy" vs. "Sick") across all folds to ensure representative training and validation phases.

> **Data Formatting Template:** We provide a reference template located at `example_xlsx/EXAMPLE_DATA.xlsx`. Ensure your input dataset follows this exact tabular structure (specifically regarding patient identifiers, sample IDs, and class labels) before running the script.

**Workflow Summary:**
* **Ingestion:** Loads the input Excel file structured according to the provided template.
* **Grouping:** Aggregates the data at the patient level.
* **Balancing (Greedy Algorithm):** Sorts patients by their total number of samples. Iteratively assigns each patient to the fold that currently has the highest deficit of their specific classes, preserving the global class distribution.
* **Output:** Generates a new Excel file (suffixed with `_folded.xlsx`), appending a `FOLD` column to the original data.

To execute the custom stratified CV pipeline:

```bash
python custom_stratified_k_folds.py
```