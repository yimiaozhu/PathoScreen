# PathoScreen: MASLD-Specific Pathway Perturbation Prediction

**PathoScreen** is a deep learning framework designed to predict the **perturbation capability of small molecules on MASLD-specific pathways**. 

---

# 🔬 Scientific Context

The model predicts whether a molecule perturbs one of **7 MASLD-related pathways** involved in
**Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD)**.

| ID | Pathway Name            |
| -- | ----------------------- |
| P0 | Fatty Acid Metabolism   |
| P1 | Liver Insulin Signaling |
| P2 | Antioxidant Defense     |
| P3 | Anti-apoptosis          |
| P4 | Mitochondrial Function  |
| P5 | Anti-inflammation       |
| P6 | TGF-beta Signaling      |

---

# 🚀 Workflow & Quick Start

## Installation

```bash
git clone --recursive https://github.com/PathoScreen-AI/PathoScreen.git

conda env create -f environment.yaml
conda activate pathoscreen
```

---

## Step 0: Data Preparation

Before running **Path A (training)** or **Path B (screening)**, embeddings for **cells** and **molecules** must be generated.

---
### 1. Prepare Resources

Please Download the following files:
* `scFoundation.ckpt` — pretrained scFoundation model
* `input_cell_matrix.pkl` — **precomputed cell embeddings** 

1. Download the files from **[Zontero Link Placeholder]**
   
2. Place them into the directory:

```
resources/
```

Example structure:

```
PathoScreen
│
├── resources
│   ├── scFoundation.ckpt
│   └── input_cell_matrix.pkl
```

Note:

`input_cell_matrix.pkl` already contains **embedded cell representations** and can be used directly for training and inference.

---

### 2. Generate Cell Embeddings (scFoundation)

If you want to generate cell embeddings from your own **gene expression matrix**, you can run the scFoundation embedding pipeline.

First initialize the submodule:

```bash
git submodule update --init --recursive
```

Then run the embedding script:

```bash
python scripts/generate_embeddings.py \
  --input_csv path/to/gene_expression.csv \
  --scfoundation_ckpt /path/to/scFoundation.ckpt \
  --output_path data/cell_emb/input_cell_matrix.pkl
```

---

### 3. Build SMILES Graph Cache

Pre-compute molecular graphs to accelerate training and inference.

Two usage modes:

* **Path A (Training)** → scan all pathway datasets
* **Path B (Screening)** → process your candidate list

Example using a candidate file:

```bash
python scripts/build_smiles_cache.py \
  --input_csv data/candidates.csv \
  --output_path data/cache/smiles_graph.pkl
```

Alternatively, scan a directory structure:

```bash
python scripts/build_smiles_cache.py \
  --pathways_root data/pathways \
  --output_path data/cache/smiles_graph.pkl
```

---

## Path A: Train Custom Models

**Target users:**
Researchers with their own **compound–cell perturbation data** who want to train models from scratch.

---

### 1. Train PathoScreen

Train the **PathoScreen model** using **5-fold cross-validation** followed by retraining.

```bash
python -m src.cli train \
  --pathway_id 0 \
  --config configs/default_train.yaml \
  --smiles_cache data/cache/smiles_graph.pkl
```

Each MASLD pathway is trained **independently** (`pathway_id = 0–6`).

---

### 2. Train Calibrator

Fit **Isotonic Regression calibrators** on the held-out test set.

```bash
python -m src.cli calibrate \
  --pathway_ids 0 \
  --test_data_root data/pathways \
  --emb_path data/cell_matrix/input_cell_matrix.pkl
```

Calibration improves probability reliability and enables **Brier score estimation**.

---

## Path B: Virtual Screening

**Target users:**
Researchers with **lists of candidate compounds** who want to screen them against specific cell states using **pre-trained models**.

---

### 1. Predict & Score

Run inference using the **pre-trained models and calibrators**.

```bash
python -m src.cli predict \
  --pathway_id 0 \
  --input_csv data/candidates.csv \
  --emb_pkl data/cell_matrix/input_cell_matrix.pkl \
  --smiles_cache data/cache/smiles_graph.pkl \
  --use_pretrained \
  --output_csv output/P0_results.csv
```

The output file contains **pathway-specific perturbation probabilities and predicted labels**.

---

## Final Step: Rank Candidates

Aggregate predictions across all **7 MASLD pathways (P0–P6)** to prioritize candidate compounds.

```bash
python scripts.rank_candidates.py \
  --pred_dir output \
  --brier_json calibration_results/brier_scores.json \
  --output_file output/final_rank.csv
```

#### Ranking Strategy

1. **Vote Count**
   Number of pathways predicted as positive.

2. **PCS (Pathway Consensus Score)**
   Brier-weighted consensus score used as a secondary ranking criterion.

Final ranking is sorted by:

```
Vote_Count → PCS (descending)
```

---

# Pathway Mapping

Seven MASLD-related pathways are indexed as **P0–P6** and defined in:

```
src/config.py
```
