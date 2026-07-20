# ESM-Mamba + Logistic Regression (`esm-up`): Modular Baseline Pipeline

This repository contains the modular, self-contained experiment pipeline for evaluating ESM-Mamba biophysical feature representations ($v_{\text{fused}}$) under four distinct biological generalization boundaries using L2-regularized Logistic Regression.

---

## 📂 Repository Layout

```
esm-up/
├── Data/HIV/                 # Raw sequence tables (antibody.csv, antigen.csv)
├── docs/                     # Documentation and scientific methodology notes
│   ├── methodology_explanation.txt
│   ├── pipeline_documentation.txt
│   ├── scientific_review.txt
│   └── legacy_results/
│
├── shared/                   # Core reusable modules, models, and shared cache
│   ├── Models.py             #   MambaCross model (Bilinear projection & 2D VMamba sweeps)
│   ├── Pretrained.py         #   ESM-2 sequence embedding extractor
│   ├── Toolkit.py            #   Evaluation metrics & helper functions
│   ├── Loader.py             #   Pair dataset loader
│   ├── Param_Model.json      #   Model hyperparameters
│   ├── cleaned_dataset.csv   #   Base dataset (all 74,730 interaction pairs)
│   └── v_fused_cache/        #   Cached 1,588-dim feature representations
│
├── experiment_1_random/      # Exp 1: Random Split (59,799 train / 14,931 test)
│   ├── train.csv & test.csv
│   ├── train_lr.py           # Standalone Logistic Regression trainer
│   └── README.md
│
├── experiment_2_novel_viruses/ # Exp 2: Novel Viruses (61,219 train / 13,511 test)
│   ├── train.csv & test.csv
│   ├── held_out_viruses.csv  # 541 held-out viral strain IDs
│   ├── train_lr.py           # Standalone Logistic Regression trainer
│   └── README.md
│
├── experiment_3_novel_antibodies/ # Exp 3: Novel Antibodies (57,903 train / 16,827 test)
│   ├── train.csv & test.csv
│   ├── held_out_antibodies.csv # 137 held-out antibody IDs
│   ├── train_lr.py           # Standalone Logistic Regression trainer
│   └── README.md
│
├── experiment_4_both_novel/  # Exp 4: Both Novel (34,774 train / 7,306 test)
│   ├── train.csv & test.csv
│   ├── excluded_pairs.csv    # 32,650 single-novel overlap pairs
│   ├── held_out_antibodies.csv & held_out_viruses.csv
│   ├── train_lr.py           # Standalone Logistic Regression trainer
│   └── README.md
│
├── extract_vfused_cache.py   # Computes & caches Phase B-D v_fused vectors
├── run_all_experiments.py    # Master runner (fits LR across all 4 experiments)
├── run_pipeline.sh           # End-to-end orchestration shell script
├── summary_results.csv       # Consolidated performance table (generated)
└── requirements.txt          # Python dependencies
```

---

## ⚡ Execution Instructions

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate     # Linux/macOS
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. End-to-End Orchestration Run
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### 3. Step-by-Step Manual Execution

#### Step 3.1: Extract ESM-2 Sequence Embeddings (Phase A)
```bash
python3 shared/Pretrained.py
```
Runs `esm2_t6_8M_UR50D` to generate `.npy` embeddings for all antibodies and antigens under `Outputs/Pretrained_HIV/`.

#### Step 3.2: Cache $v_{\text{fused}}$ Representation Vectors (Phases B-D)
```bash
python3 extract_vfused_cache.py
```
Runs bilinear projection and VMamba sequence sweeps over ESM-2 embeddings, saving 1,588-dimensional $v_{\text{fused}}$ vectors to `shared/v_fused_cache/`.

#### Step 3.3: Fit and Evaluate Logistic Regression
```bash
python3 run_all_experiments.py
```
Sequentially fits L2-regularized `LogisticRegressionCV` across all four partitions and exports consolidated metrics to `summary_results.csv`.

---

## 🔬 Standalone Experiment Execution

Each experiment folder is fully self-contained. Assuming `shared/v_fused_cache/` exists, run:

```bash
cd experiment_3_novel_antibodies
python3 train_lr.py
```
