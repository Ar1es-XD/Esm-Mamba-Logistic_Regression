# ESM-Mamba + Logistic Regression: Modular Experiment Pipeline

This repository contains the restructured, fully self-contained experiment pipeline for evaluating ESM-Mamba biophysical features under different generalization boundaries using L2-regularized Logistic Regression.

## Repository Layout

```
esm-up/
├── Data/HIV/                 # Metadata files (relational tables)
├── shared/
│   ├── v_fused_cache/        # Cached v_fused embeddings (empty until extracted)
│   ├── cleaned_dataset.csv   # Base dataset containing all 74,730 interactions
│   └── README.md             # Shared pipeline parameters and details
│
├── experiment_1_random/
│   ├── train.csv             # 59,799 rows
│   ├── test.csv              # 14,931 rows
│   ├── train_lr.py           # Independent Logistic Regression fitter
│   └── README.md             # Random split holdout context
│
├── experiment_2_novel_viruses/
│   ├── train.csv             # 61,219 rows
│   ├── test.csv              # 13,511 rows
│   ├── held_out_viruses.csv  # 541 held-out virus IDs
│   ├── train_lr.py           # Independent Logistic Regression fitter
│   └── README.md             # Virus-level holdout context
│
├── experiment_3_novel_antibodies/
│   ├── train.csv             # 57,903 rows
│   ├── test.csv              # 16,827 rows
│   ├── held_out_antibodies.csv # 137 held-out antibody IDs
│   ├── train_lr.py           # Independent Logistic Regression fitter
│   └── README.md             # Antibody-level holdout context
│
├── experiment_4_both_novel/
│   ├── train.csv             # 34,774 rows
│   ├── test.csv              # 7,306 rows
│   ├── excluded_pairs.csv    # 32,650 overlapping rows kept for audit
│   ├── held_out_antibodies.csv # 232 held-out antibody IDs
│   ├── held_out_viruses.csv  # 749 held-out virus IDs
│   ├── train_lr.py           # Independent Logistic Regression fitter
│   └── README.md             # Double holdout logic context
│
├── extract_vfused_cache.py   # Extracts Phase B-D features once for all pairs
├── run_all_experiments.py    # Master runner (runs all 4 train_lr.py and builds summary)
├── run_pipeline.sh           # End-to-end orchestration shell script
├── requirements.txt          # Python packages
└── .gitignore
```

---

## Setup & Running the Pipeline (Later on Your Rig)

Execute these steps on your machine (equipped with the **RTX PRO 4000 (24 GB VRAM)**).

### 1. Environment Setup
Create a virtual environment and install the required dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. End-to-End Orchestrated Run
You can run the entire pipeline from scratch using:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### 3. Step-by-Step execution details

#### Step 3.1: Extract ESM-2 sequence embeddings (Phase A)
```bash
python3 Pretrained.py
```
This runs the local `esm2_t6_8M_UR50D` encoder to generate `.npy` embeddings for all individual antibodies and viruses under `Outputs/Pretrained_HIV/`.

#### Step 3.2: Cache `v_fused` representation vectors (Phases B-D)
```bash
python3 extract_vfused_cache.py
```
Runs the bilinear projection and VMamba sequence sweeps forward over the ESM-2 representations, caching the resulting 1,588-dimensional `v_fused` vectors to `shared/v_fused_cache/` (keyed by `<antibody_id>___<virus_id>.npy`). This step runs once on GPU and requires no split-specific parameters.

#### Step 3.3: Fit and Evaluate Logistic Regression
To run all 4 experiments sequentially and aggregate results:
```bash
python3 run_all_experiments.py
```
This will:
1. Sequentially run `train_lr.py` in each experiment folder.
2. Read the resulting metrics from `results.json` in each folder.
3. Write a top-level `summary_results.csv` comparing the performance across all partitions.
4. Print the summary table to the console.

---

## Standalone Execution

Each experiment folder is **fully self-contained and independently runnable**. Assuming `shared/v_fused_cache/` has been generated, you can navigate directly to any experiment folder and run it without touching the others:

```bash
cd experiment_3_novel_antibodies
python3 train_lr.py
```
This script will read only `train.csv` and `test.csv` in its directory, load the features from `../shared/v_fused_cache/`, apply `StandardScaler` (fitted on its training partition only), fit L2-regularized `LogisticRegressionCV` to select $C$, and write its metrics to `results.json`.
