# ESM-Mamba + Logistic Regression Baseline

**Head-to-head comparison**: Replace only the MLP classifier (Phase E) with L2-regularised logistic regression. Everything upstream — ESM-2 embedding, bilinear contact grid, dual Mamba sweeps, and adaptive pooling — remains **identical**.

## Architecture

```
Phase A │ ESM-2 (esm2_t6_8M_UR50D)  → per-residue embeddings (320-dim)
        │   Antibody: heavy + light chains → (L_ab, 320), padded to (676, 320)
        │   Antigen:  env sequence         → (L_ag, 320), padded to (912, 320)
        ↓
Phase B │ Bilinear contact grid: x_Ab @ W @ x_Ag.T → (676, 912)
        ↓
Phase C │ Dual Mamba sweeps:
        │   mamba_hor(contacts)      → (676, 912) → pool → (676,)
        │   mamba_ver(contacts.T)    → (912, 676) → pool → (912,)
        ↓
Phase D │ Concatenation: v_fused = cat(h, v) → (1588,)
        ↓
Phase E │ ┌─────────────────────────────────────────────────────────┐
 (swap) │ │ ORIGINAL: MLP  1588→512→128→32→1 (BN+SiLU+dropout)    │
        │ │ BASELINE: L2 Logistic Regression (StandardScaler→LR)   │
        │ └─────────────────────────────────────────────────────────┘
```

## Dataset

**CATNAP HIV** (mentor's pre-processed data):
- 74,730 antibody–virus pairs (685 antibodies × 2,705 viruses)
- Binary label: neutralising (58.6%) / non-neutralising (41.4%)
- 4 pre-computed splits preserved from the mentor:

| Split | Train | Test | Excluded |
|-------|-------|------|----------|
| `random` | 59,799 | 14,931 | — |
| `ab_block` | 57,903 | 16,827 | — |
| `vir_block` | 61,219 | 13,511 | — |
| `both_block` | 34,774 | 7,306 | 32,650 |

## Pipeline (3 steps)

### Step 1: Generate ESM-2 embeddings (Phase A)
```bash
python Pretrained.py
```
Produces `Outputs/Pretrained_HIV/{ab,ag}/*.npy` — one file per antibody/antigen.

### Step 2: Train MambaCross + extract v_fused
```bash
python 02_train_and_extract_vfused.py
```
For each split:
1. Trains MambaCross end-to-end (Phases B–E with MLP)
2. Saves best checkpoint to `Results/model_{split}.pt`
3. Extracts v_fused from trained Phases B–D for all train+test pairs
4. Caches v_fused + labels to `cache/{split}_vfused_{train,test}.npy`
5. Records MLP's test AUC for comparison

### Step 3: Fit logistic regression on v_fused
```bash
python 03_fit_logistic_regression.py
```
For each split:
1. Loads cached v_fused
2. `StandardScaler` (fit on train only, transform both)
3. `LogisticRegressionCV` with L2 penalty, 5-fold CV over C ∈ {0.001, ..., 100}
4. Reports AUC, AUPR, F1, ACC alongside MLP results
5. Saves head-to-head comparison to `Results/lr_vs_mlp_comparison.csv`

### One-shot runner
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

## Target Hardware

- **CPU**: AMD Ryzen 9 9900X (12-core)
- **RAM**: 64 GB DDR5
- **GPU**: NVIDIA RTX PRO 4000 Blackwell (24 GB VRAM)
- **Framework**: PyTorch with CUDA

## Key Design Decisions

1. **v_fused extracted from MLP-trained model**: The MambaCross model is trained end-to-end with its original MLP head. After training, v_fused is extracted from the same trained model. This ensures the upstream representations (Phases B–D) are identical to what the MLP classifier saw.

2. **StandardScaler before LR**: The original MLP receives raw v_fused (no upstream normalisation). Since LR is scale-sensitive, we apply StandardScaler fitted on train data only. This is the only addition beyond the classifier swap.

3. **L2 regularisation with CV**: v_fused is 1,588-dimensional — wide enough to overfit. LogisticRegressionCV selects the best C via 5-fold cross-validation on the training set, scored by AUC.

4. **No data leakage**: Each split is handled independently. LR is trained only on that split's train rows, evaluated only on test rows, and `excl` rows in `both_block` are dropped entirely.

## File Structure

```
esm-up/
├── Data/HIV/
│   ├── ab_ag_pair.csv          # 74,730 pairs with 4 split columns
│   ├── antibody.csv            # 685 antibodies (ab_name, heavy, light)
│   └── antigen.csv             # 2,705 viruses (ag_name, ag_seq)
├── Models.py                   # MambaCross architecture (unchanged)
├── Toolkit.py                  # Metrics, Dataset, utils (unchanged)
├── Loader.py                   # Data loading helpers (unchanged)
├── Pretrained.py               # ESM-2 embedding extraction (unchanged)
├── Param_Model.json            # Hyperparameters (unchanged)
├── 02_train_and_extract_vfused.py  # NEW: train + extract v_fused
├── 03_fit_logistic_regression.py   # NEW: fit LR + compare
├── run_pipeline.sh             # NEW: master orchestrator
├── requirements.txt            # Python dependencies
└── .gitignore
```
