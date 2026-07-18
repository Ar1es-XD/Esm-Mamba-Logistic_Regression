# Shared Pipeline Assets & Context

This folder contains shared data assets and documentation that are identical and leveraged across all 4 experiments.

## 1. Cleaned Base Dataset (`cleaned_dataset.csv`)
* **Preprocessing details**: Constructed from the CATNAP HIV master neutralization database by:
  1. Extracting unique antibodies, viruses, and labels.
  2. Resolving Heavy and Light chain sequences for each antibody (H+L total sequence lengths).
  3. Relational mapping of 74,730 total interactions.
  4. Binarizing the neutralization metric into `neut` (1 = neutralizing, 0 = non-neutralizing).
* **Composition**: 43,799 neutralizing (58.6%) and 30,931 non-neutralizing (41.4%) samples.

## 2. Feature Dimensions & Padding
* **Max Length Padding (Phase A)**: 
  * Antibodies zero-padded (or truncated) to **676** residues (100th percentile / max observed length).
  * Antigens zero-padded (or truncated) to **912** residues (100th percentile / max observed length).
* **Feature Extraction**: ESM-2 model `esm2_t6_8M_UR50D` embeds residues into a 320-dimensional hidden state.
* **`v_fused` Dimensionality (Phase D)**: **1,588** dimensions.
  * Formed by average pooling the Mamba horizontal sweep output (`676`) and the Mamba vertical sweep output (`912`) and concatenating them (`676 + 912 = 1588`).

## 3. Checkpoint & Initialization
* **ESM-2 Checkpoint**: `esm2_t6_8M_UR50D.pt` (~33MB).
* **Mamba Encoder Checkpoint**: None (the bilinear projection $W$ matrix and VMamba layers are initialized from scratch at runtime using seed 42, acting as a fixed sequence-order projector for feature extraction).

## 4. Normalization Status
No normalization or scale transformations are applied to `v_fused` inside `extract_vfused_cache.py`. It is stored as raw activation vectors. Individual experiments apply a `StandardScaler` (fitted on their training partitions only) prior to fitting Logistic Regression.
