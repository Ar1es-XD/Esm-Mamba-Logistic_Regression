# Experiment 4: Both Novel

## Partition Sampling Logic
* **Type**: Independent double-holdout split (both Antibody and Virus are novel).
* **Held-out Set**:
  * **232** unique antibodies (listed in `held_out_antibodies.csv`).
  * **749** unique viruses (listed in `held_out_viruses.csv`).
* **Volume**:
  * **Train**: 34,774 pairs (interactions where *neither* the antibody nor the virus is held out).
  * **Test**: 7,306 pairs (interactions where *both* the antibody and the virus are held out).
  * **Excluded**: **32,650** pairs (interactions where only *one* of the assets is held out; kept in `excluded_pairs.csv` for audit).

## Distinctiveness from Other Experiments
This represents the most challenging generalization boundary: **extrapolation**. Neither the antibody nor the virus in any test pair has been seen anywhere in the training set. Partial overlap pairs (where only one element is novel) are completely excluded to prevent feature leakage. It evaluates how well the system models the biophysical interaction rules rather than memorizing individual asset characteristics.

## How to Run Standalone
Assuming the shared cache `../shared/v_fused_cache/` has been generated, run:
```bash
python3 train_lr.py
```
This will:
1. Load `train.csv` and `test.csv`.
2. Load corresponding `v_fused` vectors from the shared cache.
3. Fit an L2-regularized `LogisticRegressionCV` (StandardScaler fitted on train only).
4. Save the results (n, %, AUROC, AUPRC) in `results.json`.
