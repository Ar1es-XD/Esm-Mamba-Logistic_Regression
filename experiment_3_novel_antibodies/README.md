# Experiment 3: Novel Antibodies

## Partition Sampling Logic
* **Type**: Antibody-level block holdout.
* **Holdout Set**: **137** unique antibodies are held out completely from the training set.
* **Volume**: 57,903 training pairs and 16,827 testing pairs.

## Distinctiveness from Other Experiments
This experiment tests the model's ability to generalize to **completely unseen antibodies**. The training set contains no records involving any of the 137 held-out antibodies (listed in `held_out_antibodies.csv`). The test set consists entirely of interactions involving these novel antibodies. This replicates the scenario where a newly engineered antibody is tested against a panel of known virus strains.

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
