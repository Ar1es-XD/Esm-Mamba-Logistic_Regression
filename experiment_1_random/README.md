# Experiment 1: Random Split

## Partition Sampling Logic
* **Type**: Row-level random split.
* **Ratio**: 80% train (59,799 pairs) and 20% test (14,931 pairs).
* **Sampling Details**: Pairs are randomly partitioned without regard to whether the antibody or virus is present in both subsets.

## Distinctiveness from Other Experiments
Unlike Experiments 2, 3, and 4 (which assess model generalization boundaries on unseen biological assets), this experiment represents the easiest task: **interpolation**. Both the antibodies and viruses in the test set are extensively seen during training, but in different combinations. It acts as a baseline to measure the classification capacity of the features when generalization is not tested.

## How to Run Standalone
This folder is fully self-contained. Assuming the shared cache `../shared/v_fused_cache/` has been generated, run:
```bash
python3 train_lr.py
```
This will:
1. Load `train.csv` and `test.csv`.
2. Load corresponding `v_fused` vectors from the shared cache.
3. Fit an L2-regularized `LogisticRegressionCV` (StandardScaler fitted on train only).
4. Save the results (n, %, AUROC, AUPRC) in `results.json`.
