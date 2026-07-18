# Experiment 2: Novel Viruses

## Partition Sampling Logic
* **Type**: Virus-level block holdout.
* **Holdout Set**: **541** unique viruses are held out completely from the training set.
* **Volume**: 61,219 training pairs and 13,511 testing pairs.

## Distinctiveness from Other Experiments
This experiment tests the model's ability to generalize to **completely unseen virus strains**. The training set contains no records involving any of the 541 held-out viruses (listed in `held_out_viruses.csv`). The test set consists entirely of interactions with these novel strains. This replicates the scenario where an existing antibody library is tested against newly emerging virus variants.

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
