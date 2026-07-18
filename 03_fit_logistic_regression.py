# -*- coding: utf-8 -*-
"""
03_fit_logistic_regression.py

Fits L2-regularised logistic regression on the cached v_fused vectors
produced by 02_train_and_extract_vfused.py.

For each of the 4 splits:
  1. Load cached v_fused (train + test).
  2. StandardScaler fit on train, transform both.
  3. LogisticRegressionCV with L2 penalty to auto-select C.
  4. Report AUC, AUPR, F1, ACC alongside MLP results.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from Toolkit import Metrics, set_seed_all

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
CACHE_DIR = 'cache'
RESULTS_DIR = 'Results'
SPLITS = ['random', 'ab_block', 'vir_block', 'both_block']

# L2 regularisation search grid (C = 1/lambda)
C_VALUES = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


def main():
    set_seed_all(SEED)

    # Load MLP results if available
    mlp_path = os.path.join(RESULTS_DIR, 'mlp_results.csv')
    if os.path.exists(mlp_path):
        mlp_df = pd.read_csv(mlp_path, index_col=0)
    else:
        mlp_df = None
        print('Warning: mlp_results.csv not found. MLP comparison will be N/A.')

    rows = []

    for split in SPLITS:
        print(f'\n{"="*60}')
        print(f'SPLIT: {split}')
        print(f'{"="*60}')

        # Load cached v_fused
        vf_train = np.load(os.path.join(CACHE_DIR, f'{split}_vfused_train.npy'))
        y_train  = np.load(os.path.join(CACHE_DIR, f'{split}_labels_train.npy'))
        vf_test  = np.load(os.path.join(CACHE_DIR, f'{split}_vfused_test.npy'))
        y_test   = np.load(os.path.join(CACHE_DIR, f'{split}_labels_test.npy'))

        print(f'  Train: n={len(y_train)}, %neut={y_train.mean():.3f}')
        print(f'  Test:  n={len(y_test)},  %neut={y_test.mean():.3f}')
        print(f'  v_fused dim: {vf_train.shape[1]}')

        # StandardScaler — fit on train only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(vf_train)
        X_test  = scaler.transform(vf_test)

        # L2-regularised LR with cross-validated C selection
        lr = LogisticRegressionCV(
            Cs=C_VALUES,
            cv=5,
            penalty='l2',
            scoring='roc_auc',
            solver='lbfgs',
            max_iter=2000,
            random_state=SEED,
            n_jobs=-1
        )
        lr.fit(X_train, y_train.astype(int))

        best_C = lr.C_[0]
        print(f'  Best C (1/lambda): {best_C}')

        # Predict probabilities
        y_prob = lr.predict_proba(X_test)[:, 1]

        # Metrics
        auc, aupr, f1, acc = Metrics(y_test, y_prob)
        print(f'  LR  AUC={auc:.4f}  AUPR={aupr:.4f}  F1={f1:.4f}  ACC={acc:.4f}')

        # Get MLP metrics for comparison
        mlp_auc = mlp_df.loc[split, 'auc'] if mlp_df is not None and split in mlp_df.index else float('nan')
        mlp_aupr = mlp_df.loc[split, 'aupr'] if mlp_df is not None and split in mlp_df.index else float('nan')

        rows.append({
            'split': split,
            'train_n': len(y_train),
            'test_n': len(y_test),
            'test_%neut': f'{y_test.mean():.3f}',
            'LR_C': best_C,
            'LR_AUC': round(auc, 4),
            'LR_AUPR': round(aupr, 4),
            'LR_F1': round(f1, 4),
            'LR_ACC': round(acc, 4),
            'MLP_AUC': round(float(mlp_auc), 4) if not np.isnan(float(mlp_auc)) else 'N/A',
            'MLP_AUPR': round(float(mlp_aupr), 4) if not np.isnan(float(mlp_aupr)) else 'N/A',
            'AUC_delta': round(auc - float(mlp_auc), 4) if not np.isnan(float(mlp_auc)) else 'N/A',
        })

    # ── Final comparison table ──────────────────────────────────────
    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'lr_vs_mlp_comparison.csv'), index=False)

    print(f'\n{"="*80}')
    print('HEAD-TO-HEAD COMPARISON: MLP vs Logistic Regression (same v_fused)')
    print(f'{"="*80}')
    print(results_df.to_string(index=False))
    print(f'\nSaved to {RESULTS_DIR}/lr_vs_mlp_comparison.csv')


if __name__ == '__main__':
    main()
