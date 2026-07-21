# -*- coding: utf-8 -*-
"""
train_lr.py

Fits L2-regularised Logistic Regression on the cached v_fused vectors
for this experiment's train/test partition.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared/v_fused_cache'))
C_VALUES = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X, y = [], []
    missing_count = 0
    
    for _, row in df.iterrows():
        ab_id = row['antibody_id'].replace('/', '_')
        ag_id = row['virus_id']
        label = int(row['neut'])
        
        # Load from shared cache
        feat_path = os.path.join(CACHE_DIR, f"{ab_id}___{ag_id}.npy")
        if os.path.exists(feat_path):
            X.append(np.load(feat_path))
            y.append(label)
        else:
            missing_count += 1
            
    if missing_count > 0:
        print(f"Warning: {missing_count} pairs skipped due to missing cached v_fused embeddings.")
        
    return np.array(X), np.array(y)

def main():
    print("Loading training data...")
    X_train, y_train = load_data('train.csv')
    print("Loading testing data...")
    X_test, y_test = load_data('test.csv')
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: No features loaded. Please make sure to extract and cache v_fused embeddings first.")
        # Create empty results placeholder so run_all_experiments doesn't crash on run later
        results = {
            "Train n": 0,
            "Test n": 0,
            "Test %neutralizing": 0.0,
            "AUROC": 0.0,
            "AUPRC": 0.0,
            "C": 0.0
        }
        if os.path.exists('held_out_antibodies.csv'):
            results["Held-out antibodies"] = 0
        if os.path.exists('held_out_viruses.csv'):
            results["Held-out viruses"] = 0
        if os.path.exists('excluded_pairs.csv'):
            results["Excluded pairs"] = 0
            
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4)
        return
        
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    test_neut_pct = float(y_test.mean())
    print(f"Test neutralization rate: {test_neut_pct:.4f}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Fit Logistic Regression CV
    print("Fitting L2-regularised Logistic Regression CV...")
    clf = LogisticRegressionCV(
        Cs=C_VALUES,
        cv=5,
        penalty='l2',
        scoring='roc_auc',
        solver='lbfgs',
        max_iter=2000,
        random_state=SEED,
        n_jobs=-1
    )
    # Measure training time
    import time
    t0_train = time.time()
    clf.fit(X_train, y_train)
    t_train = time.time() - t0_train
    
    best_C = float(clf.C_[0])
    print(f"Best C selected: {best_C} (Training time: {t_train:.2f}s)")
    
    # Evaluate & measure inference time
    t0_inf = time.time()
    y_prob = clf.predict_proba(X_test)[:, 1]
    t_inf = time.time() - t0_inf
    
    # Calculate complementary metrics: AUROC and AUPRC
    auroc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auprc = auc(recall, precision)
    
    # Calculate supplementary metrics
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    from sklearn.metrics import precision_score, recall_score, matthews_corrcoef, confusion_matrix
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print("\n--- Evaluation Results ---")
    print(f"AUROC:     {auroc:.4f}")
    print(f"AUPRC:     {auprc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"Fit Time:  {t_train:.2f}s | Inf Time: {t_inf:.4f}s")
    
    # Save predictions CSV
    df_test = pd.read_csv('test.csv')
    df_test['y_true'] = y_test
    df_test['y_prob'] = y_prob
    df_test['y_pred'] = y_pred
    df_test['confidence'] = np.where(y_pred == 1, y_prob, 1 - y_prob)
    df_test.to_csv('predictions.csv', index=False)
    print("Saved test predictions to predictions.csv")
    
    # Save results json
    results = {
        "Train n": int(len(X_train)),
        "Test n": int(len(X_test)),
        "Test %neutralizing": round(test_neut_pct * 100, 2),
        "AUROC": round(float(auroc), 4),
        "AUPRC": round(float(auprc), 4),
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4),
        "MCC": round(float(mcc), 4),
        "C": best_C,
        "Training_Time_sec": round(t_train, 3),
        "Inference_Time_sec": round(t_inf, 4),
        "Confusion_Matrix": cm
    }
    
    if os.path.exists('held_out_antibodies.csv'):
        held_out_abs = pd.read_csv('held_out_antibodies.csv')
        results["Held-out antibodies"] = int(len(held_out_abs))
        
    if os.path.exists('held_out_viruses.csv'):
        held_out_virs = pd.read_csv('held_out_viruses.csv')
        results["Held-out viruses"] = int(len(held_out_virs))
        
    if os.path.exists('excluded_pairs.csv'):
        excl_pairs = pd.read_csv('excluded_pairs.csv')
        results["Excluded pairs"] = int(len(excl_pairs))
        
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to results.json.")

if __name__ == '__main__':
    main()
