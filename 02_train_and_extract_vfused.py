# -*- coding: utf-8 -*-
"""
02_train_and_extract_vfused.py

For each of the 4 mentor splits (random, ab_block, vir_block, both_block):
  1. Train the full MambaCross model (Phases B-E) end-to-end.
  2. Record the MLP's test AUC.
  3. Extract v_fused (Phase D output) for all train+test pairs.
  4. Cache v_fused vectors + labels to disk for LR fitting.

Run AFTER Pretrained.py has generated ESM-2 embeddings.
Designed for: AMD Ryzen 9 9900X, 64GB DDR5, RTX PRO 4000 (24GB VRAM).
"""
import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models import MambaCross
from Toolkit import Metrics, set_seed_all, make_dir, AntibodyAntigenDataset, custom_collate_fn

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
DATA_ROOT = 'Data/HIV'
CACHE_DIR = 'cache'
RESULTS_DIR = 'Results'
SPLITS = ['random', 'ab_block', 'vir_block', 'both_block']
EPOCHS = 50

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# ── Helpers ─────────────────────────────────────────────────────────────
def load_split(split_col):
    """Load train/test pairs from the mentor's hardcoded splits."""
    pairs_df = pd.read_csv(f'{DATA_ROOT}/ab_ag_pair.csv', low_memory=False)
    train_df = pairs_df[pairs_df[split_col] == 'train']
    test_df  = pairs_df[pairs_df[split_col] == 'test']
    train_pairs = list(zip(train_df['ab_name'], train_df['ag_name'], train_df['label']))
    test_pairs  = list(zip(test_df['ab_name'],  test_df['ag_name'],  test_df['label']))
    print(f'Split [{split_col}]: {len(train_pairs)} train, {len(test_pairs)} test')
    return train_pairs, test_pairs


def filter_valid_pairs(pairs):
    """Drop pairs whose ESM-2 embedding .npy files are missing."""
    valid = []
    for ab_id, ag_id, label in pairs:
        ab_path = f'Outputs/Pretrained_HIV/ab/{ab_id}.npy'
        ag_path = f'Outputs/Pretrained_HIV/ag/{ag_id}.npy'
        if os.path.exists(ab_path) and os.path.exists(ag_path):
            valid.append((ab_id, ag_id, label))
    if len(valid) < len(pairs):
        print(f'  Dropped {len(pairs) - len(valid)} pairs with missing embeddings')
    return valid


def extract_vfused(model, x_Ab, x_Ag):
    """Run Phases B-D of MambaCross, returning v_fused without the MLP head.
    
    Phase B: bilinear contact grid  (W matrix)
    Phase C: dual Mamba sweeps      (mamba_hor, mamba_ver)
    Phase D: adaptive pooling + cat (pool -> squeeze -> concat)
    """
    with torch.no_grad():
        contacts = torch.matmul(torch.matmul(x_Ab, model.W), x_Ag.transpose(1, 2))
        h = model.mamba_hor(contacts)
        v = model.mamba_ver(contacts.transpose(1, 2))
        h = model.pool(h)
        v = model.pool(v)
        v_fused = torch.cat([h.squeeze(-1), v.squeeze(-1)], dim=-1)
    return v_fused


# ── Training loop ───────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    for ab, ag, labels in tqdm(loader, desc='  train', leave=False):
        ab, ag, labels = ab.to(device), ag.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(ab, ag)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        y_true += labels.cpu().tolist()
        y_pred += preds.cpu().detach().tolist()
    auc, aupr, f1, acc = Metrics(y_true, y_pred)
    return total_loss / len(loader.dataset), auc


@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    criterion = nn.BCELoss()
    for ab, ag, labels in loader:
        ab, ag, labels = ab.to(device), ag.to(device), labels.to(device)
        preds = model(ab, ag)
        loss = criterion(preds, labels)
        total_loss += loss.item()
        y_true += labels.cpu().tolist()
        y_pred += preds.cpu().detach().tolist()
    auc, aupr, f1, acc = Metrics(y_true, y_pred)
    return auc, aupr, f1, acc, total_loss


@torch.no_grad()
def extract_all_vfused(model, loader):
    """Extract v_fused for every sample in loader."""
    model.eval()
    all_vfused, all_labels = [], []
    for ab, ag, labels in tqdm(loader, desc='  extract v_fused', leave=False):
        ab, ag = ab.to(device), ag.to(device)
        vf = extract_vfused(model, ab, ag)
        all_vfused.append(vf.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_vfused, axis=0), np.concatenate(all_labels, axis=0)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    set_seed_all(SEED)
    make_dir(CACHE_DIR)
    make_dir(RESULTS_DIR)

    with open('Param_Model.json', 'r') as f:
        param = json.load(f)

    # Determine thresholds from data (must match Pretrained.py)
    ab_info = pd.read_csv(f'{DATA_ROOT}/antibody.csv')
    ag_info = pd.read_csv(f'{DATA_ROOT}/antigen.csv')
    len_ab = ab_info['heavy'].fillna('').astype(str).str.len() + \
             ab_info['light'].fillna('').astype(str).str.len()
    len_ag = ag_info['ag_seq'].astype(str).str.len()
    thres_ab = int(np.percentile(len_ab, 100))
    thres_ag = int(np.percentile(len_ag, 100))
    print(f'Thresholds: ab={thres_ab}, ag={thres_ag}, v_fused_dim={thres_ab+thres_ag}')

    mlp_results = {}  # split -> {auc, aupr, f1, acc}

    for split in SPLITS:
        print(f'\n{"="*60}')
        print(f'SPLIT: {split}')
        print(f'{"="*60}')

        train_pairs, test_pairs = load_split(split)
        train_pairs = filter_valid_pairs(train_pairs)
        test_pairs  = filter_valid_pairs(test_pairs)

        train_ds = AntibodyAntigenDataset(train_pairs)
        test_ds  = AntibodyAntigenDataset(test_pairs)
        train_loader = DataLoader(train_ds, batch_size=param['batchsize'],
                                  shuffle=True,  num_workers=4,
                                  collate_fn=custom_collate_fn, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=param['batchsize'],
                                  shuffle=False, num_workers=4,
                                  collate_fn=custom_collate_fn, pin_memory=True)

        # Build model
        model = MambaCross(
            hor_dim=thres_ag, ver_dim=thres_ab,
            feat_dim=param['latent_dim'],
            seq_len=thres_ab + thres_ag,
            hidden_sizes=param['decoder_hidden_dims'],
            mamba_layer=param['mamba_layer'],
            pooling=param['pooling_way'],
            activation=param['activation'],
            drop_ratio=param['dropout']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=param['learning_rate'], weight_decay=0)
        criterion = nn.BCELoss()

        # ── Train ───────────────────────────────────────────────────
        best_loss = math.inf
        ckpt_path = os.path.join(RESULTS_DIR, f'model_{split}.pt')

        for epoch in range(EPOCHS):
            loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion)
            auc, aupr, f1, acc, val_loss = eval_model(model, test_loader)
            print(f'  epoch {epoch+1:3d}/{EPOCHS}  '
                  f'train_loss={loss:.4f}  test_AUC={auc:.4f}  test_AUPR={aupr:.4f}')
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), ckpt_path)
                best_metrics = dict(auc=auc, aupr=aupr, f1=f1, acc=acc)

        # Reload best
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        mlp_results[split] = best_metrics
        print(f'  Best MLP  AUC={best_metrics["auc"]:.4f}  '
              f'AUPR={best_metrics["aupr"]:.4f}')

        # ── Extract v_fused ─────────────────────────────────────────
        # Use shuffle=False for deterministic ordering
        train_loader_ordered = DataLoader(train_ds, batch_size=param['batchsize'],
                                          shuffle=False, num_workers=4,
                                          collate_fn=custom_collate_fn, pin_memory=True)

        vf_train, y_train = extract_all_vfused(model, train_loader_ordered)
        vf_test,  y_test  = extract_all_vfused(model, test_loader)

        # Save to cache
        np.save(os.path.join(CACHE_DIR, f'{split}_vfused_train.npy'), vf_train)
        np.save(os.path.join(CACHE_DIR, f'{split}_labels_train.npy'), y_train)
        np.save(os.path.join(CACHE_DIR, f'{split}_vfused_test.npy'),  vf_test)
        np.save(os.path.join(CACHE_DIR, f'{split}_labels_test.npy'),  y_test)

        print(f'  Cached v_fused  train={vf_train.shape}  test={vf_test.shape}')

    # ── Save MLP results summary ────────────────────────────────────
    mlp_df = pd.DataFrame(mlp_results).T
    mlp_df.index.name = 'split'
    mlp_df.to_csv(os.path.join(RESULTS_DIR, 'mlp_results.csv'))
    print(f'\n=== MLP Results Summary ===')
    print(mlp_df.to_string())
    print(f'\nAll v_fused cached to {CACHE_DIR}/. Run 03_fit_logistic_regression.py next.')


if __name__ == '__main__':
    main()
