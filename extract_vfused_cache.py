# -*- coding: utf-8 -*-
"""
extract_vfused_cache.py

Runs Phases B-D of the MambaCross model to extract v_fused embeddings
for all unique antibody-antigen pairs in the dataset.
Saves them to shared/v_fused_cache/ as `<antibody_id>___<virus_id>.npy`.

Optimized with a batched DataLoader to run fast on CPU/MPS.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, 'shared')
from Models import MambaCross

# Config
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_ROOT = 'shared'
CACHE_DIR = 'shared/v_fused_cache'
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class PairDataset(Dataset):
    def __init__(self, df, ab_cache, ag_cache):
        self.df = df
        self.ab_cache = ab_cache
        self.ag_cache = ag_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ab_id = row['antibody_id']
        ag_id = row['virus_id']
        safe_ab_id = ab_id.replace('/', '_')

        ab_emb = self.ab_cache[safe_ab_id]
        ag_emb = self.ag_cache[ag_id]

        return ab_id, ag_id, safe_ab_id, torch.FloatTensor(ab_emb), torch.FloatTensor(ag_emb)

def main():
    print(f"Using device for feature extraction: {device}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 1. Load Param_Model.json
    with open('shared/Param_Model.json', 'r') as f:
        param = json.load(f)
        
    # 2. Determine thresholds from original data files
    ab_info = pd.read_csv('Data/HIV/antibody.csv')
    ag_info = pd.read_csv('Data/HIV/antigen.csv')
    len_ab = ab_info['heavy'].fillna('').astype(str).str.len() + ab_info['light'].fillna('').astype(str).str.len()
    len_ag = ag_info['ag_seq'].astype(str).str.len()
    thres_ab = int(np.percentile(len_ab, 100))
    thres_ag = int(np.percentile(len_ag, 100))
    print(f"Max sequences thresholds: ab={thres_ab}, ag={thres_ag}")
    
    # 3. Initialize Model (Phases B-D)
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
    model.eval()
    
    # 4. Load base dataset
    df = pd.read_csv(os.path.join(DATA_ROOT, 'cleaned_dataset.csv'))
    unique_pairs = df[['antibody_id', 'virus_id']].drop_duplicates()
    print(f"Total unique pairs in dataset: {len(unique_pairs)}")
    
    # Filter out already cached pairs and check for missing raw files
    valid_rows = []
    missing_count = 0
    already_cached = 0
    
    for idx, row in unique_pairs.iterrows():
        ab_id = row['antibody_id']
        ag_id = row['virus_id']
        safe_ab_id = ab_id.replace('/', '_')
        out_path = os.path.join(CACHE_DIR, f"{safe_ab_id}___{ag_id}.npy")
        
        if os.path.exists(out_path):
            already_cached += 1
            continue
            
        ab_path = f"Outputs/Pretrained_HIV/ab/{safe_ab_id}.npy"
        ag_path = f"Outputs/Pretrained_HIV/ag/{ag_id}.npy"
        
        if os.path.exists(ab_path) and os.path.exists(ag_path):
            valid_rows.append(row)
        else:
            missing_count += 1
            
    print(f"Already cached: {already_cached}")
    print(f"Skipped due to missing ESM-2 files: {missing_count}")
    
    if len(valid_rows) == 0:
        print("All features are already cached! Nothing to run.")
        return
        
    valid_df = pd.DataFrame(valid_rows)
    print(f"Remaining pairs to project & cache: {len(valid_df)}")
    
    # Preload all raw ESM-2 embeddings into RAM to completely avoid slow disk I/O
    print("Preloading antibody embeddings to RAM...")
    ab_cache = {}
    for f in os.listdir('Outputs/Pretrained_HIV/ab'):
        if f.endswith('.npy') and not f.startswith('.'):
            name = f[:-4]
            ab_cache[name] = np.load(os.path.join('Outputs/Pretrained_HIV/ab', f))
            
    print("Preloading antigen embeddings to RAM...")
    ag_cache = {}
    for f in os.listdir('Outputs/Pretrained_HIV/ag'):
        if f.endswith('.npy') and not f.startswith('.'):
            name = f[:-4]
            ag_cache[name] = np.load(os.path.join('Outputs/Pretrained_HIV/ag', f))

    # 5. Run Batched Caching
    dataset = PairDataset(valid_df, ab_cache, ag_cache)
    # Using batch size of 64 and num_workers=0 (optimal for RAM-cached dataset)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=16)

    def save_array(path, arr):
        np.save(path, arr)

    extracted_count = 0
    with torch.no_grad():
        for batch_ab_ids, batch_ag_ids, batch_safe_ab_ids, ab_embs, ag_embs in tqdm(loader, desc="Caching v_fused"):
            ab_embs = ab_embs.to(device)
            ag_embs = ag_embs.to(device)
            
            # Bilinear Projection + VMamba sweeps + Pooling + Concatenation
            contacts = torch.matmul(torch.matmul(ab_embs, model.W), ag_embs.transpose(1, 2))
            h = model.mamba_hor(contacts)
            v = model.mamba_ver(contacts.transpose(1, 2))
            h = model.pool(h)
            v = model.pool(v)
            v_fused = torch.cat([h.squeeze(-1), v.squeeze(-1)], dim=-1) # (B, 1588)
            
            # Save vectors to disk in parallel using the thread pool
            v_fused_cpu = v_fused.cpu().numpy()
            for i in range(len(batch_ab_ids)):
                ab_id = batch_ab_ids[i]
                ag_id = batch_ag_ids[i]
                safe_ab_id = batch_safe_ab_ids[i]
                out_path = os.path.join(CACHE_DIR, f"{safe_ab_id}___{ag_id}.npy")
                executor.submit(save_array, out_path, v_fused_cpu[i])
                extracted_count += 1
                
    print("Waiting for all files to be written to disk...")
    executor.shutdown(wait=True)
    print(f"\nCaching complete! Generated {extracted_count} new vectors.")

if __name__ == '__main__':
    main()
