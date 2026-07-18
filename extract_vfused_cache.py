# -*- coding: utf-8 -*-
"""
extract_vfused_cache.py

Runs Phases B-D of the MambaCross model to extract v_fused embeddings
for all unique antibody-antigen pairs in the dataset.
Saves them to shared/v_fused_cache/ as `<antibody_id>___<virus_id>.npy`.

Run AFTER Pretrained.py has generated individual ESM-2 embeddings.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from Models import MambaCross

# Config
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_ROOT = 'shared'
CACHE_DIR = 'shared/v_fused_cache'
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def main():
    print(f"Using device for feature extraction: {device}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 1. Load Param_Model.json
    with open('Param_Model.json', 'r') as f:
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
    print(f"Total unique pairs to project: {len(unique_pairs)}")
    
    # 5. Extract and cache
    extracted_count = 0
    missing_count = 0
    
    with torch.no_grad():
        for idx, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Caching v_fused"):
            ab_id = row['antibody_id']
            ag_id = row['virus_id']
            
            # Safe filename replacing slash to avoid path directory issues
            safe_ab_id = ab_id.replace('/', '_')
            out_path = os.path.join(CACHE_DIR, f"{safe_ab_id}___{ag_id}.npy")
            
            # Skip if already cached
            if os.path.exists(out_path):
                continue
                
            # Load individual ESM-2 embeddings
            ab_emb_path = f"Outputs/Pretrained_HIV/ab/{ab_id}.npy"
            ag_emb_path = f"Outputs/Pretrained_HIV/ag/{ag_id}.npy"
            
            if not os.path.exists(ab_emb_path) or not os.path.exists(ag_emb_path):
                missing_count += 1
                continue
                
            # Load and convert to tensor (batch_size = 1)
            ab_emb = torch.FloatTensor(np.load(ab_emb_path)).unsqueeze(0).to(device) # (1, 676, 320)
            ag_emb = torch.FloatTensor(np.load(ag_emb_path)).unsqueeze(0).to(device) # (1, 912, 320)
            
            # Run Bilinear projection W + Dual Mamba Sweeps + Pooling + Concatenation
            contacts = torch.matmul(torch.matmul(ab_emb, model.W), ag_emb.transpose(1, 2))
            h = model.mamba_hor(contacts)
            v = model.mamba_ver(contacts.transpose(1, 2))
            h = model.pool(h)
            v = model.pool(v)
            v_fused = torch.cat([h.squeeze(-1), v.squeeze(-1)], dim=-1) # (1, 1588)
            
            # Save vector to disk
            np.save(out_path, v_fused.squeeze(0).cpu().numpy())
            extracted_count += 1
            
    print(f"\nCaching complete! Generated {extracted_count} new vectors.")
    if missing_count > 0:
        print(f"Warning: {missing_count} pairs skipped due to missing individual ESM-2 embeddings. (Run Pretrained.py first)")

if __name__ == '__main__':
    main()
