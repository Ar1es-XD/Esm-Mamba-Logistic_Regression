# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import esm
import torch.nn.functional as F
import os

data_root = 'Data/HIV'
ab_info_path = '%s/antibody.csv'%data_root
ag_info_path = '%s/antigen.csv'%data_root

ab_info = pd.read_csv(ab_info_path, index_col=None, header=0)
ag_info = pd.read_csv(ag_info_path, index_col=None, header=0)

# We want the max length to be the 100th percentile for HIV data
# Lengths of heavy and light chains
len_h = [len(str(l)) for l in ab_info['heavy']]
len_l = [len(str(l)) for l in ab_info['light']]
len_ag = [len(str(l)) for l in ag_info['ag_seq']]

# Antibody total length is Heavy + Light
len_ab = [h + l for h, l in zip(len_h, len_l)]

thres_ab = int(np.percentile(len_ab, 100))
thres_ag = int(np.percentile(len_ag, 100))
print(f"Max AB length (H+L): {thres_ab}")
print(f"Max AG length: {thres_ag}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def alphabet_coding(data_list, is_antibody, maxlen, save_dir):
    """
    If is_antibody is True, data_list is a list of (name, heavy_seq, light_seq)
    If is_antibody is False, data_list is a list of (name, ag_seq)
    """
    ESM_encoder, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    os.makedirs(save_dir, exist_ok=True)
    ESM_encoder = ESM_encoder.eval().to(device)

    for i, item in enumerate(data_list):
        if i % 10 == 0:
            print(f"Processing {i}/{len(data_list)}")
        
        name = item[0]
        try:
            with torch.no_grad():
                if is_antibody:
                    _, heavy_seq, light_seq = item
                    
                    # Embed Heavy
                    _, _, t_h = batch_converter([(name, heavy_seq)])
                    t_h = t_h.to(device)
                    L_h = (t_h != alphabet.padding_idx).sum().item()
                    res_h = ESM_encoder(t_h, repr_layers=[6], return_contacts=False)
                    emb_h = res_h["representations"][6][0, 1:L_h-1] # (len_h, 320)
                    
                    # Embed Light
                    _, _, t_l = batch_converter([(name, light_seq)])
                    t_l = t_l.to(device)
                    L_l = (t_l != alphabet.padding_idx).sum().item()
                    res_l = ESM_encoder(t_l, repr_layers=[6], return_contacts=False)
                    emb_l = res_l["representations"][6][0, 1:L_l-1] # (len_l, 320)
                    
                    # Concatenate structurally (H + L) 
                    # This avoids the fake peptide bond issue of running ESM-2 on H+L string
                    seq_feats = torch.cat([emb_h, emb_l], dim=0) # (len_h + len_l, 320)
                    
                else:
                    _, ag_seq = item
                    _, _, t_ag = batch_converter([(name, ag_seq)])
                    t_ag = t_ag.to(device)
                    L_ag = (t_ag != alphabet.padding_idx).sum().item()
                    res_ag = ESM_encoder(t_ag, repr_layers=[6], return_contacts=False)
                    seq_feats = res_ag["representations"][6][0, 1:L_ag-1] # (len_ag, 320)
                
                # Zero-pad to maxlen
                pad_size = maxlen - seq_feats.size(0)
                if pad_size > 0:
                    # F.pad pads the last dimension first, then the second to last.
                    # seq_feats is (L, 320). We want to pad L by pad_size at the bottom.
                    # format: (pad_left, pad_right, pad_top, pad_bottom)
                    padded_tensor = F.pad(seq_feats, (0, 0, 0, pad_size), mode='constant', value=0)
                else:
                    padded_tensor = seq_feats[:maxlen, :]
                
                np.save(f"{save_dir}/{name}.npy", padded_tensor.cpu().numpy())
        except Exception as e:
            print(f"Failed on {name}: {e}")

ab_data = [(row['ab_name'], row['heavy'], row['light']) for _, row in ab_info.iterrows()]
ag_data = [(row['ag_name'], row['ag_seq']) for _, row in ag_info.iterrows()]

# Filter by length just in case
ab_data = [x for x in ab_data if (len(str(x[1])) + len(str(x[2]))) <= thres_ab]
ag_data = [x for x in ag_data if len(str(x[1])) <= thres_ag]

print("Extracting Antigen Embeddings...")
alphabet_coding(ag_data, is_antibody=False, maxlen=thres_ag, save_dir='Outputs/Pretrained_HIV/ag')

print("Extracting Antibody Embeddings (Heavy + Light)...")
alphabet_coding(ab_data, is_antibody=True, maxlen=thres_ab, save_dir='Outputs/Pretrained_HIV/ab')
