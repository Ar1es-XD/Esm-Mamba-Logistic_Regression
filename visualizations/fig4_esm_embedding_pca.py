# -*- coding: utf-8 -*-
"""
fig4_esm_embedding_pca.py

Generates Figure 4.4: Principal Component Analysis (PCA) of ESM-2 Sequence Embeddings
Loads mean-pooled 320-dimensional ESM-2 representations for antibodies and antigens.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

def load_pooled_embeddings(dir_path, max_samples=500):
    files = [f for f in os.listdir(dir_path) if f.endswith('.npy') and not f.startswith('.')]
    embs = []
    names = []
    for f in files[:max_samples]:
        arr = np.load(os.path.join(dir_path, f))
        # Mean pool non-zero residue vectors
        non_zero_mask = np.abs(arr).sum(axis=-1) > 1e-6
        if np.any(non_zero_mask):
            mean_vec = arr[non_zero_mask].mean(axis=0)
        else:
            mean_vec = arr.mean(axis=0)
        embs.append(mean_vec)
        names.append(f[:-4])
    return np.array(embs), names

def generate_figure(ab_emb_dir, ag_emb_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    ab_vecs, ab_names = load_pooled_embeddings(ab_emb_dir, max_samples=600)
    ag_vecs, ag_names = load_pooled_embeddings(ag_emb_dir, max_samples=600)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Antibody ESM-2 PCA
    if len(ab_vecs) > 0:
        pca_ab = PCA(n_components=2)
        ab_pca = pca_ab.fit_transform(ab_vecs)
        var_ab = pca_ab.explained_variance_ratio_ * 100
        
        scatter_ab = axes[0].scatter(ab_pca[:, 0], ab_pca[:, 1], c='#2b5c8f', alpha=0.75, edgecolors='k', linewidths=0.4, s=40)
        axes[0].set_title(f'A. Antibody ESM-2 Feature Space (n={len(ab_vecs)})\nVar Explained: PC1={var_ab[0]:.1f}%, PC2={var_ab[1]:.1f}%', pad=12, fontweight='bold')
        axes[0].set_xlabel('Principal Component 1 (PC1)')
        axes[0].set_ylabel('Principal Component 2 (PC2)')
        
    # 2. Antigen ESM-2 PCA
    if len(ag_vecs) > 0:
        pca_ag = PCA(n_components=2)
        ag_pca = pca_ag.fit_transform(ag_vecs)
        var_ag = pca_ag.explained_variance_ratio_ * 100
        
        scatter_ag = axes[1].scatter(ag_pca[:, 0], ag_pca[:, 1], c='#d95f02', alpha=0.65, edgecolors='k', linewidths=0.4, s=35)
        axes[1].set_title(f'B. Antigen ESM-2 Feature Space (n={len(ag_vecs)})\nVar Explained: PC1={var_ag[0]:.1f}%, PC2={var_ag[1]:.1f}%', pad=12, fontweight='bold')
        axes[1].set_xlabel('Principal Component 1 (PC1)')
        axes[1].set_ylabel('Principal Component 2 (PC2)')
        
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, 'fig4_4_esm_embedding_pca.png')
    pdf_path = os.path.join(output_dir, 'fig4_4_esm_embedding_pca.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Generated Figure 4.4 -> {png_path} and {pdf_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    ab_dir = os.path.join(project_root, 'Outputs', 'Pretrained_HIV', 'ab')
    ag_dir = os.path.join(project_root, 'Outputs', 'Pretrained_HIV', 'ag')
    out_dir = os.path.join(project_root, 'visualizations', 'figures')
    generate_figure(ab_dir, ag_dir, out_dir)
