# -*- coding: utf-8 -*-
"""
fig5_fused_feature_pca.py

Generates Figure 5.1: Feature Space Projection of 1,588-dim v_fused Interaction Representations
Projects cached fused biophysical interaction vectors using PCA and t-SNE, colored by neutralization class.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

def generate_figure(cache_dir, dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(dataset_path)
    
    # Map cached files to dataset labels
    pair_label_map = {}
    for _, row in df.iterrows():
        ab_id = row['antibody_id'].replace('/', '_')
        ag_id = row['virus_id']
        pair_label_map[f"{ab_id}___{ag_id}.npy"] = int(row['neut'])
        
    cached_files = [f for f in os.listdir(cache_dir) if f.endswith('.npy') and not f.startswith('.')]
    
    X, y = [], []
    for f in cached_files:
        if f in pair_label_map:
            arr = np.load(os.path.join(cache_dir, f))
            X.append(arr)
            y.append(pair_label_map[f])
            
    X = np.array(X)
    y = np.array(y)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = {0: '#2b5c8f', 1: '#d95f02'}
    label_names = {0: 'Non-Neutralizing', 1: 'Neutralizing'}
    
    if len(X) > 0:
        # 1. PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        var_ratio = pca.explained_variance_ratio_ * 100
        
        for label_val in [0, 1]:
            mask = (y == label_val)
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=palette[label_val],
                            label=label_names[label_val], alpha=0.75, edgecolors='k', linewidths=0.3, s=40)
            
        axes[0].set_title(f'A. PCA of 1,588-dim $v_{{fused}}$ Vectors (n={len(X)})\nVar Explained: PC1={var_ratio[0]:.1f}%, PC2={var_ratio[1]:.1f}%', pad=12, fontweight='bold')
        axes[0].set_xlabel('Principal Component 1 (PC1)')
        axes[0].set_ylabel('Principal Component 2 (PC2)')
        axes[0].legend(frameon=True, facecolor='white')
        
        # 2. t-SNE
        perplexity = min(30, max(5, len(X) - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(X)
        
        for label_val in [0, 1]:
            mask = (y == label_val)
            axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=palette[label_val],
                            label=label_names[label_val], alpha=0.75, edgecolors='k', linewidths=0.3, s=40)
            
        axes[1].set_title(f'B. t-SNE Projection of Fused Feature Representations\n(Perplexity={perplexity})', pad=12, fontweight='bold')
        axes[1].set_xlabel('t-SNE Dimension 1')
        axes[1].set_ylabel('t-SNE Dimension 2')
        axes[1].legend(frameon=True, facecolor='white')
    else:
        axes[0].text(0.5, 0.5, "No cached v_fused vectors found", ha='center', va='center')
        axes[1].text(0.5, 0.5, "No cached v_fused vectors found", ha='center', va='center')
        
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, 'fig5_1_fused_feature_pca.png')
    pdf_path = os.path.join(output_dir, 'fig5_1_fused_feature_pca.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Generated Figure 5.1 -> {png_path} and {pdf_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    cache_dir = os.path.join(project_root, 'shared', 'v_fused_cache')
    dataset_path = os.path.join(project_root, 'shared', 'cleaned_dataset.csv')
    out_dir = os.path.join(project_root, 'visualizations', 'figures')
    generate_figure(cache_dir, dataset_path, out_dir)
