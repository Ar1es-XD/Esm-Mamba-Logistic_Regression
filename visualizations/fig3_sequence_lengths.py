# -*- coding: utf-8 -*-
"""
fig3_sequence_lengths.py

Generates Figure 4.3: Sequence Length Distribution of Antibodies and Antigens
Analyzes heavy + light chain antibody lengths and viral antigen envelope protein lengths.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def generate_figure(ab_csv_path, ag_csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    ab_df = pd.read_csv(ab_csv_path)
    ag_df = pd.read_csv(ag_csv_path)
    
    # Calculate lengths
    len_h = ab_df['heavy'].fillna('').astype(str).str.len()
    len_l = ab_df['light'].fillna('').astype(str).str.len()
    ab_lengths = len_h + len_l
    ag_lengths = ag_df['ag_seq'].astype(str).str.len()
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel A: Antibodies (Heavy + Light)
    sns.histplot(ab_lengths, kde=True, ax=axes[0], color='#2b5c8f', bins=25, edgecolor='black', linewidth=0.6)
    max_ab = int(np.max(ab_lengths))
    p95_ab = int(np.percentile(ab_lengths, 95))
    mean_ab = np.mean(ab_lengths)
    
    axes[0].axvline(mean_ab, color='red', linestyle='--', linewidth=1.5, label=f'Mean Length ({mean_ab:.1f} aa)')
    axes[0].axvline(max_ab, color='black', linestyle=':', linewidth=1.5, label=f'Max Truncation Limit ({max_ab} aa)')
    
    axes[0].set_title('A. Antibody Sequence Length Distribution (Heavy + Light)', pad=12, fontweight='bold')
    axes[0].set_xlabel('Combined Amino Acid Length')
    axes[0].set_ylabel('Antibody Frequency Count')
    axes[0].legend(frameon=True, facecolor='white')
    
    # Panel B: Antigens (gp120 / gp160 Envelopes)
    sns.histplot(ag_lengths, kde=True, ax=axes[1], color='#d95f02', bins=30, edgecolor='black', linewidth=0.6)
    max_ag = int(np.max(ag_lengths))
    mean_ag = np.mean(ag_lengths)
    
    axes[1].axvline(mean_ag, color='blue', linestyle='--', linewidth=1.5, label=f'Mean Length ({mean_ag:.1f} aa)')
    axes[1].axvline(max_ag, color='black', linestyle=':', linewidth=1.5, label=f'Max Truncation Limit ({max_ag} aa)')
    
    axes[1].set_title('B. Antigen Sequence Length Distribution (gp120/gp160)', pad=12, fontweight='bold')
    axes[1].set_xlabel('Amino Acid Length')
    axes[1].set_ylabel('Antigen Frequency Count')
    axes[1].legend(frameon=True, facecolor='white')
    
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, 'fig4_3_sequence_lengths.png')
    pdf_path = os.path.join(output_dir, 'fig4_3_sequence_lengths.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Generated Figure 4.3 -> {png_path} and {pdf_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    ab_path = os.path.join(project_root, 'Data', 'HIV', 'antibody.csv')
    ag_path = os.path.join(project_root, 'Data', 'HIV', 'antigen.csv')
    out_dir = os.path.join(project_root, 'visualizations', 'figures')
    generate_figure(ab_path, ag_path, out_dir)
