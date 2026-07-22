# -*- coding: utf-8 -*-
"""
fig1_dataset_distribution.py

Generates Figure 4.1: Dataset Composition and Class Distribution
Analyzes the 74,730 HIV antibody-antigen interaction pairs.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
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

def generate_figure(dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(dataset_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 1.3, 1.3]})
    
    # Color palette
    colors = ['#2b5c8f', '#d95f02']
    
    # 1. Class Distribution
    class_counts = df['neut'].value_counts()
    class_pcts = df['neut'].value_counts(normalize=True) * 100
    labels = ['Neutralizing (1)' if idx == 1 else 'Non-Neutralizing (0)' for idx in class_counts.index]
    colors = ['#d95f02' if idx == 1 else '#2b5c8f' for idx in class_counts.index]
    
    bars = axes[0].bar(labels, class_counts.values, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    axes[0].set_title('A. Neutralization Target Class Distribution', pad=12, fontweight='bold')
    axes[0].set_ylabel('Number of Interaction Pairs (n)')
    axes[0].set_ylim(0, max(class_counts.values) * 1.15)
    
    for bar, count, pct in zip(bars, class_counts.values, class_pcts.values):
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 1000, f"{count:,}\n({pct:.1f}%)", 
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    # 2. Top 15 Most Frequent Antibodies
    ab_counts = df['antibody_id'].value_counts().head(15)
    sns.barplot(x=ab_counts.values, y=ab_counts.index, ax=axes[1], palette='Blues_r', edgecolor='black', linewidth=0.6)
    axes[1].set_title('B. Top 15 Most Frequent Antibodies', pad=12, fontweight='bold')
    axes[1].set_xlabel('Number of Pairs (n)')
    axes[1].set_ylabel('Antibody ID')
    
    for i, v in enumerate(ab_counts.values):
        axes[1].text(v + 10, i, f"{v}", va='center', fontsize=9)
    axes[1].set_xlim(0, max(ab_counts.values) * 1.12)
    
    # 3. Top 15 Most Frequent Viruses
    vir_counts = df['virus_id'].value_counts().head(15)
    sns.barplot(x=vir_counts.values, y=vir_counts.index, ax=axes[2], palette='Oranges_r', edgecolor='black', linewidth=0.6)
    axes[2].set_title('C. Top 15 Most Frequent Viral Strains', pad=12, fontweight='bold')
    axes[2].set_xlabel('Number of Pairs (n)')
    axes[2].set_ylabel('Virus Strain ID')
    
    for i, v in enumerate(vir_counts.values):
        axes[2].text(v + 5, i, f"{v}", va='center', fontsize=9)
    axes[2].set_xlim(0, max(vir_counts.values) * 1.12)
    
    plt.tight_layout()
    
    # Save artifacts
    png_path = os.path.join(output_dir, 'fig4_1_dataset_distribution.png')
    pdf_path = os.path.join(output_dir, 'fig4_1_dataset_distribution.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Generated Figure 4.1 -> {png_path} and {pdf_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    data_file = os.path.join(project_root, 'shared', 'cleaned_dataset.csv')
    out_dir = os.path.join(project_root, 'visualizations', 'figures')
    generate_figure(data_file, out_dir)
