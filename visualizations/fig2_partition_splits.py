# -*- coding: utf-8 -*-
"""
fig2_partition_splits.py

Generates Figure 4.2: Data Split and Partitioning Breakdown Across Generalization Boundaries
Compares Train, Test, and Excluded pair counts across all 4 experiments.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def generate_figure(summary_csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(summary_csv_path)
    
    experiments = df['Experiment'].tolist()
    train_n = [int(x) for x in df['Train n'].tolist()]
    test_n = [int(x) for x in df['Test n'].tolist()]
    
    excluded_n = []
    for val in df['Excluded pairs'].tolist():
        if pd.isna(val) or val == 'N/A':
            excluded_n.append(0)
        else:
            excluded_n.append(int(float(val)))
            
    x = np.arange(len(experiments))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, train_n, width, label='Train Pairs', color='#1f77b4', edgecolor='black', linewidth=0.7)
    rects2 = ax.bar(x, test_n, width, label='Test Pairs', color='#ff7f0e', edgecolor='black', linewidth=0.7)
    rects3 = ax.bar(x + width, excluded_n, width, label='Excluded Pairs', color='#d62728', edgecolor='black', linewidth=0.7)
    
    ax.set_title('Figure 4.2: Data Partitioning Across Generalization Boundaries', pad=15, fontweight='bold')
    ax.set_ylabel('Number of Interaction Pairs (n)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.set_ylim(0, 75000)
    
    # Annotate bars with sample sizes and percentages
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:,}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8.5, rotation=0)
                            
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, 'fig4_2_partition_splits.png')
    pdf_path = os.path.join(output_dir, 'fig4_2_partition_splits.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Generated Figure 4.2 -> {png_path} and {pdf_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    summary_path = os.path.join(project_root, 'summary_results.csv')
    out_dir = os.path.join(project_root, 'visualizations', 'figures')
    generate_figure(summary_path, out_dir)
