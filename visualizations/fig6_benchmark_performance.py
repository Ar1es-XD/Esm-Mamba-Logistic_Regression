# -*- coding: utf-8 -*-
"""
fig6_benchmark_performance.py

Generates Figure 6.1: Benchmark Model Performance (AUROC and AUPRC) Across Generalization Settings
Compares predictive capacity across Random Split, Novel Viruses, Novel Antibodies, and Both Novel.
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
    auroc_scores = [float(x) for x in df['AUROC'].tolist()]
    auprc_scores = [float(x) for x in df['AUPRC'].tolist()]
    
    x = np.arange(len(experiments))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, auroc_scores, width, label='AUROC', color='#1f77b4', edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, auprc_scores, width, label='AUPRC', color='#2ca02c', edgecolor='black', linewidth=0.8)
    
    ax.set_title('Figure 6.1: Logistic Regression Benchmark Performance Across Generalization Boundaries', pad=15, fontweight='bold')
    ax.set_ylabel('Metric Score', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontweight='bold')
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random Chance (0.50)')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower left')
    
    # Annotate exact numbers on bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1f77b4')
                    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#2ca02c')
                    
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, 'fig6_1_benchmark_performance.png')
    pdf_path = os.path.join(output_dir, 'fig6_1_benchmark_performance.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Generated Figure 6.1 -> {png_path} and {pdf_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    summary_path = os.path.join(project_root, 'summary_results.csv')
    out_dir = os.path.join(project_root, 'visualizations', 'figures')
    generate_figure(summary_path, out_dir)
