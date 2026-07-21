# -*- coding: utf-8 -*-
"""
fig7_generalization_degradation.py

Generates Figure 6.2: Degradation of Predictive Power Across Biological Complexity Levels
Highlights interpolation-to-extrapolation drop and antibody vs virus generalization asymmetry.
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
    auroc = [float(x) for x in df['AUROC'].tolist()]
    auprc = [float(x) for x in df['AUPRC'].tolist()]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    # Line plot with markers
    ax.plot(experiments, auroc, marker='o', markersize=9, linewidth=2.5, color='#1f77b4', label='AUROC')
    ax.plot(experiments, auprc, marker='s', markersize=9, linewidth=2.5, color='#2ca02c', label='AUPRC')
    
    # Fill between to show gap
    ax.fill_between(range(len(experiments)), auroc, auprc, color='#1f77b4', alpha=0.1, label='AUROC–AUPRC Gap')
    
    ax.set_title('Figure 6.2: Generalization Degradation Curve Across Biological Settings', pad=15, fontweight='bold')
    ax.set_ylabel('Metric Score', fontweight='bold')
    ax.set_xlabel('Biological Generalization Setting', fontweight='bold')
    ax.set_ylim(0.65, 0.95)

    # Annotate key drops and asymmetry
    for i, (exp, a_score, p_score) in enumerate(zip(experiments, auroc, auprc)):
        ax.annotate(f'{a_score:.4f}', (i, a_score), xytext=(0, 10), textcoords='offset points', ha='center', fontweight='bold', color='#1f77b4')
        ax.annotate(f'{p_score:.4f}', (i, p_score), xytext=(0, -18), textcoords='offset points', ha='center', fontweight='bold', color='#2ca02c')

    # Add annotation text for antibody vs virus asymmetry
    asym_gap = auroc[1] - auroc[2]
    ax.annotate(f'Antibody-Virus Asymmetry Gap: Δ={asym_gap:.4f}',
                xy=(1.5, (auroc[1] + auroc[2])/2), xytext=(1.5, 0.91),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='linen', edgecolor='orange'))

    # Add total drop annotation
    total_drop = auroc[0] - auroc[3]
    ax.annotate(f'Interpolation vs Extrapolation Drop: Δ={total_drop:.4f}',
                xy=(3, auroc[3]), xytext=(2.6, 0.68),
                arrowprops=dict(facecolor='red', shrink=0.08, width=1, headwidth=6),
                ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='mistyrose', edgecolor='red'))

    ax.legend(frameon=True, facecolor='white', loc='lower left')
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, 'fig6_2_generalization_degradation.png')
    pdf_path = os.path.join(output_dir, 'fig6_2_generalization_degradation.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Generated Figure 6.2 -> {png_path} and {pdf_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    summary_path = os.path.join(project_root, 'summary_results.csv')
    out_dir = os.path.join(project_root, 'visualizations', 'figures')
    generate_figure(summary_path, out_dir)
