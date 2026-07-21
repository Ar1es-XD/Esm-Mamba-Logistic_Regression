# -*- coding: utf-8 -*-
"""
run_all_visualizations.py

Master script to run all figure generation scripts sequentially.
Outputs all PNG (300 DPI) and PDF figures to visualizations/figures/.
"""

import os
import sys
import subprocess

VISUALIZATION_SCRIPTS = [
    "fig1_dataset_distribution.py",
    "fig2_partition_splits.py",
    "fig3_sequence_lengths.py",
    "fig4_esm_embedding_pca.py",
    "fig5_fused_feature_pca.py",
    "fig6_benchmark_performance.py",
    "fig7_generalization_degradation.py",
    "fig8_model_diagnostics.py"
]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    python_exe = sys.executable
    
    print("=================================================================")
    print(" Generating Thesis Visualizations (PNG 300 DPI + PDF)")
    print("=================================================================\n")
    
    for script_name in VISUALIZATION_SCRIPTS:
        script_path = os.path.join(script_dir, script_name)
        if os.path.exists(script_path):
            print(f">>> Running {script_name}...")
            try:
                subprocess.run([python_exe, script_name], cwd=script_dir, check=True)
            except Exception as e:
                print(f"Error running {script_name}: {e}")
        else:
            print(f"Warning: Script {script_name} not found.")
            
    print("\n=================================================================")
    print(f" All Visualizations Generated in {figures_dir}")
    print("=================================================================")

if __name__ == '__main__':
    main()
