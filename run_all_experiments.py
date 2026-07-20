# -*- coding: utf-8 -*-
"""
run_all_experiments.py

Runs all 4 self-contained experiments in sequence and aggregates their metrics
into a top-level summary_results.csv file.
"""
import os
import sys
import json
import subprocess
import pandas as pd

# Experiment directories and names mapped to prompt requirements
EXPERIMENTS = [
    ("experiment_1_random", "Random"),
    ("experiment_2_novel_viruses", "Novel viruses"),
    ("experiment_3_novel_antibodies", "Novel antibodies"),
    ("experiment_4_both_novel", "Both novel")
]

def main():
    print("=================================================================")
    # Correct name check
    print(" Running All Logistic Regression Experiments In Sequence")
    print("=================================================================\n")
    
    summary_rows = []
    
    for folder, name in EXPERIMENTS:
        print(f"\n>>> Running {name} Experiment ({folder})...")
        script_path = os.path.join(folder, "train_lr.py")
        
        # Check if train_lr.py exists
        if not os.path.exists(script_path):
            print(f"Error: {script_path} not found.")
            continue
            
        # Invoke train_lr.py in its own directory
        # This keeps the execution context fully self-contained inside the experiment folder
        try:
            subprocess.run([sys.executable, "train_lr.py"], cwd=folder, check=True)
        except Exception as e:
            print(f"Subprocess failed for {name}: {e}")
            
        # Read the resulting results.json
        results_json_path = os.path.join(folder, "results.json")
        if os.path.exists(results_json_path):
            try:
                with open(results_json_path, 'r') as f:
                    metrics = json.load(f)
                    
                summary_rows.append({
                    "Experiment": name,
                    "Train n": metrics.get("Train n", 0),
                    "Test n": metrics.get("Test n", 0),
                    "Test %neutralizing": f"{metrics.get('Test %neutralizing', 0.0):.2f}%",
                    "AUROC": f"{metrics.get('AUROC', 0.0):.4f}",
                    "AUPRC": f"{metrics.get('AUPRC', 0.0):.4f}",
                    "C": metrics.get("C", "N/A"),
                    "Held-out antibodies": metrics.get("Held-out antibodies", "N/A"),
                    "Held-out viruses": metrics.get("Held-out viruses", "N/A"),
                    "Excluded pairs": metrics.get("Excluded pairs", "N/A")
                })
            except Exception as e:
                print(f"Failed to read results for {name}: {e}")
        else:
            print(f"Warning: results.json not found for {name}.")
            summary_rows.append({
                "Experiment": name,
                "Train n": "N/A",
                "Test n": "N/A",
                "Test %neutralizing": "N/A",
                "AUROC": "N/A",
                "AUPRC": "N/A",
                "C": "N/A",
                "Held-out antibodies": "N/A",
                "Held-out viruses": "N/A",
                "Excluded pairs": "N/A"
            })

    # Write summary_results.csv at the top level
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = "summary_results.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    
    print("\n=================================================================")
    print("                      SUMMARY RESULTS TABLE                      ")
    print("=================================================================")
    print(summary_df.to_string(index=False))
    print("=================================================================")
    print(f"\nSaved summary results table to {summary_csv_path}\n")

if __name__ == '__main__':
    main()
