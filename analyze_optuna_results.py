#!/usr/bin/env python3

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

def analyze_optuna_results(results_dir="experiments/optuna_search"):
    """
    Analyze Optuna hyperparameter search results.
    Creates simple summary of all runs performed.
    """
    
    print("=" * 80)
    print("OPTUNA RESULTS ANALYZER")
    print("=" * 80)
    
    # Find all individual result files
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    result_files = [f for f in os.listdir(results_dir) if f.startswith("result_") and f.endswith(".json")]
    
    if not result_files:
        print("No individual Optuna result files found!")
        print(f"Looking in: {results_dir}")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Load all results
    all_results = []
    failed_loads = 0
    
    for result_file in result_files:
        try:
            with open(os.path.join(results_dir, result_file), 'r') as f:
                result_data = json.load(f)
                all_results.append(result_data)
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")
            failed_loads += 1
    
    if not all_results:
        print("No valid result files could be loaded!")
        return
    
    print(f"Successfully loaded {len(all_results)} results")
    if failed_loads > 0:
        print(f"Failed to load {failed_loads} files")
    print("")
    
    # Create DataFrame for easy handling
    results_summary = []
    
    for result in all_results:
        results_summary.append({
            "task_type": result.get("task_type"),
            "dataset_name": result.get("dataset_name"),
            "target_column": result.get("target_column"),
            "use_global_features": result.get("use_global_features"),
            "final_test_metric": result.get("final_test_metric"),
            "validation_score": result.get("validation_score"),
            "n_trials": result.get("n_trials")
        })
    
    df_results = pd.DataFrame(results_summary)
    
    # Save CSV file
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(results_dir, f"optuna_results_summary_{timestamp_str}.csv")
    df_results.to_csv(results_csv, index=False)
    
    # Create summary report
    summary_report = []
    summary_report.append("=" * 80)
    summary_report.append("OPTUNA HYPERPARAMETER SEARCH - RESULTS SUMMARY")
    summary_report.append("=" * 80)
    summary_report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_report.append(f"Total results analyzed: {len(all_results)}")
    summary_report.append("")
    
    # Overall statistics
    summary_report.append("OVERALL STATISTICS:")
    summary_report.append("-" * 40)
    summary_report.append(f"Total experiments completed: {len(df_results)}")
    summary_report.append(f"Unique datasets: {df_results['dataset_name'].nunique()}")
    summary_report.append(f"Regression experiments: {len(df_results[df_results['task_type'] == 'regression'])}")
    summary_report.append(f"Classification experiments: {len(df_results[df_results['task_type'] == 'classification'])}")
    summary_report.append(f"With global features: {len(df_results[df_results['use_global_features'] == True])}")
    summary_report.append(f"Without global features: {len(df_results[df_results['use_global_features'] == False])}")
    summary_report.append("")
    
    # Dataset averages
    summary_report.append("DATASET AVERAGES:")
    summary_report.append("-" * 40)
    
    for dataset in sorted(df_results['dataset_name'].unique()):
        dataset_data = df_results[df_results['dataset_name'] == dataset]
        summary_report.append(f"\n{dataset}:")
        
        # Group by global features and task type
        for use_global in [True, False]:
            global_data = dataset_data[dataset_data['use_global_features'] == use_global]
            if len(global_data) == 0:
                continue
                
            global_str = "with global" if use_global else "without global"
            
            # Separate by task type
            for task_type in ['regression', 'classification']:
                task_data = global_data[global_data['task_type'] == task_type]
                if len(task_data) == 0:
                    continue
                
                # Calculate statistics
                valid_metrics = task_data['final_test_metric'].dropna()
                if len(valid_metrics) > 0:
                    avg_metric = valid_metrics.mean()
                    std_metric = valid_metrics.std() if len(valid_metrics) > 1 else 0
                    min_metric = valid_metrics.min()
                    max_metric = valid_metrics.max()
                    
                    summary_report.append(f"  {task_type:12s} {global_str:15s}: "
                                        f"avg={avg_metric:.4f} ±{std_metric:.4f} "
                                        f"(min={min_metric:.4f}, max={max_metric:.4f}, n={len(valid_metrics)})")
    
    # Results by dataset
    summary_report.append("\nDETAILED RESULTS BY DATASET:")
    summary_report.append("-" * 40)
    
    for dataset in sorted(df_results['dataset_name'].unique()):
        dataset_data = df_results[df_results['dataset_name'] == dataset]
        summary_report.append(f"\n{dataset}:")
        
        for _, row in dataset_data.iterrows():
            target_str = f" ({row['target_column']})" if row['target_column'] else ""
            global_str = "with global" if row['use_global_features'] else "without global"
            metric = row['final_test_metric']
            metric_str = f"{metric:.4f}" if metric is not None else "FAILED"
            
            summary_report.append(f"  {row['task_type']:12s}{target_str:20s} {global_str:15s}: {metric_str}")
    
    # Save summary report
    summary_file = os.path.join(results_dir, f"optuna_summary_{timestamp_str}.txt")
    with open(summary_file, 'w') as f:
        f.write("\n".join(summary_report))
    
    # Print summary
    print("\n".join(summary_report))
    
    print(f"\nFiles generated:")
    print(f"  Results CSV: {results_csv}")
    print(f"  Summary report: {summary_file}")
    
    return {
        "results_df": df_results,
        "files_created": [results_csv, summary_file]
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna hyperparameter search results")
    parser.add_argument("--results_dir", type=str, default="experiments/optuna_summaries", 
                       help="Directory containing Optuna results (default: experiments/optuna_summaries)")
    
    args = parser.parse_args()
    
    # Analyze results
    analyze_optuna_results(args.results_dir)

if __name__ == "__main__":
    main()
