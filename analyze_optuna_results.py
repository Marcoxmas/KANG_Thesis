#!/usr/bin/env python3

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

def analyze_optuna_results(results_dir="experiments/optuna_search", training_results_dir="experiments/training_results"):
    """
    Analyze Optuna hyperparameter search results and training results.
    Creates simple summary of all runs performed.
    """
    
    print("=" * 80)
    print("OPTUNA RESULTS ANALYZER")
    print("=" * 80)
    
    # Find Optuna result files (legacy format)
    optuna_results = []
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if f.startswith("best_params_") and f.endswith(".json")]
        
        for result_file in result_files:
            try:
                with open(os.path.join(results_dir, result_file), 'r') as f:
                    result_data = json.load(f)
                    # Convert no_self_loops to use_self_loops for consistency
                    if "no_self_loops" in result_data:
                        result_data["use_self_loops"] = not result_data["no_self_loops"]
                    elif "use_self_loops" not in result_data:
                        # Default to None/unknown for older optuna results
                        result_data["use_self_loops"] = None
                    
                    # Parse head type from filename - only for multitask
                    if "singlehead" in result_file:
                        result_data["head_type"] = "singlehead"
                    elif "multihead" in result_file:
                        result_data["head_type"] = "multihead"
                    else:
                        # For legacy files, check the multitask status and single_head parameter
                        is_multitask = result_data.get("multitask", False)
                        if is_multitask:
                            result_data["head_type"] = "singlehead" if result_data.get("single_head", False) else "multihead"
                        else:
                            # Single-task doesn't have head type distinction
                            result_data["head_type"] = "N/A"
                    
                    result_data["_source"] = "optuna"
                    result_data["_file"] = result_file
                    optuna_results.append(result_data)
            except Exception as e:
                print(f"Warning: Could not load Optuna result {result_file}: {e}")
    
    # Find training result files (new format with seed tracking)
    training_results = []
    if os.path.exists(training_results_dir):
        training_files = [f for f in os.listdir(training_results_dir) if f.startswith("training_result_") and f.endswith(".json")]
        for training_file in training_files:
            try:
                with open(os.path.join(training_results_dir, training_file), 'r') as f:
                    training_data = json.load(f)
                    # Parse head type from filename - only for multitask
                    if "singlehead" in training_file:
                        head_type = "singlehead"
                    elif "multihead" in training_file:
                        head_type = "multihead"
                    else:
                        # For legacy files or single-task, check the single_head parameter and multitask status
                        is_multitask = training_data["training_info"]["multitask"]
                        if is_multitask:
                            # For multitask, determine head type from parameter if available
                            head_type = "singlehead" if training_data.get("single_head", False) else "multihead"
                        else:
                            # Single-task doesn't have head type distinction
                            head_type = "N/A"
                    # Convert training result format to match optuna format for analysis
                    converted_data = {
                        "task_type": training_data["training_info"]["task_type"],
                        "dataset_name": training_data["training_info"]["dataset_name"],
                        "target_column": training_data["training_info"]["target_column"],
                        "multitask": training_data["training_info"]["multitask"],
                        "multitask_assays": training_data["training_info"]["multitask_assays"],
                        "multitask_targets": training_data["training_info"]["multitask_targets"],
                        "use_global_features": training_data["training_info"]["use_global_features"],
                        "use_self_loops": training_data["training_info"]["use_self_loops"],
                        "head_type": head_type,
                        "final_test_metric": training_data["results"]["final_test_metric"],
                        "validation_score": training_data["results"].get("best_validation_accuracy") or training_data["results"].get("best_validation_score"),
                        "seed": training_data["training_info"]["seed"],
                        "timestamp": training_data["training_info"]["timestamp"],
                        "_source": "training",
                        "_file": training_file
                    }
                    training_results.append(converted_data)
            except Exception as e:
                print(f"Warning: Could not load training result {training_file}: {e}")
    
    # Combine all results
    all_results = optuna_results + training_results
    
    if not all_results:
        print("No result files found!")
        print(f"Checked directories:")
        print(f"  Optuna results: {results_dir}")
        print(f"  Training results: {training_results_dir}")
        return
    
    print(f"Found {len(optuna_results)} Optuna results and {len(training_results)} training results")
    print(f"Total results: {len(all_results)}")
    print("")
    
    # Create DataFrame for easy handling
    results_summary = []
    
    for result in all_results:
        # Handle multi-task information
        multitask = result.get("multitask", False)
        multitask_info = ""
        head_type = result.get("head_type", "unknown")
        if multitask:
            if result.get("multitask_assays"):
                multitask_info = f"MT({len(result['multitask_assays'])} assays)"
            elif result.get("multitask_targets"):
                multitask_info = f"MT({len(result['multitask_targets'])} targets)"
            else:
                multitask_info = "MT(default)"
        results_summary.append({
            "task_type": result.get("task_type"),
            "dataset_name": result.get("dataset_name"),
            "target_column": result.get("target_column"),
            "multitask": multitask,
            "multitask_info": multitask_info,
            "multitask_assays": result.get("multitask_assays"),
            "multitask_targets": result.get("multitask_targets"),
            "use_global_features": result.get("use_global_features"),
            "use_self_loops": result.get("use_self_loops"),
            "head_type": head_type,
            "final_test_metric": result.get("final_test_metric"),
            "validation_score": result.get("validation_score"),
            "n_trials": result.get("n_trials"),
            "seed": result.get("seed"),
            "source": result.get("_source", "optuna"),
            "timestamp": result.get("timestamp"),
            "filename": result.get("_file")
        })
    
    df_results = pd.DataFrame(results_summary)
    
    # Save CSV file - use the results_dir (optuna directory) for backwards compatibility
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(results_dir, f"combined_results_summary_{timestamp_str}.csv")
    df_results.to_csv(results_csv, index=False)
    
    # Create summary report
    summary_report = []
    summary_report.append("=" * 80)
    summary_report.append("COMBINED RESULTS SUMMARY (OPTUNA + TRAINING)")
    summary_report.append("=" * 80)
    summary_report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_report.append(f"Total results analyzed: {len(all_results)}")
    summary_report.append(f"  - Optuna hyperparameter search results: {len(optuna_results)}")
    summary_report.append(f"  - Training results with seed tracking: {len(training_results)}")
    summary_report.append("")
    
    # Overall statistics
    summary_report.append("OVERALL STATISTICS:")
    summary_report.append("-" * 40)
    summary_report.append(f"Total experiments completed: {len(df_results)}")
    summary_report.append(f"Unique datasets: {df_results['dataset_name'].nunique()}")
    summary_report.append(f"Regression experiments: {len(df_results[df_results['task_type'] == 'regression'])}")
    summary_report.append(f"Classification experiments: {len(df_results[df_results['task_type'] == 'classification'])}")
    summary_report.append(f"Multi-task experiments: {len(df_results[df_results['multitask'] == True])}")
    summary_report.append(f"Single-task experiments: {len(df_results[df_results['multitask'] == False])}")
    summary_report.append(f"With global features: {len(df_results[df_results['use_global_features'] == True])}")
    summary_report.append(f"Without global features: {len(df_results[df_results['use_global_features'] == False])}")
    
    # Head type statistics (only for multitask experiments)
    multitask_df = df_results[df_results['multitask'] == True]
    if len(multitask_df) > 0:
        singlehead_count = len(multitask_df[multitask_df['head_type'] == 'singlehead'])
        multihead_count = len(multitask_df[multitask_df['head_type'] == 'multihead'])
        unknown_head_count = len(multitask_df[~multitask_df['head_type'].isin(['singlehead', 'multihead'])])
        
        if singlehead_count > 0 or multihead_count > 0:
            summary_report.append(f"Multitask single head experiments: {singlehead_count}")
            summary_report.append(f"Multitask multi head experiments: {multihead_count}")
            if unknown_head_count > 0:
                summary_report.append(f"Multitask unknown head type: {unknown_head_count}")
    
    # Self loops statistics (where available) - treat unknown as with self loops
    self_loops_true = len(df_results[df_results['use_self_loops'] == True])
    self_loops_false = len(df_results[df_results['use_self_loops'] == False])
    self_loops_unknown = len(df_results[df_results['use_self_loops'].isna()])
    
    # Combine true and unknown as "with self loops"
    self_loops_with = self_loops_true + self_loops_unknown
    
    if self_loops_with > 0 or self_loops_false > 0:
        summary_report.append(f"With self loops (incl. unknown): {self_loops_with}")
        summary_report.append(f"Without self loops: {self_loops_false}")
        if self_loops_unknown > 0:
            summary_report.append(f"  (of which {self_loops_unknown} had unknown self loops status)")
    
    # Seed statistics (only for training results)
    training_df = df_results[df_results['source'] == 'training']
    if len(training_df) > 0:
        unique_seeds = training_df['seed'].dropna().nunique()
        summary_report.append(f"Training experiments with seed tracking: {len(training_df)}")
        summary_report.append(f"Unique seeds used: {unique_seeds}")
    
    summary_report.append("")
    
    # Dataset averages
    summary_report.append("DATASET AVERAGES:")
    summary_report.append("-" * 40)
    
    for dataset in sorted(df_results['dataset_name'].unique()):
        dataset_data = df_results[df_results['dataset_name'] == dataset]
        summary_report.append(f"\n{dataset}:")
        
        # Group by global features, self loops, and task type
        for use_global in [True, False]:
            global_data = dataset_data[dataset_data['use_global_features'] == use_global]
            if len(global_data) == 0:
                continue
                
            global_str = "with global" if use_global else "without global"
            
            # Group by self loops (treat unknown/None as True)
            for use_self_loops in [True, False]:
                    if use_self_loops:
                        # Include both explicit True and unknown/None cases
                        self_loops_data = global_data[(global_data['use_self_loops'] == True) | (global_data['use_self_loops'].isna())]
                        self_loops_str = "with loops"
                    else:
                        # Only explicit False cases
                        self_loops_data = global_data[global_data['use_self_loops'] == False]
                        self_loops_str = "no loops"
                    
                    if len(self_loops_data) == 0:
                        continue
                    
                    # Group by head type - include N/A for single-task experiments
                    for head_type in ['N/A', 'singlehead', 'multihead']:
                        head_data = self_loops_data[self_loops_data['head_type'] == head_type]
                        if len(head_data) == 0:
                            continue
                        
                        # Separate by task type and multi-task status
                        for task_type in ['regression', 'classification']:
                            task_data = head_data[head_data['task_type'] == task_type]
                            if len(task_data) == 0:
                                continue
                            
                            # Separate single-task vs multi-task
                            for is_multitask in [False, True]:
                                mt_data = task_data[task_data['multitask'] == is_multitask]
                                if len(mt_data) == 0:
                                    continue
                                
                                mt_str = "multi-task" if is_multitask else "single-task"
                                
                                # Calculate statistics
                                valid_metrics = mt_data['final_test_metric'].dropna()
                                if len(valid_metrics) > 0:
                                    avg_metric = valid_metrics.mean()
                                    std_metric = valid_metrics.std() if len(valid_metrics) > 1 else 0
                                    min_metric = valid_metrics.min()
                                    max_metric = valid_metrics.max()
                                    
                                    summary_report.append(f"  {task_type:12s} {mt_str:10s} {global_str:15s} {self_loops_str:10s} {head_type:10s}: "
                                                        f"avg={avg_metric:.4f} ±{std_metric:.4f} "
                                                        f"(min={min_metric:.4f}, max={max_metric:.4f}, n={len(valid_metrics)})")    
    # Results by dataset
    summary_report.append("\nDETAILED RESULTS BY DATASET:")
    summary_report.append("-" * 40)
    
    for dataset in sorted(df_results['dataset_name'].unique()):
        dataset_data = df_results[df_results['dataset_name'] == dataset]
        summary_report.append(f"\n{dataset}:")
        
        for _, row in dataset_data.iterrows():
            # Create target description
            if row['multitask']:
                if row['multitask_assays']:
                    target_desc = f"MT-{len(row['multitask_assays'])}assays"
                elif row['multitask_targets']:
                    target_desc = f"MT-{len(row['multitask_targets'])}targets"
                else:
                    target_desc = "MT-default"
                # Add head type for multitask
                head_desc = f"({row['head_type']})" if row['head_type'] not in ['N/A', 'unknown'] else ""
                target_desc += head_desc
            else:
                target_desc = row['target_column'] if row['target_column'] else "N/A"
            
            global_str = "with global" if row['use_global_features'] else "without global"
            self_loops_str = ""
            if pd.notna(row['use_self_loops']):
                self_loops_str = "with loops" if row['use_self_loops'] else "no loops"
            else:
                self_loops_str = "with loops"  # Treat unknown as with loops
            
            head_type_str = row['head_type'] if row['head_type'] != 'unknown' else 'multi-head'
            
            metric = row['final_test_metric']
            metric_str = f"{metric:.4f}" if metric is not None else "FAILED"
            
            # Add seed and source information
            source_str = f"[{row['source']}]"
            seed_str = f"seed={row['seed']}" if row['seed'] is not None else "no-seed"
            
            summary_report.append(f"  {row['task_type']:12s} {target_desc:20s} {global_str:15s} {self_loops_str:12s} {head_type_str:10s} {source_str:9s} {seed_str:10s}: {metric_str}")
    
    # Add seed analysis for training results
    if len(training_df) > 0:
        summary_report.append("\nSEED ANALYSIS (TRAINING RESULTS ONLY):")
        summary_report.append("-" * 40)
        
        # Group by configuration (everything except seed) to show seed variations
        config_cols = ['dataset_name', 'task_type', 'target_column', 'multitask', 'use_global_features', 'use_self_loops', 'head_type']
        
        for _, group in training_df.groupby(config_cols):
            if len(group) > 1:  # Only show configurations with multiple seeds
                config_desc = f"{group.iloc[0]['dataset_name']} {group.iloc[0]['task_type']}"
                if group.iloc[0]['target_column']:
                    config_desc += f" {group.iloc[0]['target_column']}"
                if group.iloc[0]['multitask']:
                    config_desc += " multitask"
                config_desc += " with-global" if group.iloc[0]['use_global_features'] else " without-global"
                
                # Add self loops information
                if pd.notna(group.iloc[0]['use_self_loops']):
                    config_desc += " with-loops" if group.iloc[0]['use_self_loops'] else " no-loops"
                
                # Add head type information
                head_type = group.iloc[0]['head_type']
                if head_type != 'unknown':
                    config_desc += f" {head_type}"
                
                summary_report.append(f"\n{config_desc}:")
                
                # Calculate seed statistics
                valid_metrics = group['final_test_metric'].dropna()
                if len(valid_metrics) > 0:
                    mean_metric = valid_metrics.mean()
                    std_metric = valid_metrics.std() if len(valid_metrics) > 1 else 0
                    summary_report.append(f"  Mean: {mean_metric:.4f} ± {std_metric:.4f} ({len(valid_metrics)} seeds)")
                    
                    # Show individual seed results
                    for _, seed_result in group.iterrows():
                        seed_metric = seed_result['final_test_metric']
                        seed_metric_str = f"{seed_metric:.4f}" if seed_metric is not None else "FAILED"
                        summary_report.append(f"    Seed {seed_result['seed']}: {seed_metric_str}")
    
    # Save summary report
    summary_file = os.path.join(results_dir, f"combined_summary_{timestamp_str}.txt")
    with open(summary_file, 'w') as f:
        f.write("\n".join(summary_report))
    
    # Print summary
    print("\n".join(summary_report))
    
    print(f"\nFiles generated:")
    print(f"  Results CSV: {results_csv}")
    print(f"  Summary report: {summary_file}")
    
    return {
        "results_df": df_results,
        "files_created": [results_csv, summary_file],
        "optuna_results_count": len(optuna_results),
        "training_results_count": len(training_results)
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna hyperparameter search results and training results")
    parser.add_argument("--results_dir", type=str, default="experiments/optuna_search", 
                       help="Directory containing Optuna results (default: experiments/optuna_search)")
    parser.add_argument("--training_results_dir", type=str, default="experiments/training_results", 
                       help="Directory containing training results with seed tracking (default: experiments/training_results)")

    args = parser.parse_args()
    
    # Analyze results
    analyze_optuna_results(args.results_dir, args.training_results_dir)

if __name__ == "__main__":
    main()
