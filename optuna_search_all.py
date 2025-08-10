#!/usr/bin/env python3

import argparse
import numpy as np
import json
import os
import sys
from datetime import datetime
from optuna_search_main import optuna_search

def run_all_optuna_searches(n_trials=20, use_subset=True, subset_ratio=0.3, first_target_only=False):
    """
    Run Optuna hyperparameter search for all datasets and configurations.
    Tests both with and without global features.
    """
    
    print("=" * 80)
    print("KANG THESIS - COMPREHENSIVE OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 80)
    print(f"Starting comprehensive Optuna search across all datasets...")
    print(f"Trials per search: {n_trials}")
    print(f"Use subset during search: {use_subset} ({subset_ratio*100:.0f}% if enabled)")
    if first_target_only:
        print("QUICK TEST MODE: Testing only the first target from each dataset")
    print("")
    
    # Results directory
    results_dir = "experiments/optuna_search"
    summary_dir = "experiments/optuna_summaries"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    
    # Define datasets and targets
    test_configurations = []
    
    # QM8 targets (regression)
    qm8_targets = ["E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"]
    if first_target_only:
        qm8_targets = qm8_targets[:1]  # Just E1-CC2
    
    for target in qm8_targets:
        test_configurations.append(("regression", "QM8", target))
    
    # QM9 targets (regression)
    qm9_targets = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv"]
    if first_target_only:
        qm9_targets = qm9_targets[:1]  # Just mu
        
    for target in qm9_targets:
        test_configurations.append(("regression", "QM9", target))
    
    # HIV (classification)
    test_configurations.append(("classification", "HIV", None))
    
    # ToxCast targets (classification)
    toxcast_assays = [
        "TOX21_AhR_LUC_Agonist",
        "TOX21_Aromatase_Inhibition", 
        "TOX21_AutoFluor_HEK293_Cell_blue",
        "TOX21_p53_BLA_p3_ch1",
        "TOX21_p53_BLA_p4_ratio"
    ]
    if first_target_only:
        toxcast_assays = toxcast_assays[:1]  # Just TOX21_AhR_LUC_Agonist
        
    for assay in toxcast_assays:
        test_configurations.append(("classification", "TOXCAST", assay))
    
    total_tests = len(test_configurations) * 2  # 2 because we test with/without global features
    
    print(f"TOTAL SEARCHES TO PERFORM: {total_tests}")
    print(f"Estimated time: {total_tests * 30} minutes (assuming 30 min per search)")
    print("")
    
    # Track all results
    all_results = {}
    completed_tests = 0
    failed_tests = []
    
    start_time = datetime.now()
    
    for task_type, dataset_name, target_column in test_configurations:
        for use_global_features in [False, True]:
            completed_tests += 1
            global_str = "with_global" if use_global_features else "without_global"
            
            print("=" * 60)
            print(f"SEARCH {completed_tests}/{total_tests}: {task_type} - {dataset_name}")
            if target_column:
                print(f"Target: {target_column}")
            print(f"Global features: {use_global_features}")
            print("=" * 60)
            
            try:
                final_test_metric, best_params = optuna_search(
                    task_type=task_type,
                    dataset_name=dataset_name,
                    target_column=target_column,
                    use_subset=use_subset,
                    subset_ratio=subset_ratio,
                    use_global_features=use_global_features,
                    n_trials=n_trials
                )
                
                # Store result
                key = f"{task_type}_{dataset_name}"
                if target_column:
                    key += f"_{target_column}"
                
                if key not in all_results:
                    all_results[key] = {}
                
                all_results[key][global_str] = {
                    "final_test_metric": final_test_metric,
                    "best_params": best_params,
                    "task_type": task_type,
                    "dataset_name": dataset_name,
                    "target_column": target_column,
                    "use_global_features": use_global_features
                }
                
                print(f"✓ SUCCESS: Final test metric = {final_test_metric}")
                
            except Exception as e:
                print(f"✗ FAILED: {e}")
                failed_tests.append({
                    "task_type": task_type,
                    "dataset_name": dataset_name,
                    "target_column": target_column,
                    "use_global_features": use_global_features,
                    "error": str(e)
                })
            
            print(f"Progress: {completed_tests}/{total_tests} ({completed_tests/total_tests*100:.1f}%)")
            elapsed = datetime.now() - start_time
            if completed_tests > 0:
                avg_time = elapsed.total_seconds() / completed_tests
                remaining_time = avg_time * (total_tests - completed_tests)
                print(f"Elapsed: {elapsed}, Estimated remaining: {remaining_time/60:.1f} minutes")
            print("")
    
    # Generate comprehensive summary
    print("=" * 80)
    print("GENERATING COMPREHENSIVE SUMMARY")
    print("=" * 80)
    
    # Calculate averages per task type
    task_averages = {"regression": {"without_global": [], "with_global": []}, 
                     "classification": {"without_global": [], "with_global": []}}
    
    summary_data = []
    
    for key, results in all_results.items():
        task_type = results.get("without_global", results.get("with_global", {})).get("task_type")
        
        without_metric = results.get("without_global", {}).get("final_test_metric")
        with_metric = results.get("with_global", {}).get("final_test_metric")
        
        summary_row = {
            "experiment": key,
            "task_type": task_type,
            "without_global": without_metric,
            "with_global": with_metric
        }
        
        if without_metric is not None and with_metric is not None:
            if task_type == "regression":
                # For regression (MAE), lower is better
                improvement = ((without_metric - with_metric) / without_metric) * 100
                better = "with_global" if with_metric < without_metric else "without_global"
            else:
                # For classification (accuracy/AUC), higher is better
                improvement = ((with_metric - without_metric) / without_metric) * 100
                better = "with_global" if with_metric > without_metric else "without_global"
            
            summary_row["improvement_pct"] = improvement
            summary_row["better_config"] = better
        
        summary_data.append(summary_row)
        
        # Add to task averages
        if without_metric is not None:
            task_averages[task_type]["without_global"].append(without_metric)
        if with_metric is not None:
            task_averages[task_type]["with_global"].append(with_metric)
    
    # Calculate and print task-level averages
    task_summary = {}
    for task_type in ["regression", "classification"]:
        without_vals = task_averages[task_type]["without_global"]
        with_vals = task_averages[task_type]["with_global"]
        
        if without_vals and with_vals:
            without_avg = np.mean(without_vals)
            with_avg = np.mean(with_vals)
            
            if task_type == "regression":
                overall_improvement = ((without_avg - with_avg) / without_avg) * 100
                better_overall = "with_global" if with_avg < without_avg else "without_global"
            else:
                overall_improvement = ((with_avg - without_avg) / without_avg) * 100
                better_overall = "with_global" if with_avg > without_avg else "without_global"
            
            task_summary[task_type] = {
                "without_global_avg": without_avg,
                "with_global_avg": with_avg,
                "improvement_pct": overall_improvement,
                "better_config": better_overall,
                "num_experiments": min(len(without_vals), len(with_vals))
            }
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    detailed_results = {
        "timestamp": timestamp,
        "n_trials": n_trials,
        "use_subset": use_subset,
        "subset_ratio": subset_ratio,
        "first_target_only": first_target_only,
        "total_tests": total_tests,
        "completed_tests": completed_tests,
        "failed_tests": failed_tests,
        "all_results": all_results,
        "summary_data": summary_data,
        "task_summary": task_summary
    }
    
    results_file = f"{summary_dir}/optuna_comprehensive_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Generate human-readable summary
    summary_text = []
    summary_text.append("=" * 80)
    summary_text.append("OPTUNA HYPERPARAMETER SEARCH - COMPREHENSIVE SUMMARY")
    summary_text.append("=" * 80)
    summary_text.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_text.append(f"Total searches completed: {completed_tests}/{total_tests}")
    summary_text.append(f"Failed searches: {len(failed_tests)}")
    summary_text.append("")
    
    # Task-level summary
    summary_text.append("TASK-LEVEL AVERAGES:")
    summary_text.append("-" * 40)
    
    for task_type, data in task_summary.items():
        summary_text.append(f"\n{task_type.upper()}:")
        summary_text.append(f"  Without global features: {data['without_global_avg']:.4f}")
        summary_text.append(f"  With global features:    {data['with_global_avg']:.4f}")
        summary_text.append(f"  Better configuration:    {data['better_config']}")
        summary_text.append(f"  Improvement:             {abs(data['improvement_pct']):.2f}%")
        summary_text.append(f"  Number of experiments:   {data['num_experiments']}")
    
    # Individual results
    summary_text.append("\n\nINDIVIDUAL RESULTS:")
    summary_text.append("-" * 40)
    
    for row in summary_data:
        summary_text.append(f"\n{row['experiment']}:")
        if row['without_global'] is not None:
            summary_text.append(f"  Without global: {row['without_global']:.4f}")
        if row['with_global'] is not None:
            summary_text.append(f"  With global:    {row['with_global']:.4f}")
        if 'better_config' in row:
            summary_text.append(f"  Better:         {row['better_config']} ({abs(row['improvement_pct']):.2f}% improvement)")
    
    if failed_tests:
        summary_text.append("\n\nFAILED TESTS:")
        summary_text.append("-" * 40)
        for failed in failed_tests:
            summary_text.append(f"  {failed['task_type']} - {failed['dataset_name']}")
            if failed['target_column']:
                summary_text.append(f"    Target: {failed['target_column']}")
            summary_text.append(f"    Global: {failed['use_global_features']}")
            summary_text.append(f"    Error: {failed['error']}")
    
    summary_text_str = "\n".join(summary_text)
    
    summary_file = f"{summary_dir}/optuna_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(summary_text_str)
    
    print(summary_text_str)
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    
    return detailed_results

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Optuna hyperparameter search")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials per search (default: 20)")
    parser.add_argument("--use_subset", action="store_true", help="Use subset during hyperparameter search")
    parser.add_argument("--subset_ratio", type=float, default=0.3, help="Ratio of dataset to use for subset (default: 0.3)")
    parser.add_argument("--first_target_only", action="store_true", help="Test only the first target from each dataset (for quick testing)")
    
    args = parser.parse_args()
    
    run_all_optuna_searches(
        n_trials=args.n_trials,
        use_subset=args.use_subset,
        subset_ratio=args.subset_ratio,
        first_target_only=args.first_target_only
    )

if __name__ == "__main__":
    main()
