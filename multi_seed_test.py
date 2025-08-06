#!/usr/bin/env python3

import argparse
import numpy as np
import json
import os
from graph_regression import graph_regression
from graph_classification import graph_classification

def multi_seed_test(task_type, dataset_name, target_column=None, num_seeds=5, epochs=100, patience=50):
    """
    Test performance with multiple seeds for both with and without global features.
    """
    seeds = [42, 123, 456, 789, 999][:num_seeds]
    
    results = {
        'without_global_features': [],
        'with_global_features': []
    }
    
    print(f"=== Multi-Seed Test: {task_type} on {dataset_name} ===")
    print(f"Testing {num_seeds} seeds: {seeds}")
    print(f"Epochs: {epochs}, Patience: {patience}")
    
    for use_global in [False, True]:
        config_name = "with_global_features" if use_global else "without_global_features"
        print(f"\n--- Testing {config_name} ---")
        
        seed_results = []
        
        for i, seed in enumerate(seeds):
            print(f"\nSeed {i+1}/{num_seeds}: {seed}")
            
            # Create args namespace
            args = argparse.Namespace()
            args.dataset_name = dataset_name
            if target_column:
                args.target_column = target_column
            args.epochs = epochs
            args.patience = patience
            args.lr = 0.004
            args.wd = 0.005
            args.dropout = 0.1
            args.hidden_channels = 32
            args.layers = 2
            args.num_grids = 6
            args.batch_size = 64
            args.grid_min = 0
            args.grid_max = 1
            args.log_freq = epochs // 10
            args.use_global_features = use_global
            
            # Additional args for classification
            if task_type == "classification":
                args.use_weighted_loss = True
                args.use_roc_auc = True
                args.gamma = 1.0
                args.return_history = False
            
            # Override seed in utils
            import sys
            sys.path.append('src')
            from utils import set_seed
            set_seed(seed)
            
            try:
                if task_type == "classification":
                    result = graph_classification(args)
                    metric_name = "ROC-AUC" if args.use_roc_auc else "Accuracy"
                else:
                    result = graph_regression(args)
                    metric_name = "MAE"
                
                seed_results.append({
                    'seed': seed,
                    'result': result,
                    'metric': metric_name
                })
                print(f"Seed {seed} - {metric_name}: {result:.4f}")
                
            except Exception as e:
                print(f"Error with seed {seed}: {e}")
                seed_results.append({
                    'seed': seed,
                    'result': None,
                    'metric': metric_name,
                    'error': str(e)
                })
        
        results[config_name] = seed_results
    
    # Analyze results
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for config_name, seed_results in results.items():
        valid_results = [r['result'] for r in seed_results if r['result'] is not None]
        
        if valid_results:
            mean_val = np.mean(valid_results)
            std_val = np.std(valid_results)
            min_val = np.min(valid_results)
            max_val = np.max(valid_results)
            
            print(f"\n{config_name.replace('_', ' ').title()}:")
            print(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
            print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
            print(f"  Individual results: {[f'{v:.4f}' for v in valid_results]}")
        else:
            print(f"\n{config_name.replace('_', ' ').title()}: No valid results")
    
    # Statistical comparison
    without_results = [r['result'] for r in results['without_global_features'] if r['result'] is not None]
    with_results = [r['result'] for r in results['with_global_features'] if r['result'] is not None]
    
    if without_results and with_results:
        print(f"\nSTATISTICAL COMPARISON:")
        
        without_mean = np.mean(without_results)
        with_mean = np.mean(with_results)
        
        if task_type == "regression":
            # For regression (MAE), lower is better
            improvement = ((without_mean - with_mean) / without_mean) * 100
            better_config = "with global features" if with_mean < without_mean else "without global features"
        else:
            # For classification (ROC-AUC/Accuracy), higher is better
            improvement = ((with_mean - without_mean) / without_mean) * 100
            better_config = "with global features" if with_mean > without_mean else "without global features"
        
        print(f"  Better configuration: {better_config}")
        print(f"  Performance difference: {abs(improvement):.2f}%")
        
        # Paired t-test (since we use the same seeds for both configurations)
        try:
            from scipy.stats import ttest_rel
            t_stat, p_value = ttest_rel(without_results, with_results)
            print(f"  Paired t-test p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  Difference is statistically significant (p < 0.05)")
            else:
                print(f"  Difference is not statistically significant (p >= 0.05)")
            
            # Effect size (Cohen's d for paired samples)
            differences = np.array(with_results) - np.array(without_results)
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)  # Use sample std
            if std_diff > 0:
                cohens_d = mean_diff / std_diff
                print(f"  Cohen's d (effect size): {cohens_d:.4f}")
                # Interpret effect size
                if abs(cohens_d) < 0.2:
                    effect_interpretation = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect_interpretation = "small"
                elif abs(cohens_d) < 0.8:
                    effect_interpretation = "medium"
                else:
                    effect_interpretation = "large"
                print(f"  Effect size interpretation: {effect_interpretation}")
            else:
                print(f"  Cohen's d: Cannot compute (zero variance)")
                
        except ImportError:
            print("  (scipy not available for paired t-test)")
        except Exception as e:
            print(f"  (could not perform paired t-test: {e})")
    
    # Save results
    output_dir = "experiments/multi_seed_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/{task_type}_{dataset_name}"
    if target_column:
        output_file += f"_{target_column}"
    output_file += f"_{num_seeds}seeds.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Multi-seed performance comparison")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--target_column", type=str, default=None)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    
    args = parser.parse_args()
    
    multi_seed_test(
        task_type=args.task,
        dataset_name=args.dataset_name,
        target_column=args.target_column,
        num_seeds=args.num_seeds,
        epochs=args.epochs,
        patience=args.patience
    )

if __name__ == "__main__":
    main()
