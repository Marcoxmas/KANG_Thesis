import argparse
import json
import os
from pathlib import Path
import torch
from tqdm.auto import tqdm
from datetime import datetime

# Import your existing training functions
from graph_classification import graph_classification
from graph_regression import graph_regression
from plotting_utils import plot_training_metrics
from src.utils import set_seed

def create_result_filename(dataset_name, target_column, multitask, multitask_assays, multitask_targets, use_global_features, task_type, seed):
    """Create a result filename with seed tracking."""
    results_dir = Path("experiments/training_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create base filename
    result_file = f"training_result_{task_type}_{dataset_name}"
    
    if multitask:
        if multitask_assays:
            # Create a short hash for assays
            assay_str = " ".join(sorted(multitask_assays))
            assay_hash = str(sum(ord(c) for c in assay_str) % 10**8)
            result_file += f"_multitask_{assay_hash}"
        elif multitask_targets:
            # Create a short hash for targets
            target_str = " ".join(sorted(multitask_targets))
            target_hash = str(sum(ord(c) for c in target_str) % 10**8)
            result_file += f"_multitask_{target_hash}"
        else:
            result_file += "_multitask_default"
    elif target_column:
        result_file += f"_{target_column}"
    
    result_file += f"_global_{use_global_features}_seed_{seed}.json"
    
    return results_dir / result_file

def save_training_results(results, args, seed, best_params_file, final_test_metric):
    """Save training results to a JSON file with comprehensive information."""
    
    # Determine multitask assays/targets from args
    multitask_assays = getattr(args, 'multitask_assays', None)
    multitask_targets = getattr(args, 'multitask_targets', None)
    
    # Create result filename
    task_type = "classification" if args.dataset_name in ["HIV", "TOXCAST"] else "regression"
    result_file = create_result_filename(
        args.dataset_name,
        getattr(args, 'target_column', None),
        getattr(args, 'multitask', False),
        multitask_assays,
        multitask_targets,
        args.use_global_features,
        task_type,
        seed
    )
    
    # Create comprehensive result data
    result_data = {
        "training_info": {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "task_type": task_type,
            "dataset_name": args.dataset_name,
            "target_column": getattr(args, 'target_column', None),
            "multitask": getattr(args, 'multitask', False),
            "multitask_assays": multitask_assays,
            "multitask_targets": multitask_targets,
            "use_global_features": args.use_global_features,
            "use_self_loops": args.use_self_loops,
            "epochs_run": args.epochs,
            "patience": args.patience,
            "best_params_file": str(best_params_file)
        },
        "hyperparameters": {
            "lr": args.lr,
            "wd": getattr(args, 'wd', None),
            "hidden_channels": args.hidden_channels,
            "layers": args.layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "num_grids": getattr(args, 'num_grids', None),
            "gamma": getattr(args, 'gamma', None),
            "grid_min": args.grid_min,
            "grid_max": args.grid_max
        },
        "results": {
            "final_test_metric": final_test_metric
        }
    }
    
    # Add training history if available
    if isinstance(results, dict):
        result_data["results"].update(results)
    
    # Save results
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=4)
    
    print(f"\nTraining results saved to: {result_file}")
    return result_file

def find_best_params_file(dataset_name, target_column, multitask, use_global_features, task_type):
    """Find the best parameters file based on the given criteria."""
    optuna_dir = Path("experiments/optuna_search")
    
    # Replicate the exact filename creation logic from optuna_search_main.py
    param_file = f"best_params_{task_type}_{dataset_name}"
    
    if multitask:
        # For multitask, we need to find the file with the hash
        # Since we don't know the exact targets that were used, we'll search for pattern
        pattern = f"best_params_{task_type}_{dataset_name}_multitask_*_global_{use_global_features}.json"
        matching_files = list(optuna_dir.glob(pattern))
        
        if matching_files:
            # If multiple files found, return the first one (could be improved with selection logic)
            return matching_files[0]
        
        # Try alternative patterns for multitask
        alt_patterns = [
            f"best_params_{task_type}_{dataset_name}_multitask_default_global_{use_global_features}.json",
        ]
        
        for pattern in alt_patterns:
            filepath = optuna_dir / pattern
            if filepath.exists():
                return filepath
                
    else:
        # For single task - exact match with the optuna_search_main.py logic
        if target_column:
            param_file += f"_{target_column}"
        param_file += f"_global_{use_global_features}.json"
        
        filepath = optuna_dir / param_file
        if filepath.exists():
            return filepath
    
    # If no exact match found, list available files for debugging
    print(f"Could not find matching file for:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Task type: {task_type}")
    print(f"  Multitask: {multitask}")
    print(f"  Target column: {target_column}")
    print(f"  Use global features: {use_global_features}")
    print(f"  Expected pattern: {pattern if multitask else param_file}")
    print("\nAvailable files in optuna_search directory:")
    
    # Show relevant files
    relevant_files = []
    for f in optuna_dir.glob("best_params_*.json"):
        if dataset_name.lower() in f.name.lower() and task_type in f.name:
            relevant_files.append(f.name)
    
    if relevant_files:
        print("Relevant files found:")
        for f in sorted(relevant_files):
            print(f"  {f}")
    else:
        print("No relevant files found. All best_params files:")
        for f in sorted(optuna_dir.glob("best_params_*.json")):
            print(f"  {f.name}")
    
    return None

def load_best_params(filepath):
    """Load best parameters from JSON file."""
    with open(filepath, 'r') as f:
        params = json.load(f)
    return params

def create_args_from_params(params, use_self_loops, epochs, patience, task_type):
    """Create an arguments object from the loaded parameters."""
    class Args:
        pass
    
    args = Args()
    
    # Set parameters from the loaded best params
    for key, value in params.items():
        setattr(args, key, value)
    
    # Add the self_loops parameter
    args.use_self_loops = use_self_loops
    
    # Override with custom training parameters
    args.epochs = epochs
    args.patience = patience
    args.log_freq = max(1, epochs // 10)  # Ensure log_freq is at least 1
    
    # Set grid parameters based on task type
    if task_type == "classification":
        args.grid_min = -1
        args.grid_max = 2.5
    else:  # regression
        args.grid_min = -1
        args.grid_max = 1.5
    
    # Set default values for parameters that might not be in the JSON
    if not hasattr(args, 'return_history'):
        args.return_history = True
    if not hasattr(args, 'use_weighted_loss'):
        args.use_weighted_loss = False
    
    # Always use ROC-AUC for classification tasks
    args.use_roc_auc = (task_type == 'classification')
    
    # Always use the full dataset, override any subset settings from hyperparameter search
    args.use_subset = False
    args.subset_ratio = 1.0
    
    return args

def main():
    parser = argparse.ArgumentParser(description="Train model with best Optuna hyperparameters")
    parser.add_argument("--dataset_name", type=str, required=True, 
                       help="Dataset name (e.g., HIV, QM9, QM8, TOXCAST)")
    parser.add_argument("--target_column", type=str, default=None,
                       help="Target column (required for single-task, ignored for multitask)")
    parser.add_argument("--multitask", action="store_true",
                       help="Use multitask learning")
    parser.add_argument("--use_global_features", action="store_true",
                       help="Use global molecular features")
    parser.add_argument("--no_self_loops", action="store_true",
                       help="Disable self loops in the GNN (default: use self loops)")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of training epochs (default: 200)")
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    cmd_args = parser.parse_args()
    
    # Set random seed
    set_seed(cmd_args.seed)
    
    # Determine task type based on dataset
    if cmd_args.dataset_name in ["HIV", "TOXCAST"]:
        task_type = "classification"
    elif cmd_args.dataset_name in ["QM9", "QM8"]:
        task_type = "regression"
    else:
        raise ValueError(f"Unknown dataset: {cmd_args.dataset_name}")
    
    # Validate arguments
    if not cmd_args.multitask and cmd_args.target_column is None:
        raise ValueError("target_column is required when multitask=False")
    
    # Find the best parameters file
    best_params_file = find_best_params_file(
        cmd_args.dataset_name,
        cmd_args.target_column,
        cmd_args.multitask,
        cmd_args.use_global_features,
        task_type
    )
    
    if best_params_file is None or not best_params_file.exists():
        print(f"Error: Could not find best parameters file for the given configuration.")
        return
    
    print(f"Loading best parameters from: {best_params_file}")
    
    # Load best parameters
    best_params = load_best_params(best_params_file)
    print(f"Best parameters loaded:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Create arguments object
    args = create_args_from_params(best_params, not cmd_args.no_self_loops, cmd_args.epochs, cmd_args.patience, task_type)
    
    # Override with command line arguments
    args.use_self_loops = not cmd_args.no_self_loops  # Default True, False only if --no_self_loops is used
    
    print(f"\nTraining configuration:")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Task type: {task_type}")
    print(f"  Multitask: {args.multitask}")
    if not args.multitask:
        print(f"  Target column: {args.target_column}")
    print(f"  Use global features: {args.use_global_features}")
    print(f"  Use self loops: {args.use_self_loops}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Patience: {args.patience}")
    print(f"  Log frequency: {args.log_freq}")
    print(f"  Grid range: [{args.grid_min}, {args.grid_max}]")
    print(f"  Use ROC-AUC: {args.use_roc_auc}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Hidden channels: {args.hidden_channels}")
    print(f"  Layers: {args.layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Batch size: {args.batch_size}")
    
    # Train the model
    print(f"\nStarting training...")
    
    try:
        if task_type == "classification":
            results = graph_classification(args, return_history=True)
            if isinstance(results, tuple) and len(results) >= 4:
                best_val_acc, train_losses, val_metrics, final_test_metric = results
                # Create training plots
                plot_dataset_name = f"{args.dataset_name}_multitask" if args.multitask else args.dataset_name
                try:
                    plot_training_metrics(train_losses, val_metrics, task_type, plot_dataset_name)
                except Exception as e:
                    print(f"Warning: Could not generate training plots: {e}")
                
                # Prepare results dict
                results_dict = {
                    "best_validation_accuracy": best_val_acc,
                    "train_losses": train_losses,
                    "val_metrics": val_metrics
                }
            else:
                # Handle case where return_history=False or different return format
                final_test_metric = results if isinstance(results, (int, float)) else None
                results_dict = {}
                
        else:  # regression
            results = graph_regression(args, return_history=True)
            if isinstance(results, tuple) and len(results) >= 4:
                best_val_score, train_losses, val_metrics, final_test_metric = results
                # Create training plots
                plot_dataset_name = f"{args.dataset_name}_multitask" if args.multitask else args.dataset_name
                try:
                    plot_training_metrics(train_losses, val_metrics, task_type, plot_dataset_name, getattr(args, 'target_column', None))
                except Exception as e:
                    print(f"Warning: Could not generate training plots: {e}")
                
                # Prepare results dict
                results_dict = {
                    "best_validation_score": best_val_score,
                    "train_losses": train_losses,
                    "val_metrics": val_metrics
                }
            else:
                # Handle case where return_history=False or different return format
                final_test_metric = results if isinstance(results, (int, float)) else None
                results_dict = {}
        
        print(f"\nTraining completed successfully!")
        
        # Save training results with seed tracking
        result_file = save_training_results(results_dict, args, cmd_args.seed, best_params_file, final_test_metric)
        
        # Print final results
        print(f"Final results:")
        print(f"  Final test metric: {final_test_metric:.4f}" if final_test_metric is not None else "  Final test metric: N/A")
        print(f"  Seed used: {cmd_args.seed}")
        print(f"  Results saved to: {result_file}")
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
