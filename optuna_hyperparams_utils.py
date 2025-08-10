#!/usr/bin/env python3

import json
import os
import argparse
from pathlib import Path

def get_best_hyperparameters(task_type, dataset_name, target_column=None, use_global_features=False, 
                           results_dir="experiments/optuna_search"):
    """
    Retrieve the best hyperparameters found by Optuna for a specific configuration.
    
    Returns:
        dict: Best hyperparameters, or None if not found
    """
    
    # Construct the expected filename
    param_file = f"{results_dir}/best_params_{task_type}_{dataset_name}"
    if target_column:
        param_file += f"_{target_column}"
    param_file += f"_global_{use_global_features}.json"
    
    if not os.path.exists(param_file):
        print(f"Hyperparameter file not found: {param_file}")
        return None
    
    try:
        with open(param_file, 'r') as f:
            params = json.load(f)
        return params
    except Exception as e:
        print(f"Error reading hyperparameter file {param_file}: {e}")
        return None

def list_available_hyperparameters(results_dir="experiments/optuna_search"):
    """
    List all available hyperparameter files and their configurations.
    """
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return []
    
    param_files = [f for f in os.listdir(results_dir) if f.startswith("best_params_") and f.endswith(".json")]
    
    if not param_files:
        print("No hyperparameter files found!")
        return []
    
    configurations = []
    
    for param_file in param_files:
        try:
            # Parse filename to extract configuration
            name = param_file.replace("best_params_", "").replace(".json", "")
            parts = name.split("_")
            
            if len(parts) >= 3:
                task_type = parts[0]
                dataset_name = parts[1]
                
                # Find where "global" appears
                global_idx = -1
                for i, part in enumerate(parts):
                    if part == "global":
                        global_idx = i
                        break
                
                if global_idx > 0:
                    target_column = "_".join(parts[2:global_idx]) if global_idx > 2 else None
                    use_global_features = parts[global_idx + 1] == "True"
                    
                    # Load the file to get additional info
                    with open(os.path.join(results_dir, param_file), 'r') as f:
                        params = json.load(f)
                    
                    configurations.append({
                        "task_type": task_type,
                        "dataset_name": dataset_name,
                        "target_column": target_column,
                        "use_global_features": use_global_features,
                        "filename": param_file,
                        "best_value": params.get("best_value"),
                        "n_trials": params.get("n_trials"),
                        "params": params
                    })
        
        except Exception as e:
            print(f"Error parsing {param_file}: {e}")
            continue
    
    return configurations

def create_training_args_from_optuna(task_type, dataset_name, target_column=None, 
                                   use_global_features=False, epochs=200, patience=50,
                                   results_dir="experiments/optuna_search"):
    """
    Create an argparse.Namespace object with the best hyperparameters from Optuna
    for use in training scripts.
    """
    
    import argparse
    
    # Get best hyperparameters
    best_params = get_best_hyperparameters(task_type, dataset_name, target_column, 
                                         use_global_features, results_dir)
    
    if best_params is None:
        print("Could not retrieve best hyperparameters. Using default values.")
        best_params = {}
    
    # Create args namespace with best parameters
    args = argparse.Namespace()
    
    # Basic configuration
    args.dataset_name = dataset_name
    args.target_column = target_column
    args.use_global_features = use_global_features
    args.epochs = epochs
    args.patience = patience
    
    # Hyperparameters from Optuna (with defaults if not found)
    args.lr = best_params.get("lr", 0.004)
    args.wd = best_params.get("wd", 0.005)
    args.hidden_channels = best_params.get("hidden_channels", 64)
    args.layers = best_params.get("layers", 2)
    args.dropout = best_params.get("dropout", 0.1)
    args.num_grids = best_params.get("num_grids", 12)
    args.batch_size = best_params.get("batch_size", 128)
    args.gamma = best_params.get("gamma", 1.0)
    
    # Fixed parameters
    args.grid_min = -1.1
    args.grid_max = 1.1
    args.log_freq = epochs // 10
    args.use_weighted_loss = True
    args.use_roc_auc = True
    args.use_subset = False
    args.subset_ratio = 1.0
    args.return_history = False
    
    print(f"Created training arguments using Optuna best parameters:")
    print(f"  lr: {args.lr}, wd: {args.wd}, hidden_channels: {args.hidden_channels}")
    print(f"  layers: {args.layers}, dropout: {args.dropout}, num_grids: {args.num_grids}")
    print(f"  batch_size: {args.batch_size}, gamma: {args.gamma}")
    
    return args

def generate_training_script(task_type, dataset_name, target_column=None, 
                           use_global_features=False, output_file=None,
                           results_dir="experiments/optuna_search"):
    """
    Generate a Python script that uses the best hyperparameters from Optuna.
    """
    
    best_params = get_best_hyperparameters(task_type, dataset_name, target_column, 
                                         use_global_features, results_dir)
    
    if best_params is None:
        print("Could not retrieve best hyperparameters.")
        return None
    
    if output_file is None:
        output_file = f"train_{task_type}_{dataset_name}"
        if target_column:
            output_file += f"_{target_column}"
        output_file += f"_global_{use_global_features}_optuna.py"
    
    script_lines = []
    script_lines.append("#!/usr/bin/env python3")
    script_lines.append("# Auto-generated training script using Optuna best hyperparameters")
    script_lines.append("# Generated from Optuna hyperparameter search results")
    script_lines.append("")
    script_lines.append("import argparse")
    
    if task_type == "classification":
        script_lines.append("from graph_classification import graph_classification")
    else:
        script_lines.append("from graph_regression import graph_regression")
    
    script_lines.append("")
    script_lines.append("def main():")
    script_lines.append("    # Best hyperparameters from Optuna search")
    script_lines.append("    args = argparse.Namespace()")
    script_lines.append("")
    script_lines.append(f"    # Dataset configuration")
    script_lines.append(f"    args.dataset_name = '{dataset_name}'")
    if target_column:
        script_lines.append(f"    args.target_column = '{target_column}'")
    script_lines.append(f"    args.use_global_features = {use_global_features}")
    script_lines.append("")
    script_lines.append("    # Best hyperparameters from Optuna")
    script_lines.append(f"    args.lr = {best_params.get('lr', 0.004)}")
    script_lines.append(f"    args.wd = {best_params.get('wd', 0.005)}")
    script_lines.append(f"    args.hidden_channels = {best_params.get('hidden_channels', 64)}")
    script_lines.append(f"    args.layers = {best_params.get('layers', 2)}")
    script_lines.append(f"    args.dropout = {best_params.get('dropout', 0.1)}")
    script_lines.append(f"    args.num_grids = {best_params.get('num_grids', 12)}")
    script_lines.append(f"    args.batch_size = {best_params.get('batch_size', 128)}")
    script_lines.append(f"    args.gamma = {best_params.get('gamma', 1.0)}")
    script_lines.append("")
    script_lines.append("    # Training configuration")
    script_lines.append("    args.epochs = 200")
    script_lines.append("    args.patience = 50")
    script_lines.append("    args.grid_min = -1.1")
    script_lines.append("    args.grid_max = 1.1")
    script_lines.append("    args.log_freq = 20")
    script_lines.append("    args.use_weighted_loss = True")
    script_lines.append("    args.use_roc_auc = True")
    script_lines.append("    args.use_subset = False")
    script_lines.append("    args.subset_ratio = 1.0")
    script_lines.append("    args.return_history = True")
    script_lines.append("")
    script_lines.append("    print(f'Training {task_type} on {dataset_name} using Optuna best hyperparameters...')")
    script_lines.append("    print(f'Global features: {use_global_features}')")
    if target_column:
        script_lines.append("    print(f'Target: {target_column}')")
    script_lines.append("    print('Best hyperparameters:')")
    script_lines.append("    print(f'  lr: {args.lr}, wd: {args.wd}, hidden_channels: {args.hidden_channels}')")
    script_lines.append("    print(f'  layers: {args.layers}, dropout: {args.dropout}, num_grids: {args.num_grids}')")
    script_lines.append("    print(f'  batch_size: {args.batch_size}, gamma: {args.gamma}')")
    script_lines.append("    print()")
    script_lines.append("")
    script_lines.append("    # Run training")
    
    if task_type == "classification":
        script_lines.append("    result = graph_classification(args)")
        script_lines.append("    print(f'Final test performance: {result:.4f}')")
    else:
        script_lines.append("    result = graph_regression(args)")
        script_lines.append("    print(f'Final test MAE: {result:.4f}')")
    
    script_lines.append("")
    script_lines.append("    return result")
    script_lines.append("")
    script_lines.append("if __name__ == '__main__':")
    script_lines.append("    main()")
    
    # Write script to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(script_lines))
    
    print(f"Training script generated: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Retrieve and use Optuna hyperparameters")
    parser.add_argument("--action", type=str, choices=["list", "get", "create_args", "generate_script"], 
                       default="list", help="Action to perform")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], 
                       help="Task type")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--target", type=str, help="Target column (for regression)")
    parser.add_argument("--global_features", action="store_true", 
                       help="Use global features configuration")
    parser.add_argument("--results_dir", type=str, default="experiments/optuna_search",
                       help="Directory containing Optuna results")
    parser.add_argument("--output_file", type=str, help="Output file for generated script")
    
    args = parser.parse_args()
    
    if args.action == "list":
        print("Available hyperparameter configurations:")
        print("=" * 80)
        configurations = list_available_hyperparameters(args.results_dir)
        
        for config in configurations:
            print(f"Task: {config['task_type']:<12} Dataset: {config['dataset_name']:<10}")
            if config['target_column']:
                print(f"  Target: {config['target_column']}")
            print(f"  Global features: {config['use_global_features']}")
            print(f"  Best validation score: {config['best_value']}")
            print(f"  Trials: {config['n_trials']}")
            print(f"  File: {config['filename']}")
            print("-" * 40)
    
    elif args.action == "get":
        if not args.task or not args.dataset:
            print("Error: --task and --dataset are required for 'get' action")
            return
        
        params = get_best_hyperparameters(args.task, args.dataset, args.target, 
                                        args.global_features, args.results_dir)
        if params:
            print("Best hyperparameters:")
            print(json.dumps(params, indent=2))
    
    elif args.action == "create_args":
        if not args.task or not args.dataset:
            print("Error: --task and --dataset are required for 'create_args' action")
            return
        
        training_args = create_training_args_from_optuna(args.task, args.dataset, args.target,
                                                       args.global_features, results_dir=args.results_dir)
        print("Training arguments created successfully")
        
    elif args.action == "generate_script":
        if not args.task or not args.dataset:
            print("Error: --task and --dataset are required for 'generate_script' action")
            return
        
        script_file = generate_training_script(args.task, args.dataset, args.target,
                                             args.global_features, args.output_file, args.results_dir)
        if script_file:
            print(f"Training script generated: {script_file}")

if __name__ == "__main__":
    main()
