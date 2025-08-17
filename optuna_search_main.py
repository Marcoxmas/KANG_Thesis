import argparse
import numpy as np
import optuna
import os
import json
from graph_classification import graph_classification
from graph_regression import graph_regression
from plotting_utils import plot_training_metrics

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization with Optuna")
    parser.add_argument("--task", type=str, choices=["classification", "regression"], required=True, help="Task type")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--target_column", type=str, default="mu", help="Target column for regression tasks")
    parser.add_argument("--use_subset", action="store_true", help="Use a smaller subset of the dataset during tuning")
    parser.add_argument("--subset_ratio", type=float, default=0.3, help="Ratio of dataset to use for subset (default: 0.3)")
    parser.add_argument("--use_global_features", action="store_true", help="Use global molecular features")
    parser.add_argument("--no_self_loops", action="store_true", help="Disable self loops in the GNN (default: use self loops)")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials (default: 20)")
    # Multi-task arguments
    parser.add_argument("--multitask", action="store_true", help="Use multi-task learning")
    parser.add_argument("--multitask_assays", type=str, nargs='+', default=None, 
                       help="Specific assays for multi-task classification (ToxCast). If None, uses default assays")
    parser.add_argument("--multitask_targets", type=str, nargs='+', default=None, 
                       help="Specific targets for multi-task regression (QM8/QM9). If None, uses default targets")
    parser.add_argument("--task_weights", type=str, default=None, 
                       help="JSON string with task weights for multi-task loss")
    return parser.parse_args()

def optuna_search(task_type, dataset_name, target_column, use_subset=True, subset_ratio=0.3, use_global_features=False, no_self_loops=False, n_trials=20, multitask=False, multitask_assays=None, multitask_targets=None, task_weights=None):
    def objective(trial):
        try:
            import argparse
            args = argparse.Namespace()
            args.dataset_name = dataset_name
            args.target_column = target_column
            args.lr = trial.suggest_float("lr", 0.001, 0.01, log=True)
            args.wd = trial.suggest_float("wd", 1e-5, 1e-3, log=True)
            args.hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
            args.layers = trial.suggest_int("layers", 3, 9)
            args.dropout = trial.suggest_float("dropout", 0.1, 0.5)
            args.num_grids = trial.suggest_categorical("num_grids", [4, 6, 8, 10, 12])
            args.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            args.gamma = trial.suggest_float("gamma", 0.5, 2.5)
            args.grid_min = -1
            args.grid_max = 2.5 if task_type == "classification" else 1.5
            args.epochs = 100
            args.patience = 30
            args.log_freq = args.epochs // 10
            args.use_weighted_loss = True
            args.use_roc_auc = True
            # Enable subset for faster hyperparameter tuning
            args.use_subset = use_subset
            args.subset_ratio = subset_ratio
            args.use_global_features = use_global_features
            args.no_self_loops = no_self_loops
            # Multi-task arguments
            args.multitask = multitask
            args.multitask_assays = multitask_assays
            args.multitask_targets = multitask_targets
            args.task_weights = task_weights

            if task_type == "classification":
                print("Running classification with:", args)
                best_val_acc, test_metric = graph_classification(args)
                # Check for invalid results
                if best_val_acc is None or np.isnan(best_val_acc) or np.isinf(best_val_acc):
                    raise ValueError("Invalid validation accuracy returned")
                return best_val_acc  # maximize accuracy (use validation for optimization)
            elif task_type == "regression":
                print("Running regression with:", args)
                best_val_score, test_metric = graph_regression(args)
                # Check for invalid results
                if best_val_score is None or np.isnan(best_val_score) or np.isinf(best_val_score):
                    raise ValueError("Invalid validation score returned")
                return best_val_score  # minimize MAE (use validation for optimization)
        except Exception as e:
            print(f"Trial failed with error: {e}")
            # Return a penalty value instead of raising
            if task_type == "classification":
                return 0.0  # Worst possible accuracy
            else:
                return float('inf')  # Worst possible MAE
    study = optuna.create_study(direction="minimize" if task_type == "regression" else "maximize")
    study.optimize(objective, n_trials=n_trials)

    # Create unique filename for best parameters
    results_dir = "experiments/optuna_search"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create unique filename based on dataset, target, and multi-task settings
    param_file = f"{results_dir}/best_params_{task_type}_{dataset_name}"
    if multitask:
        if multitask_assays:
            # Create a short hash for assays
            assay_str = " ".join(sorted(multitask_assays))
            assay_hash = str(sum(ord(c) for c in assay_str) % 10**8)
            param_file += f"_multitask_{assay_hash}"
        elif multitask_targets:
            # Create a short hash for targets
            target_str = " ".join(sorted(multitask_targets))
            target_hash = str(sum(ord(c) for c in target_str) % 10**8)
            param_file += f"_multitask_{target_hash}"
        else:
            param_file += "_multitask_default"
    elif target_column:
        param_file += f"_{target_column}"
    param_file += f"_global_{use_global_features}.json"

    # Save best hyperparameters
    with open(param_file, "w") as f:
        best_params = study.best_trial.params.copy()
        best_params["use_subset"] = use_subset
        best_params["subset_ratio"] = subset_ratio
        best_params["use_global_features"] = use_global_features
        best_params["no_self_loops"] = no_self_loops
        best_params["task_type"] = task_type
        best_params["dataset_name"] = dataset_name
        best_params["target_column"] = target_column
        best_params["multitask"] = multitask
        best_params["multitask_assays"] = multitask_assays
        best_params["multitask_targets"] = multitask_targets
        best_params["task_weights"] = task_weights
        best_params["best_value"] = study.best_value
        best_params["n_trials"] = n_trials
        json.dump(best_params, f, indent=4)

    print(f"\nBest hyperparameters saved to: {param_file}")
    print("Best hyperparameters:")
    print(study.best_trial.params)
    if use_subset:
        print(f"Note: Hyperparameters found using {subset_ratio*100:.0f}% of the dataset for faster tuning.")
    
    # COMMENTED OUT: Final training moved to train_with_best_params.py
    # This allows for better seed tracking and separate training runs
    print("\nOptuna search completed. Use train_with_best_params.py for final training with seed tracking.")
    
    # # Run the best trial again to get final test performance on FULL dataset
    # print("\nRunning best hyperparameters on FULL dataset to get final test performance...")
    # import argparse
    # best_args = argparse.Namespace()
    # best_args.dataset_name = dataset_name
    # best_args.target_column = target_column
    # 
    # # Apply best hyperparameters
    # for param, value in study.best_trial.params.items():
    #     setattr(best_args, param, value)
    # 
    # # Set fixed parameters - DISABLE subset for final training
    # best_args.grid_min = -1
    # best_args.grid_max = 2.5 if task_type == "classification" else 1.5
    # best_args.epochs = 200
    # best_args.patience = 50
    # best_args.log_freq = best_args.epochs // 10
    # best_args.use_weighted_loss = True
    # best_args.use_roc_auc = True
    # best_args.return_history = True  # Flag to return training history
    # best_args.use_subset = False  # Use full dataset for final training
    # best_args.subset_ratio = 1.0
    # best_args.use_global_features = use_global_features
    # best_args.no_self_loops = no_self_loops
    # # Multi-task arguments
    # best_args.multitask = multitask
    # best_args.multitask_assays = multitask_assays
    # best_args.multitask_targets = multitask_targets
    # best_args.task_weights = task_weights
    # 
    # final_test_metric = None
    # 
    # if task_type == "classification":
    #     try:
    #         best_val_acc, train_losses, val_metrics, final_test_metric = graph_classification(best_args, return_history=True)
    #         # Create appropriate plot name for multi-task
    #         plot_dataset_name = f"{dataset_name}_multitask" if multitask else dataset_name
    #         plot_training_metrics(train_losses, val_metrics, task_type, plot_dataset_name)
    #     except Exception as e:
    #         print(f"Warning: Could not generate training plots for classification: {e}")
    #         try:
    #             best_val_acc, final_test_metric = graph_classification(best_args)
    #         except Exception as e2:
    #             print(f"Error getting final test metric: {e2}")
    # elif task_type == "regression":
    #     try:
    #         best_val_score, train_losses, val_metrics, final_test_metric = graph_regression(best_args, return_history=True)
    #         # Create appropriate plot name for multi-task
    #         plot_dataset_name = f"{dataset_name}_multitask" if multitask else dataset_name
    #         plot_training_metrics(train_losses, val_metrics, task_type, plot_dataset_name, target_column)
    #     except Exception as e:
    #         print(f"Warning: Could not generate training plots for regression: {e}")
    #         try:
    #             best_val_score, final_test_metric = graph_regression(best_args)
    #         except Exception as e2:
    #             print(f"Error getting final test metric: {e2}")
    
    # Return validation score from hyperparameter search (no final training)
    final_test_metric = None  # Will be set by train_with_best_params.py
    
    return study.best_value, study.best_trial.params

if __name__ == "__main__":
    args = get_args()
    
    # Parse comma-separated multitask arguments
    if args.multitask_assays and len(args.multitask_assays) == 1 and ',' in args.multitask_assays[0]:
        args.multitask_assays = args.multitask_assays[0].split(',')
    if args.multitask_targets and len(args.multitask_targets) == 1 and ',' in args.multitask_targets[0]:
        args.multitask_targets = args.multitask_targets[0].split(',')
    
    validation_score, best_params = optuna_search(
        args.task, 
        args.dataset_name, 
        args.target_column, 
        args.use_subset, 
        args.subset_ratio, 
        args.use_global_features,
        args.no_self_loops,
        args.n_trials,
        args.multitask,
        args.multitask_assays,
        args.multitask_targets,
        args.task_weights
    )
    
    print(f"Hyperparameter search completed with validation score: {validation_score}")
