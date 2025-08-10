import argparse
from graph_regression import graph_regression
from graph_classification import graph_classification
from plotting_utils import plot_training_metrics

def get_args():
    parser = argparse.ArgumentParser(description="Run single training with plots")
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], required=True, help='Task type: classification or regression')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--target_column', type=str, default=None, help='Target column for regression tasks')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels in the model')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers in the model')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--num_grids', type=int, default=10, help='Number of grids for KAN layers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma parameter for learning rate scheduling')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=300, help='Early stopping patience')
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set constant arguments
    args.grid_min = 0
    if args.task == "classification":
        args.grid_max = 2.1
    else:
        args.grid_max = 1.1
    
    args.log_freq = args.epochs // 10
    args.use_weighted_loss = True
    args.use_roc_auc = True
    args.return_history = True  # Flag to return training history
    args.use_subset = False  # Use full dataset for final training
    args.subset_ratio = 1.0
    args.use_global_features = True  # Use global molecular features

    if args.task == "classification":
        try:
            _, train_losses, val_metrics, _ = graph_classification(args, return_history=True)
            plot_training_metrics(train_losses, val_metrics, args.task, args.dataset_name)
        except Exception as e:
            print(f"Warning: Could not generate training plots for classification: {e}")
    elif args.task == "regression":
        try:
            _, train_losses, val_metrics, _ = graph_regression(args, return_history=True)
            plot_training_metrics(train_losses, val_metrics, args.task, args.dataset_name, args.target_column)
        except Exception as e:
            print(f"Warning: Could not generate training plots for regression: {e}")

if __name__ == "__main__":
    main()