import argparse
from email import parser
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.nn.functional import l1_loss
import math

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from qm9_dataset import QM9GraphDataset
from qm8_dataset import QM8GraphDataset

from src.KANG_regression import KANG
from src.utils import set_seed

# Mixed precision training
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 42
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
	parser = argparse.ArgumentParser(description="GKAN - Graph Regression Example")
	parser.add_argument("--dataset_name", type=str, default="QM9", help="Dataset name")
	parser.add_argument("--target_column", type=str, default="mu", help="Target column")
	parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
	parser.add_argument("--patience", type=int, default=300, help="Early stopping patience")
	parser.add_argument("--lr", type=float, default=0.004, help="Learning rate")
	parser.add_argument("--wd", type=float, default=0.005, help="Weight decay")
	parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
	parser.add_argument("--hidden_channels", type=int, default=32, help="Hidden layer dimension")
	parser.add_argument("--layers", type=int, default=2, help="Number of GNN layers")
	parser.add_argument("--num_grids", type=int, default=6, help="Number of splines")
	parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
	parser.add_argument("--grid_min", type=int, default=-10, help="")
	parser.add_argument("--grid_max", type=int, default=3, help="")
	parser.add_argument("--log_freq", type=int, default=10, help="Logging frequency (epochs)")
	parser.add_argument("--use_global_features", action="store_true", help="Use global molecular features")
	return parser.parse_args()

def graph_regression(args, return_history=False):
	if args.dataset_name == "QM9":
		dataset_path = f'./dataset/{args.dataset_name}_{args.target_column}'
		dataset = QM9GraphDataset(root=dataset_path, target_column=args.target_column, use_global_features=args.use_global_features)
		print(f"QM9 dataset loaded with target column: {args.target_column}")
		if args.use_global_features:
			print("Using global molecular features")
		dataset.print_dataset_info()	

	elif args.dataset_name == "QM8":
		dataset_path = f'./dataset/{args.dataset_name}_{args.target_column}'
		dataset = QM8GraphDataset(root=dataset_path, target_column=args.target_column, use_global_features=args.use_global_features)
		print(f"QM8 dataset loaded with target column: {args.target_column}")
		if args.use_global_features:
			print("Using global molecular features")
		dataset.print_dataset_info()

	# Apply subset for faster hyperparameter tuning if specified
	if getattr(args, "use_subset", False):
		subset_size = int(len(dataset) * getattr(args, "subset_ratio", 0.3))
		dataset = dataset[:subset_size]
		print(f"Using subset of {subset_size} samples ({getattr(args, 'subset_ratio', 0.3)*100:.0f}% of full dataset) for faster tuning")

	shuffled_dataset = dataset.shuffle()
	train_size = int(0.8 * len(dataset))
	val_size = int(0.1 * len(dataset))
	train_dataset = shuffled_dataset[:train_size]
	val_dataset = shuffled_dataset[train_size:train_size + val_size]
	test_dataset = shuffled_dataset[train_size + val_size:]
	# DataLoader settings - reduce workers on Windows due to RDKit pickling issues
	num_workers = 0 if args.use_global_features else 4  # RDKit functions can't be pickled on Windows
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
	                         pin_memory=True, num_workers=num_workers, persistent_workers=num_workers>0)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
	                       pin_memory=True, num_workers=min(num_workers, 2), persistent_workers=num_workers>0)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
	                        pin_memory=True, num_workers=min(num_workers, 2), persistent_workers=num_workers>0)

	criterion = nn.L1Loss()

	# Modify output_dim to 1 for regression
	model = KANG(
		dataset.num_node_features,
		args.hidden_channels,
		1,  # Regression output
		args.layers,
		args.grid_min,
		args.grid_max,
		args.num_grids,
		args.dropout,
		device=device,
		use_global_features=args.use_global_features,
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

	# Initialize mixed precision scaler if available
	scaler = GradScaler() if AMP_AVAILABLE and device.type == 'cuda' else None
	use_amp = scaler is not None

	best_val_score = float("inf")
	early_stop_counter = 0
	best_epoch = -1
	best_model_path = f"./experiments/graph_regression/gkan.pth"

	# Initialize history tracking if requested
	train_losses = [] if return_history else None
	val_metrics = [] if return_history else None

	print(f"Training with mixed precision: {use_amp}")
	print(f"DataLoader settings: pin_memory=True, num_workers={num_workers} (adaptive for global features)")

	for epoch in range(args.epochs):
		model.train()
		epoch_loss = 0
		for data in train_loader:
			optimizer.zero_grad()
			data = data.to(device, non_blocking=True)
			global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
			
			if use_amp:
				with autocast():
					out = model(data.x, data.edge_index, data.batch, global_features).view(-1)
					loss = criterion(out, data.y.view(-1).float())
				epoch_loss += loss.item()
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			else:
				out = model(data.x, data.edge_index, data.batch, global_features).view(-1)
				loss = criterion(out, data.y.view(-1).float())
				epoch_loss += loss.item()
				loss.backward()
				optimizer.step()
		epoch_loss /= len(train_loader)

		# Validation
		model.eval()
		total_loss = 0
		total_mae = 0
		with torch.no_grad():
			for data in val_loader:
				data = data.to(device, non_blocking=True)
				global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
				
				if use_amp:
					with autocast():
						preds = model(data.x, data.edge_index, data.batch, global_features).view(-1)
						targets = data.y.view(-1).float()
						total_loss += criterion(preds, targets).item()
						total_mae += torch.abs(preds - targets).mean().item()
				else:
					preds = model(data.x, data.edge_index, data.batch, global_features).view(-1)
					targets = data.y.view(-1).float()
					total_loss += criterion(preds, targets).item()
					total_mae += torch.abs(preds - targets).mean().item()
		avg_val_loss = total_loss / len(val_loader)
		avg_val_mae = total_mae / len(val_loader)

		# Track training history if requested
		if return_history:
			train_losses.append(epoch_loss)
			val_metrics.append(avg_val_mae)

		if avg_val_mae < best_val_score:
			best_val_score = avg_val_mae
			best_epoch = epoch
			torch.save(model.state_dict(), best_model_path)
			early_stop_counter = 0
		else:
			early_stop_counter += 1
		if early_stop_counter >= args.patience:
			break

		if epoch % args.log_freq == 0 or epoch == args.epochs - 1:
			print(f"Epoch {epoch:03d}: Train Loss: {epoch_loss:.4f}, Val MSE: {avg_val_loss:.4f}, Val RMSE: {math.sqrt(avg_val_loss):.4f}, Val MAE: {avg_val_mae:.4f}")

	print(f"\nBest model was saved at epoch {best_epoch} with val MAE: {best_val_score:.4f}")
	model.load_state_dict(torch.load(best_model_path))
	model.eval()

	# Test Evaluation
	total_loss = 0
	total_mae = 0
	with torch.no_grad():
		for data in test_loader:
			data = data.to(device, non_blocking=True)
			global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
			
			if use_amp:
				with autocast():
					preds = model(data.x, data.edge_index, data.batch, global_features).view(-1)
					targets = data.y.view(-1).float()
					total_loss += criterion(preds, targets).item()
					total_mae += torch.abs(preds - targets).mean().item()
			else:
				preds = model(data.x, data.edge_index, data.batch, global_features).view(-1)
				targets = data.y.view(-1).float()
				total_loss += criterion(preds, targets).item()
				total_mae += torch.abs(preds - targets).mean().item()
	test_rmse = math.sqrt(total_loss / len(test_loader))
	test_mae = total_mae / len(test_loader)
	print(f'Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}')

	if return_history:
		return best_val_score, train_losses, val_metrics
	else:
		return best_val_score


def main():
    args = get_args()
    graph_regression(args)

if __name__ == "__main__":
	main()
