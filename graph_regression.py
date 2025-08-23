import argparse
from email import parser
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.nn.functional import l1_loss
import math

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from qm9_dataset import QM9GraphDataset, QM9MultiTaskDataset
from qm8_dataset import QM8GraphDataset, QM8MultiTaskDataset

from src.KANG_regression import KANG
from src.KANG_MultiTask_Regression import KANG_MultiTask_Regression
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
	parser.add_argument("--no_self_loops", action="store_true", help="Disable self loops in the GNN (default: use self loops)")
	# Multi-task arguments
	parser.add_argument("--multitask", action="store_true", help="Use multi-task learning")
	parser.add_argument("--multitask_targets", type=str, nargs='+', default=None, 
	                   help="Specific targets for multi-task learning. If None, uses default targets")
	parser.add_argument("--task_weights", type=str, default=None, 
	                   help="JSON string with task weights for multi-task loss")
	return parser.parse_args()

def graph_regression(args, return_history=False):
	# Check if multi-task learning is enabled
	if args.multitask:
		return graph_regression_multitask(args, return_history)
	
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
		use_self_loops=not getattr(args, 'no_self_loops', False)
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
			edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
			
			if use_amp:
				with autocast():
					out = model(data.x, data.edge_index, data.batch, global_features, data.edge_attr).view(-1)
					loss = criterion(out, data.y.view(-1).float())
				epoch_loss += loss.item()
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			else:
				out = model(data.x, data.edge_index, data.batch, global_features, data.edge_attr).view(-1)
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
						preds = model(data.x, data.edge_index, data.batch, global_features, data.edge_attr).view(-1)
						targets = data.y.view(-1).float()
						total_loss += criterion(preds, targets).item()
						total_mae += torch.abs(preds - targets).mean().item()
				else:
					preds = model(data.x, data.edge_index, data.batch, global_features, data.edge_attr).view(-1)
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
					preds = model(data.x, data.edge_index, data.batch, global_features, data.edge_attr).view(-1)
					targets = data.y.view(-1).float()
					total_loss += criterion(preds, targets).item()
					total_mae += torch.abs(preds - targets).mean().item()
			else:
				preds = model(data.x, data.edge_index, data.batch, global_features, data.edge_attr).view(-1)
				targets = data.y.view(-1).float()
				total_loss += criterion(preds, targets).item()
				total_mae += torch.abs(preds - targets).mean().item()
	test_rmse = math.sqrt(total_loss / len(test_loader))
	test_mae = total_mae / len(test_loader)
	print(f'Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}')

	if return_history:
		return best_val_score, train_losses, val_metrics, test_mae  # Return validation score for optimization, test MAE separately
	else:
		return best_val_score, test_mae  # Return validation score for optimization, test MAE separately


def graph_regression_multitask(args, return_history=False):
	"""Multi-task graph regression for QM8/QM9 datasets"""
	import json
	
	print("Multi-Task Graph Regression Mode")
	print("=" * 50)
	
	# Determine which targets to use
	if args.multitask_targets is not None:
		target_columns = args.multitask_targets
		print(f"Using specified targets: {target_columns}")
	else:
		# Use default sets based on dataset
		if args.dataset_name == "QM9":
			target_columns = ['mu', 'alpha', 'homo', 'lumo', 'gap']
		elif args.dataset_name == "QM8":
			target_columns = ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2']
		else:
			raise ValueError(f"Unsupported dataset for multi-task: {args.dataset_name}")
		
		print(f"Using default target set ({len(target_columns)} targets):")
		for i, target in enumerate(target_columns):
			print(f"  {i+1}. {target}")
	
	# Load multi-task dataset
	if args.dataset_name == "QM9":
		# Create a short hash for QM9 targets
		target_str = " ".join(sorted(target_columns))
		target_hash = str(sum(ord(c) for c in target_str) % 10**8)
		dataset_path = f'./dataset/QM9_multitask_{target_hash}'
		dataset = QM9MultiTaskDataset(
			root=dataset_path,
			target_columns=target_columns,
			use_global_features=args.use_global_features
		)
	elif args.dataset_name == "QM8":
		# Create a short hash for QM8 targets
		target_str = " ".join(sorted(target_columns))
		target_hash = str(sum(ord(c) for c in target_str) % 10**8)
		dataset_path = f'./dataset/QM8_multitask_{target_hash}'
		dataset = QM8MultiTaskDataset(
			root=dataset_path,
			target_columns=target_columns,
			use_global_features=args.use_global_features
		)
	else:
		raise ValueError(f"Unsupported dataset for multi-task: {args.dataset_name}")
	
	print(f"Dataset loaded: {len(dataset)} molecules, {dataset.get_num_tasks()} tasks")
	if args.use_global_features:
		print("Using global molecular features")
	
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
	num_workers = 0 if args.use_global_features else 4
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
	                         pin_memory=True, num_workers=num_workers, persistent_workers=num_workers>0)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
	                       pin_memory=True, num_workers=min(num_workers, 2), persistent_workers=num_workers>0)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
	                        pin_memory=True, num_workers=min(num_workers, 2), persistent_workers=num_workers>0)

	# Parse task weights if provided
	task_weights = None
	if args.task_weights:
		try:
			task_weights_dict = json.loads(args.task_weights)
			task_weights = [task_weights_dict.get(target, 1.0) for target in target_columns]
			task_weights = torch.tensor(task_weights).to(device)
			print(f"Using custom task weights: {task_weights.tolist()}")
		except:
			print("Warning: Could not parse task weights, using uniform weights")
			task_weights = None

	criterion = nn.L1Loss()

	# Create multi-task model - select based on single_head parameter
	if getattr(args, 'single_head', False):
		from src.KANG_MultiTask_Regression_SingleHead import KANG_MultiTask_Regression_SingleHead
		model = KANG_MultiTask_Regression_SingleHead(
			dataset.num_node_features,
			args.hidden_channels,
			len(target_columns),  # Number of tasks
			args.layers,
			args.grid_min,
			args.grid_max,
			args.num_grids,
			args.dropout,
			device=device,
			use_global_features=args.use_global_features,
			use_self_loops=not getattr(args, 'no_self_loops', False)
		).to(device)
		print("Using single head multi-task model")
	else:
		model = KANG_MultiTask_Regression(
			dataset.num_node_features,
			args.hidden_channels,
			len(target_columns),  # Number of tasks
			args.layers,
			args.grid_min,
			args.grid_max,
			args.num_grids,
			args.dropout,
			device=device,
			use_global_features=args.use_global_features,
			use_self_loops=not getattr(args, 'no_self_loops', False)
		).to(device)
		print("Using multi head multi-task model")
	
	print(f"Multi-task model created with {sum(p.numel() for p in model.parameters()):,} parameters")
	
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

	# Initialize mixed precision scaler if available
	scaler = GradScaler() if AMP_AVAILABLE and device.type == 'cuda' else None
	use_amp = scaler is not None

	best_val_metric = float('inf')  # MAE - lower is better
	early_stop_counter = 0
	best_epoch = -1
	best_model_path = f"./experiments/graph_regression/gkan_multitask.pth"

	# Initialize history tracking if requested
	train_losses = [] if return_history else None
	val_metrics = [] if return_history else None

	print(f"Training multi-task model for {args.epochs} epochs...")
	print(f"Training with mixed precision: {use_amp}")
	
	for epoch in range(args.epochs):
		model.train()
		epoch_loss = 0
		total_valid_batches = 0
		
		for data in train_loader:
			optimizer.zero_grad()
			data = data.to(device, non_blocking=True)
			global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
			edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
			
			# Reshape labels for multi-task
			num_graphs = data.batch.max().item() + 1
			labels = data.y.view(num_graphs, len(target_columns))
			
			if use_amp:
				with autocast():
					outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
					# Multi-task loss computation
					total_loss = 0
					valid_tasks = 0
					
					for task_idx in range(len(target_columns)):
						task_mask = ~torch.isnan(labels[:, task_idx])
						if task_mask.sum() > 0:
							task_pred = outputs[task_mask, task_idx].view(-1)
							task_target = labels[task_mask, task_idx].float()
							task_loss = criterion(task_pred, task_target)
							weight = task_weights[task_idx] if task_weights is not None else 1.0
							total_loss += weight * task_loss
							valid_tasks += 1
					
					if valid_tasks > 0:
						avg_loss = total_loss / valid_tasks
						epoch_loss += avg_loss.item()
						total_valid_batches += 1
				
				if valid_tasks > 0:
					scaler.scale(avg_loss).backward()
					scaler.step(optimizer)
					scaler.update()
			else:
				outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
				# Multi-task loss computation
				total_loss = 0
				valid_tasks = 0
				
				for task_idx in range(len(target_columns)):
					task_mask = ~torch.isnan(labels[:, task_idx])
					if task_mask.sum() > 0:
						task_pred = outputs[task_mask, task_idx].view(-1)
						task_target = labels[task_mask, task_idx].float()
						task_loss = criterion(task_pred, task_target)
						weight = task_weights[task_idx] if task_weights is not None else 1.0
						total_loss += weight * task_loss
						valid_tasks += 1
				
				if valid_tasks > 0:
					avg_loss = total_loss / valid_tasks
					epoch_loss += avg_loss.item()
					total_valid_batches += 1
					avg_loss.backward()
					optimizer.step()

		# Validation evaluation
		model.eval()
		with torch.no_grad():
			# Multi-task MAE evaluation
			task_maes = []
			
			for task_idx in range(len(target_columns)):
				total_error = 0
				total_count = 0
				
				for data in val_loader:
					data = data.to(device, non_blocking=True)
					global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
					edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
					
					num_graphs = data.batch.max().item() + 1
					labels = data.y.view(num_graphs, len(target_columns))
					
					if use_amp:
						with autocast():
							outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
					else:
						outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
					
					# Get MAE for this task
					task_mask = ~torch.isnan(labels[:, task_idx])
					if task_mask.sum() > 0:
						pred = outputs[task_mask, task_idx].view(-1)
						target = labels[task_mask, task_idx].float()
						total_error += torch.abs(pred - target).sum().item()
						total_count += target.size(0)
				
				task_mae = total_error / total_count if total_count > 0 else float('inf')
				task_maes.append(task_mae)
			
			val_metric = sum(task_maes) / len(task_maes)  # Average MAE across tasks

		# Track training history if requested
		if return_history:
			avg_epoch_loss = epoch_loss / total_valid_batches if total_valid_batches > 0 else 0
			train_losses.append(avg_epoch_loss)
			val_metrics.append(val_metric)
		
		if val_metric < best_val_metric:
			best_epoch = epoch
			best_val_metric = val_metric
			torch.save(model.state_dict(), best_model_path)
			early_stop_counter = 0
		else:
			early_stop_counter += 1
		
		if early_stop_counter >= args.patience:
			print(f"Early stopping at epoch {epoch}")
			break

		if epoch % args.log_freq == 0 or epoch == args.epochs - 1:
			avg_epoch_loss = epoch_loss / total_valid_batches if total_valid_batches > 0 else 0
			print(f"Epoch {epoch:03d}: Train Loss: {avg_epoch_loss:.4f}, Val Multi-Task MAE: {val_metric:.4f}")

	# Load best model and evaluate on test set
	print(f"\nBest model saved at epoch {best_epoch} with validation metric: {best_val_metric:.4f}")
	model.load_state_dict(torch.load(best_model_path))
	model.eval()
	
	# Test evaluation
	print("Evaluating on test set...")
	with torch.no_grad():
		# Multi-task test MAE
		task_test_maes = []
		
		for task_idx in range(len(target_columns)):
			total_error = 0
			total_count = 0
			
			for data in test_loader:
				data = data.to(device, non_blocking=True)
				global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
				edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
				
				num_graphs = data.batch.max().item() + 1
				labels = data.y.view(num_graphs, len(target_columns))
				
				if use_amp:
					with autocast():
						outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
				else:
					outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
				
				task_mask = ~torch.isnan(labels[:, task_idx])
				if task_mask.sum() > 0:
					pred = outputs[task_mask, task_idx].view(-1)
					target = labels[task_mask, task_idx].float()
					total_error += torch.abs(pred - target).sum().item()
					total_count += target.size(0)
			
			task_mae = total_error / total_count if total_count > 0 else float('inf')
			task_test_maes.append(task_mae)
			print(f"  {target_columns[task_idx]}: MAE = {task_mae:.4f}")
		
		test_metric = sum(task_test_maes) / len(task_test_maes)
		print(f"Average Test MAE: {test_metric:.4f}")

	print("=" * 50)
	print("Multi-Task Training Complete!")
	
	if return_history:
		return best_val_metric, train_losses, val_metrics, test_metric
	else:
		return best_val_metric, test_metric


def main():
    args = get_args()
    graph_regression(args)

if __name__ == "__main__":
	main()
