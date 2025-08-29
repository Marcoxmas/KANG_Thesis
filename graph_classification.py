import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.metrics import roc_auc_score
from toxcast_dataset import ToxCastGraphDataset
from hiv_dataset import HIVGraphDataset

from src.KANG import KANG
from src.KANG_MultiTask import KANG_MultiTask
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

class FocalLoss(nn.Module):
	def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
		super(FocalLoss, self).__init__()
		self.alpha = alpha  # Class weights
		self.gamma = gamma  # Focusing parameter
		self.reduction = reduction
		
	def forward(self, inputs, targets):
		# Convert to probabilities
		ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
		pt = torch.exp(-ce_loss)

		# Clamp pt to avoid numerical issues
		pt = torch.clamp(pt, min=1e-8, max=1.0)
		focal_loss = (1 - pt) ** self.gamma * ce_loss
		
		if self.reduction == 'mean':
			return focal_loss.mean()
		elif self.reduction == 'sum':
			return focal_loss.sum()
		else:
			return focal_loss

def generate_model_filename(args, is_multitask=False):
	"""Generate a unique model filename based on training configuration"""
	components = []
	
	# Base model type
	if is_multitask:
		components.append("gkan_multitask")
	else:
		components.append("gkan")
	
	# Dataset name
	components.append(args.dataset_name.lower())
	
	# Key hyperparameters
	components.append(f"hc{args.hidden_channels}")
	components.append(f"l{args.layers}")
	components.append(f"g{args.num_grids}")
	components.append(f"lr{args.lr}")
	components.append(f"wd{args.wd}")
	components.append(f"d{args.dropout}")
	
	# Feature flags
	if args.use_global_features:
		components.append("gf")
	if args.use_3d_geo:
		components.append("3d")
	if getattr(args, 'no_self_loops', False):
		components.append("nosl")
	
	# Loss/evaluation specific
	if args.use_weighted_loss:
		components.append("wl")
	if args.use_roc_auc:
		components.append("auc")
		if hasattr(args, 'gamma') and args.gamma != 1.0:
			components.append(f"gamma{args.gamma}")
	
	# Multi-task specific options
	if is_multitask:
		if getattr(args, 'single_head', False):
			components.append("sh")
		if hasattr(args, 'multitask_assays') and args.multitask_assays:
			# Add hash of assays for uniqueness
			assay_str = "_".join(sorted(args.multitask_assays))
			assay_hash = str(abs(hash(assay_str)) % 10000)
			components.append(f"a{assay_hash}")
	
	return "_".join(components) + ".pth"

def get_args():
	parser = argparse.ArgumentParser(description="GKAN - Graph Classification Example")
	parser.add_argument("--dataset_name", type=str, default="MUTAG", help="Dataset name")
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
	parser.add_argument("--use_weighted_loss", action="store_true", help="Use weighted loss to handle class imbalance")
	parser.add_argument("--use_roc_auc", action="store_true", help="Evaluate using ROC-AUC instead of accuracy")
	parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter for Focal Loss (default: 1.0)")
	parser.add_argument("--return_history", action="store_true", help="Return training history for plotting")
	parser.add_argument("--use_global_features", action="store_true", help="Use global molecular features")
	parser.add_argument("--use_3d_geo", action="store_true", help="Use 3D geometric features for molecular graphs")
	parser.add_argument("--no_self_loops", action="store_true", help="Disable self loops in the GNN (default: use self loops)")
	# Multi-task arguments
	parser.add_argument("--multitask", action="store_true", help="Use multi-task learning for ToxCast dataset")
	parser.add_argument("--multitask_assays", type=str, nargs='+', default=None, 
	                   help="Specific assays for multi-task learning. If None, uses all available assays")
	parser.add_argument("--task_weights", type=str, default=None, 
	                   help="JSON string with task weights for multi-task loss (e.g., '{\"task1\": 1.0, \"task2\": 2.0}')")
	return parser.parse_args()

def graph_classification(args, return_history=False):
	# Check if multi-task learning is enabled
	if args.multitask and args.dataset_name == "TOXCAST":
		return graph_classification_multitask(args, return_history)
	
	# Determine the dataset type based on the dataset name
	if args.dataset_name in ["MUTAG", "PROTEINS"]:
		dataset_path = f'./dataset/{args.dataset_name}'
		dataset = TUDataset(root=dataset_path, name=args.dataset_name)
		if args.use_global_features:
			print("Warning: Global features not supported for TUDataset. Ignoring --use_global_features flag.")
			args.use_global_features = False
		if args.use_3d_geo:
			print("Warning: 3D geometry not supported for TUDataset. Ignoring --use_3d_geo flag.")
			args.use_3d_geo = False
	elif args.dataset_name == "HIV":
		dataset_path = f'./dataset/{args.dataset_name}'
		dataset = HIVGraphDataset(root=dataset_path, use_global_features=args.use_global_features, use_3d_geo=args.use_3d_geo)
		if args.use_global_features:
			print("Using global molecular features")
		if args.use_3d_geo:
			print("Using 3D geometric features")
	elif args.dataset_name == "TOXCAST":
		dataset_path = f'./dataset/TOXCAST_{args.target_column}'
		dataset = ToxCastGraphDataset(root=dataset_path, target_column=args.target_column, use_global_features=args.use_global_features, use_3d_geo=args.use_3d_geo)
		if args.use_global_features:
			print("Using global molecular features")
		if args.use_3d_geo:
			print("Using 3D geometric features")
	else:
		# Fallback for other datasets
		dataset_path = f'./dataset/{args.dataset_name}'
		dataset = ToxCastGraphDataset(root=dataset_path, target_column=args.dataset_name, use_global_features=args.use_global_features, use_3d_geo=args.use_3d_geo)
		if args.use_global_features:
			print("Using global molecular features")
		if args.use_3d_geo:
			print("Using 3D geometric features")

	# Apply subset for faster hyperparameter tuning if specified
	if getattr(args, "use_subset", False):
		subset_size = int(len(dataset) * getattr(args, "subset_ratio", 0.3))
		dataset = dataset[:subset_size]
		print(f"Using subset of {subset_size} samples ({getattr(args, 'subset_ratio', 0.3)*100:.0f}% of full dataset) for faster tuning")

	shuffled_dataset = dataset.shuffle()
	train_size 		= int(0.8 * len(dataset))
	val_size 			= int(0.1 * len(dataset))
	train_dataset = shuffled_dataset[:train_size]
	val_dataset 	= shuffled_dataset[train_size:train_size + val_size]
	test_dataset 	= shuffled_dataset[train_size + val_size:]
	# DataLoader settings - reduce workers on Windows due to RDKit pickling issues
	num_workers = 0 if args.use_global_features else 4  # RDKit functions can't be pickled on Windows
	train_loader 	= DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
	                            pin_memory=True, num_workers=num_workers, persistent_workers=num_workers>0)
	val_loader 		= DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
	                           pin_memory=True, num_workers=min(num_workers, 2), persistent_workers=num_workers>0)
	test_loader 	= DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
	                          pin_memory=True, num_workers=min(num_workers, 2), persistent_workers=num_workers>0)

	# Handle class imbalance and loss function selection
	weights = None
	if args.use_weighted_loss:
		from collections import Counter
		labels = [int(data.y.item()) for data in dataset]
		label_counts = Counter(labels)
		total = sum(label_counts.values())
		class_weights = [total / label_counts[i] for i in range(dataset.num_classes)]
		weights = torch.tensor(class_weights).to(device)
	
	# Choose loss function based on whether ROC-AUC optimization is requested
	if args.use_roc_auc:
		# Use Focal Loss for better ROC-AUC alignment
		criterion = FocalLoss(alpha=weights, gamma=args.gamma)
	else:
		# Use standard CrossEntropy Loss (with or without weights)
		if weights is not None:
			criterion = nn.CrossEntropyLoss(weight=weights)
		else:
			criterion = nn.CrossEntropyLoss()

	model = KANG(
		dataset.num_node_features,
		args.hidden_channels,
		dataset.num_classes,
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

	best_val_acc = 0
	early_stop_counter = 0
	best_epoch = -1
	best_model_path = f"./experiments/graph_classification/{generate_model_filename(args, is_multitask=False)}"

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
			data.y = data.y.long()  # Convert labels to LongTensor
			global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
			edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
			
			if use_amp:
				with autocast():
					out = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
					loss = criterion(out, data.y)
				epoch_loss += loss.item()
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			else:
				out = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
				loss = criterion(out, data.y)
				epoch_loss += loss.item()
				loss.backward()
				optimizer.step()

		# Validation evaluation
		model.eval()
		with torch.no_grad():
			if args.use_roc_auc:
				from sklearn.metrics import roc_auc_score
				all_probs = []
				all_targets = []
				for data in val_loader:
					data = data.to(device, non_blocking=True)
					data.y = data.y.long()
					global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
					edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
					
					if use_amp:
						with autocast():
							out = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
							probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
					else:
						out = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
						probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
					
					targets = data.y.cpu().numpy()
					all_probs.extend(probs)
					all_targets.extend(targets)
				val_acc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0
			else:
				correct = 0
				total = 0
				for data in val_loader:
					data = data.to(device, non_blocking=True)
					data.y = data.y.long()  # Convert labels to LongTensor
					global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
					edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
					
					if use_amp:
						with autocast():
							out = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
							pred = out.argmax(dim=1)
					else:
						out = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
						pred = out.argmax(dim=1)
					
					correct += (pred == data.y).sum().item()
					total += data.y.size(0)
				val_acc = correct / total if total > 0 else 0
		
		# Track training history if requested
		if return_history:
			train_losses.append(epoch_loss)
			val_metrics.append(val_acc)
		
		if val_acc > best_val_acc:
			best_epoch = epoch
			best_val_acc = val_acc
			torch.save(model.state_dict(), best_model_path)
			early_stop_counter = 0
		else:
			early_stop_counter += 1
		if early_stop_counter >= args.patience:
			break

		if epoch % args.log_freq == 0 or epoch == args.epochs - 1:
			metric_name = "Val ROC-AUC" if args.use_roc_auc else "Val Acc"
			print(f"Epoch {epoch:03d}: Train Loss: {epoch_loss:.4f}, {metric_name}: {val_acc:.4f}")


	# Load best model and evaluate on test set
	metric_name = "val roc auc" if args.use_roc_auc else "val acc"
	print(f"\nBest model was saved at epoch {best_epoch} with {metric_name}: {best_val_acc:.4f}")
	model.load_state_dict(torch.load(best_model_path))
	model.eval()
	correct = 0
	total = 0
	all_probs = []
	all_targets = []
	with torch.no_grad():
		for data in test_loader:
			data = data.to(device, non_blocking=True)
			global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
			edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
			
			if use_amp:
				with autocast():
					out = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
					pred = out.argmax(dim=1)
					# For ROC-AUC
					probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy() if out.shape[1] > 1 else torch.sigmoid(out).detach().cpu().numpy().flatten()
			else:
				out = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
				pred = out.argmax(dim=1)
				# For ROC-AUC
				probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy() if out.shape[1] > 1 else torch.sigmoid(out).detach().cpu().numpy().flatten()
			
			correct += (pred == data.y).sum().item()
			total += data.y.size(0)
			targets = data.y.cpu().numpy()
			all_probs.extend(probs)
			all_targets.extend(targets)
	test_acc = correct / total if total > 0 else 0

	# Compute ROC-AUC if possible
	test_roc_auc = None
	try:
		from sklearn.metrics import roc_auc_score
		if len(set(all_targets)) > 1:
			test_roc_auc = roc_auc_score(all_targets, all_probs)
	except ImportError:
		print("scikit-learn is not installed. ROC-AUC will not be computed.")

	print(f'Test Loss: Test Acc: {test_acc:.4f}')
	if test_roc_auc is not None:
		print(f'Test ROC-AUC: {test_roc_auc:.4f}')
	else:
		print('Test ROC-AUC: Not available (only one class present or scikit-learn not installed)')

	# Determine which test metric to return based on args.use_roc_auc preference
	test_metric = test_roc_auc if (args.use_roc_auc and test_roc_auc is not None) else test_acc

	# # ---------------------------
	# # Plot and save the training metrics:
	# # ---------------------------
	# epochs_range = list(range(len(train_losses)))
	# fig, ax1 = plt.subplots(figsize=(8, 6))
	# ax2 = ax1.twinx()
	# ax1.plot(epochs_range, train_losses, "r-", label="Train Loss")
	# ax2.plot(epochs_range, train_accs, "b-", label="Train Accuracy")
	# ax1.set_xlabel("Epoch")
	# ax1.set_ylabel("Loss", color="r")
	# ax2.set_ylabel("Accuracy", color="b")
	# ax1.tick_params(axis="y", labelcolor="r")
	# ax2.tick_params(axis="y", labelcolor="b")
	# fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
	# plt.savefig("./experiments/graph_classification/images/train_plot.png", dpi=300)
	# plt.close(fig)

	# # ---------------------------
	# # Plot and save the validation metrics:
	# # ---------------------------
	# fig, ax1 = plt.subplots(figsize=(8, 6))
	# ax2 = ax1.twinx()
	# ax1.plot(epochs_range, val_losses, "r-", label="Val Loss")
	# ax2.plot(epochs_range, val_accs, "b-", label="Val Accuracy")
	# ax1.set_xlabel("Epoch")
	# ax1.set_ylabel("Loss", color="r")
	# ax2.set_ylabel("Accuracy", color="b")
	# ax1.tick_params(axis="y", labelcolor="r")
	# ax2.tick_params(axis="y", labelcolor="b")
	# fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
	# plt.savefig("./experiments/graph_classification/images/val_plot.png", dpi=300)
	# plt.close(fig)

	if return_history:
		return best_val_acc, train_losses, val_metrics, test_metric  # Return validation score for optimization, test metric separately
	else:
		return best_val_acc, test_metric  # Return validation score for optimization, test metric separately

def graph_classification_multitask(args, return_history=False):
	"""Multi-task graph classification for ToxCast dataset"""
	import json
	from toxcast_dataset import ToxCastMultiTaskDataset, get_available_assays
	
	print("Multi-Task Graph Classification Mode")
	print("=" * 50)
	
	# Determine which assays to use
	if args.multitask_assays is not None:
		target_assays = args.multitask_assays
		print(f"Using specified assays: {target_assays}")
	else:
		# Use a default set of well-populated assays
		target_assays = [
			'TOX21_AhR_LUC_Agonist',
			'TOX21_Aromatase_Inhibition', 
			'TOX21_AutoFluor_HEK293_Cell_blue',
			'TOX21_p53_BLA_p3_ch1',
			'TOX21_p53_BLA_p4_ratio'
		]
		print(f"Using default assay set ({len(target_assays)} assays):")
		for i, assay in enumerate(target_assays):
			print(f"  {i+1}. {assay}")
	
	# Load multi-task dataset
	# Create a short hash
	assay_str = " ".join(sorted(target_assays))
	assay_hash = str(sum(ord(c) for c in assay_str) % 10**8)
	dataset_path = f'./dataset/TOXCAST_multitask_{assay_hash}'
	dataset = ToxCastMultiTaskDataset(
		root=dataset_path,
		target_columns=target_assays,
		use_global_features=args.use_global_features,
		use_3d_geo=args.use_3d_geo
	)
	
	print(f"Dataset loaded: {len(dataset)} molecules, {dataset.get_num_tasks()} tasks")
	if args.use_global_features:
		print("Using global molecular features")
	if args.use_3d_geo:
		print("Using 3D geometric features")
	
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
			task_weights = [task_weights_dict.get(assay, 1.0) for assay in target_assays]
			task_weights = torch.tensor(task_weights).to(device)
			print(f"Using custom task weights: {task_weights.tolist()}")
		except:
			print("Warning: Could not parse task weights, using uniform weights")
			task_weights = None

	# Handle class imbalance - for multi-task, we could compute per-task weights but keep simple for now
	criterion = nn.CrossEntropyLoss()
	if args.use_roc_auc:
		print("Note: Multi-task with ROC-AUC evaluation")

	# Create multi-task model - select based on single_head parameter
	if getattr(args, 'single_head', False):
		from src.KANG_MultiTask_SingleHead import KANG_MultiTask_SingleHead
		model = KANG_MultiTask_SingleHead(
			dataset.num_node_features,
			args.hidden_channels,
			len(target_assays),  # Number of tasks
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
		model = KANG_MultiTask(
			dataset.num_node_features,
			args.hidden_channels,
			len(target_assays),  # Number of tasks
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

	best_val_metric = 0
	early_stop_counter = 0
	best_epoch = -1
	best_model_path = f"./experiments/graph_classification/{generate_model_filename(args, is_multitask=True)}"

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
			labels = data.y.view(num_graphs, len(target_assays))
			
			if use_amp:
				with autocast():
					outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
					# Multi-task loss computation
					total_loss = 0
					valid_tasks = 0
					
					for task_idx in range(len(target_assays)):
						task_mask = ~torch.isnan(labels[:, task_idx])
						if task_mask.sum() > 0:
							task_pred = outputs[task_mask, task_idx]
							task_target = labels[task_mask, task_idx].long()
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
				
				for task_idx in range(len(target_assays)):
					task_mask = ~torch.isnan(labels[:, task_idx])
					if task_mask.sum() > 0:
						task_pred = outputs[task_mask, task_idx]
						task_target = labels[task_mask, task_idx].long()
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
			if args.use_roc_auc:
				# Multi-task ROC-AUC evaluation
				from sklearn.metrics import roc_auc_score
				task_aucs = []
				
				for task_idx in range(len(target_assays)):
					all_probs = []
					all_targets = []
					
					for data in val_loader:
						data = data.to(device, non_blocking=True)
						global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
						edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
						
						num_graphs = data.batch.max().item() + 1
						labels = data.y.view(num_graphs, len(target_assays))
						
						if use_amp:
							with autocast():
								outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
						else:
							outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
						
						# Get probabilities for this task
						task_mask = ~torch.isnan(labels[:, task_idx])
						if task_mask.sum() > 0:
							probs = torch.softmax(outputs[task_mask, task_idx], dim=1)[:, 1].cpu().numpy()
							targets = labels[task_mask, task_idx].cpu().numpy()
							all_probs.extend(probs)
							all_targets.extend(targets)
					
					if len(set(all_targets)) > 1:  # Need both classes
						task_auc = roc_auc_score(all_targets, all_probs)
						task_aucs.append(task_auc)
					else:
						task_aucs.append(0.0)
				
				val_metric = sum(task_aucs) / len(task_aucs)  # Average AUC across tasks
			else:
				# Multi-task accuracy evaluation
				task_accs = []
				
				for task_idx in range(len(target_assays)):
					correct = 0
					total = 0
					
					for data in val_loader:
						data = data.to(device, non_blocking=True)
						global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
						edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
						
						num_graphs = data.batch.max().item() + 1
						labels = data.y.view(num_graphs, len(target_assays))
						
						if use_amp:
							with autocast():
								outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
						else:
							outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
						
						# Get predictions for this task
						task_mask = ~torch.isnan(labels[:, task_idx])
						if task_mask.sum() > 0:
							pred = outputs[task_mask, task_idx].argmax(dim=1)
							target = labels[task_mask, task_idx].long()
							correct += (pred == target).sum().item()
							total += target.size(0)
					
					task_acc = correct / total if total > 0 else 0
					task_accs.append(task_acc)
				
				val_metric = sum(task_accs) / len(task_accs)  # Average accuracy across tasks

		# Track training history if requested
		if return_history:
			avg_epoch_loss = epoch_loss / total_valid_batches if total_valid_batches > 0 else 0
			train_losses.append(avg_epoch_loss)
			val_metrics.append(val_metric)
		
		if val_metric > best_val_metric:
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
			metric_name = "Val Multi-Task ROC-AUC" if args.use_roc_auc else "Val Multi-Task Acc"
			avg_epoch_loss = epoch_loss / total_valid_batches if total_valid_batches > 0 else 0
			print(f"Epoch {epoch:03d}: Train Loss: {avg_epoch_loss:.4f}, {metric_name}: {val_metric:.4f}")

	# Load best model and evaluate on test set
	print(f"\nBest model saved at epoch {best_epoch} with validation metric: {best_val_metric:.4f}")
	model.load_state_dict(torch.load(best_model_path))
	model.eval()
	
	# Test evaluation
	print("Evaluating on test set...")
	with torch.no_grad():
		if args.use_roc_auc:
			# Multi-task test ROC-AUC
			task_test_aucs = []
			
			for task_idx in range(len(target_assays)):
				all_probs = []
				all_targets = []
				
				for data in test_loader:
					data = data.to(device, non_blocking=True)
					global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
					edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
					
					num_graphs = data.batch.max().item() + 1
					labels = data.y.view(num_graphs, len(target_assays))
					
					if use_amp:
						with autocast():
							outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
					else:
						outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
					
					task_mask = ~torch.isnan(labels[:, task_idx])
					if task_mask.sum() > 0:
						probs = torch.softmax(outputs[task_mask, task_idx], dim=1)[:, 1].cpu().numpy()
						targets = labels[task_mask, task_idx].cpu().numpy()
						all_probs.extend(probs)
						all_targets.extend(targets)
				
				if len(set(all_targets)) > 1:
					task_auc = roc_auc_score(all_targets, all_probs)
					task_test_aucs.append(task_auc)
					print(f"  {target_assays[task_idx]}: AUC = {task_auc:.4f}")
				else:
					task_test_aucs.append(0.0)
					print(f"  {target_assays[task_idx]}: AUC = N/A (single class)")
			
			test_metric = sum(task_test_aucs) / len(task_test_aucs)
			print(f"Average Test ROC-AUC: {test_metric:.4f}")
		else:
			# Multi-task test accuracy
			task_test_accs = []
			
			for task_idx in range(len(target_assays)):
				correct = 0
				total = 0
				
				for data in test_loader:
					data = data.to(device, non_blocking=True)
					global_features = data.global_features if args.use_global_features and hasattr(data, 'global_features') else None
					edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
					
					num_graphs = data.batch.max().item() + 1
					labels = data.y.view(num_graphs, len(target_assays))
					
					if use_amp:
						with autocast():
							outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
					else:
						outputs = model(data.x, data.edge_index, data.batch, global_features, edge_attr)
					
					task_mask = ~torch.isnan(labels[:, task_idx])
					if task_mask.sum() > 0:
						pred = outputs[task_mask, task_idx].argmax(dim=1)
						target = labels[task_mask, task_idx].long()
						correct += (pred == target).sum().item()
						total += target.size(0)
				
				task_acc = correct / total if total > 0 else 0
				task_test_accs.append(task_acc)
				print(f"  {target_assays[task_idx]}: Acc = {task_acc:.4f}")
			
			test_metric = sum(task_test_accs) / len(task_test_accs)
			print(f"Average Test Accuracy: {test_metric:.4f}")

	print("=" * 50)
	print("Multi-Task Training Complete!")
	
	if return_history:
		return best_val_metric, train_losses, val_metrics, test_metric
	else:
		return best_val_metric, test_metric

def main():
	args = get_args()
	graph_classification(args, return_history=args.return_history)

if __name__ == "__main__":
	main()