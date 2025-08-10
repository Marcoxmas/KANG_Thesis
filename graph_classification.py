import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from toxcast_dataset import ToxCastGraphDataset
from hiv_dataset import HIVGraphDataset

from src.KANG import KANG
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
	return parser.parse_args()

def graph_classification(args, return_history=False):
	# Determine the dataset type based on the dataset name
	if args.dataset_name in ["MUTAG", "PROTEINS"]:
		dataset_path = f'./dataset/{args.dataset_name}'
		dataset = TUDataset(root=dataset_path, name=args.dataset_name)
		if args.use_global_features:
			print("Warning: Global features not supported for TUDataset. Ignoring --use_global_features flag.")
			args.use_global_features = False
	elif args.dataset_name == "HIV":
		dataset_path = f'./dataset/{args.dataset_name}'
		dataset = HIVGraphDataset(root=dataset_path, use_global_features=args.use_global_features)
		if args.use_global_features:
			print("Using global molecular features")
	elif args.dataset_name == "TOXCAST":
		dataset_path = f'./dataset/TOXCAST_{args.target_column}'
		dataset = ToxCastGraphDataset(root=dataset_path, target_column=args.target_column, use_global_features=args.use_global_features)
		if args.use_global_features:
			print("Using global molecular features")
	else:
		# Fallback for other datasets
		dataset_path = f'./dataset/{args.dataset_name}'
		dataset = ToxCastGraphDataset(root=dataset_path, target_column=args.dataset_name, use_global_features=args.use_global_features)
		if args.use_global_features:
			print("Using global molecular features")

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
	).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

	# Initialize mixed precision scaler if available
	scaler = GradScaler() if AMP_AVAILABLE and device.type == 'cuda' else None
	use_amp = scaler is not None

	best_val_acc = 0
	early_stop_counter = 0
	best_epoch = -1
	best_model_path = f"./experiments/graph_classification/gkan.pth"

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
			
			if use_amp:
				with autocast():
					out = model(data.x, data.edge_index, data.batch, global_features)
					loss = criterion(out, data.y)
				epoch_loss += loss.item()
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			else:
				out = model(data.x, data.edge_index, data.batch, global_features)
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
					
					if use_amp:
						with autocast():
							out = model(data.x, data.edge_index, data.batch, global_features)
							probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
					else:
						out = model(data.x, data.edge_index, data.batch, global_features)
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
					
					if use_amp:
						with autocast():
							out = model(data.x, data.edge_index, data.batch, global_features)
							pred = out.argmax(dim=1)
					else:
						out = model(data.x, data.edge_index, data.batch, global_features)
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
			
			if use_amp:
				with autocast():
					out = model(data.x, data.edge_index, data.batch, global_features)
					pred = out.argmax(dim=1)
					# For ROC-AUC
					probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy() if out.shape[1] > 1 else torch.sigmoid(out).detach().cpu().numpy().flatten()
			else:
				out = model(data.x, data.edge_index, data.batch, global_features)
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

def main():
	args = get_args()
	graph_classification(args, return_history=args.return_history)

if __name__ == "__main__":
	main()