import torch
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops

from torch_geometric.nn.models import MLP

from src.KANGConv import KANGConv
from src.KANLinear import KANLinear
from src.KAND import KAND
from src.global_features import get_global_feature_dim

class KANG_MultiTask_Regression(nn.Module):
	def __init__(self,
			in_channels, 
			hidden_channels, 
			num_tasks,
			num_layers=2, 
			grid_min=-1,
			grid_max=1,
			num_grids=2,
			dropout=0.0, 
			device='cpu', 
			aggr='mean',
			residuals=False,
			kan=True,
			linspace = False,
			trainable_grid = True,
			bsplines=False,
			use_global_features=False
		):
		super(KANG_MultiTask_Regression, self).__init__()
		self.dropout = dropout
		self.residuals = residuals
		self.use_global_features = use_global_features
		self.num_tasks = num_tasks
		self.convs = nn.ModuleList()

		# Calculate final layer input dimension
		final_input_dim = hidden_channels
		if self.use_global_features:
			final_input_dim += get_global_feature_dim()  # Add 200 global features

		# First Layer
		self.convs.append(
			KANGConv(
				in_channels, 
				hidden_channels,
				grid_min,
				grid_max,
				num_grids,
				device,
				aggr,
				kan,
				linspace = linspace,
				trainable_grid = trainable_grid,
				bsplines = bsplines
			)
		)

		# Subsequent Conv layers
		for _ in range(num_layers-1):
			self.convs.append(
				KANGConv(
					hidden_channels, 
					hidden_channels,
					grid_min,
					grid_max,
					num_grids,
					device,
					aggr,
					kan,
					linspace = linspace,
					trainable_grid = trainable_grid,
					bsplines = bsplines
				)
			)

		# Multi-task regression output heads
		self.task_heads = nn.ModuleList()
		for _ in range(num_tasks):
			if kan:
				if bsplines:
					self.task_heads.append(
						KANLinear(
							final_input_dim,
							1,  # Single value for regression
							grid_size=num_grids,
							grid_range=[grid_min, grid_max]
						)
					)
				else:
					self.task_heads.append(
						KAND(
							final_input_dim,
							1,  # Single value for regression
							grid_min,
							grid_max,
							num_grids,
							device=device,
							linspace = linspace,
							trainable_grid = trainable_grid
						)
					)
			else:
				self.task_heads.append(MLP([final_input_dim, 1]))

		self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_channels, elementwise_affine=False, bias=False) for _ in range(num_layers)])

	def forward(self, x, edge_index, batch=None, global_features=None):
		x.requires_grad_(True)
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = F.dropout(x, p=self.dropout, training=self.training)

		res = None
		for i, conv in enumerate(self.convs):
			if self.residuals and i > 0: res = x  
			x = conv(x, edge_index)
			x = self.layer_norms[i](x)
			x = F.dropout(x, p=self.dropout, training=self.training)
			if self.residuals and i > 0: x += res

		# 2. Readout layer
		if batch != None:
			x = global_mean_pool(x, batch)
		
		# 3. Concatenate global features if available
		if self.use_global_features and global_features is not None:
			# Reshape global features to [batch_size, num_features]
			# PyTorch Geometric concatenates global features, so we need to reshape
			batch_size = x.size(0)
			global_features = global_features.view(batch_size, -1)
			x = torch.cat([x, global_features], dim=1)
		
		# 4. Multi-task regression outputs
		task_outputs = []
		for task_head in self.task_heads:
			task_out = task_head(x)  # [batch_size, 1]
			task_outputs.append(task_out)
		
		# Stack outputs: [batch_size, num_tasks, 1]
		stacked_outputs = torch.stack(task_outputs, dim=1)
		
		# Remove the last dimension for easier handling: [batch_size, num_tasks]
		return stacked_outputs.squeeze(-1)
	
	def encode(self, x, edge_index, batch=None, global_features=None):
		"""
		Encode the graph into a shared molecular representation.
		Used for feature extraction and analysis.
		"""
		x.requires_grad_(True)

		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = F.dropout(x, p=self.dropout, training=self.training)

		res = None
		for i, conv in enumerate(self.convs):
			if self.residuals and i > 0: res = x  
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = conv(x, edge_index)
			x = self.layer_norms[i](x)
			if self.residuals and i > 0: x += res

		# 2. Readout layer
		if batch != None:
			x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
		
		# 3. Concatenate global features if available
		if self.use_global_features and global_features is not None:
			# Reshape global features to [batch_size, num_features]
			# PyTorch Geometric concatenates global features, so we need to reshape
			batch_size = x.size(0)
			global_features = global_features.view(batch_size, -1)
			x = torch.cat([x, global_features], dim=1)

		return x
	
	def predict_single_task(self, x, edge_index, batch=None, global_features=None, task_idx=0):
		"""
		Predict for a single task. Useful for single-task evaluation.
		"""
		encoded = self.encode(x, edge_index, batch, global_features)
		task_output = self.task_heads[task_idx](encoded)
		return task_output.squeeze(-1)  # Remove last dimension: [batch_size]
	
	def get_task_predictions(self, x, edge_index, batch=None, global_features=None, task_indices=None):
		"""
		Get predictions for specific tasks only.
		"""
		encoded = self.encode(x, edge_index, batch, global_features)
		
		if task_indices is None:
			task_indices = list(range(self.num_tasks))
		
		task_outputs = []
		for task_idx in task_indices:
			task_output = self.task_heads[task_idx](encoded)
			task_outputs.append(task_output)
		
		# Stack and squeeze: [batch_size, num_selected_tasks]
		stacked_outputs = torch.stack(task_outputs, dim=1)
		return stacked_outputs.squeeze(-1)
	
	def decode(self, z, edge_label_index):
		"""For compatibility with single-task models"""
		return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

	def decode_all(self, z):
		"""For compatibility with single-task models"""
		prob_adj = z @ z.t()
		return (prob_adj > 0).nonzero(as_tuple=False).t()
