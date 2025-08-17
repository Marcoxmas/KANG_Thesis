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

class KANG(nn.Module):
	def __init__(self,
			in_channels, 
			hidden_channels, 
			out_channels, 
			num_layers, 
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
		super(KANG, self).__init__()
		self.dropout		= dropout
		self.residuals	= residuals
		self.use_global_features = use_global_features
		self.convs			= nn.ModuleList()

		# Store config for edge encoder lazy init
		self._grid_min = grid_min
		self._grid_max = grid_max
		self._num_grids = num_grids
		self._device = device
		self._linspace = linspace
		self._trainable_grid = trainable_grid

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

		# Readout Layer
		if kan:
			if bsplines:
				self.out_layer = KANLinear(
					final_input_dim,
					out_channels,
					grid_size=num_grids,
					grid_range=[grid_min, grid_max]
				)
			else:
				self.out_layer = KAND(
					final_input_dim,
					out_channels,
					grid_min,
					grid_max,
					num_grids,
					device=device,
					linspace = linspace,
					trainable_grid = trainable_grid
				)
		else:
			self.out_layer = MLP([final_input_dim, out_channels])

		self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_channels, elementwise_affine=False, bias=False) for _ in range(num_layers)])

	def forward(self, x, edge_index, batch=None, global_features=None, edge_attr=None):
		x.requires_grad_(True)
		# Add self-loops and extend edge_attr for gated message passing
		if edge_attr is not None:
			edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0.0, num_nodes=x.size(0))
		else:
			edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

		x = F.dropout(x, p=self.dropout, training=self.training)

		res = None
		for i, conv in enumerate(self.convs):
			if self.residuals and i > 0: res = x  
			x = conv(x, edge_index, edge_attr)
			x = self.layer_norms[i](x)
			x = F.dropout(x, p=self.dropout, training=self.training)
			if self.residuals and i > 0: x += res

		# 2. Readout layer
		if batch != None:
			x = global_mean_pool(x, batch)
		
		# 3. Concatenate global features if available
		if self.use_global_features and global_features is not None:
			# More efficient reshaping - avoid division operation
			batch_size = x.size(0)
			global_features = global_features.view(batch_size, -1)
			x = torch.cat([x, global_features], dim=1)
		
		x = self.out_layer(x)

		return x
	
	def encode(self, x, edge_index, batch=None, global_features=None, edge_attr=None):
		x.requires_grad_(True)

		# Add self-loops and extend edge_attr for encoding
		if edge_attr is not None:
			edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0.0, num_nodes=x.size(0))
		else:
			edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

		x = F.dropout(x, p=self.dropout, training=self.training)

		res = None
		for i, conv in enumerate(self.convs):
			if self.residuals and i > 0: res = x  
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = conv(x, edge_index, edge_attr)
			x = self.layer_norms[i](x)
			if self.residuals and i > 0: x += res

		# 2. Readout layer
		if batch != None:
			x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
		
		# 3. Concatenate global features if available
		if self.use_global_features and global_features is not None:
			# More efficient reshaping - avoid division operation
			batch_size = x.size(0)
			global_features = global_features.view(batch_size, -1)
			x = torch.cat([x, global_features], dim=1)

		return x
	
	def decode(self, z, edge_label_index):
		return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)  # product of a pair of nodes on each edge

	def decode_all(self, z):
		prob_adj = z @ z.t()
		return (prob_adj > 0).nonzero(as_tuple=False).t()