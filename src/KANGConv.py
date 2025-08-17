from typing import Union, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils import spmm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.nn.models import MLP

from src.KANLinear import KANLinear
from src.KAND import KAND

class KANGConv(MessagePassing):
	def __init__(
			self, 
			in_channels, 
			out_channels, 
			grid_min=-1,
			grid_max=1,
			num_grids=2,
			device='cpu', 
			aggr='mean',
			kan=True,
			linspace = False,
			trainable_grid = True,
			bsplines=False,
			**kwargs
		):

		kwargs.setdefault('aggr', aggr) 
		super().__init__(**kwargs)

		self.device = device
		self.in_channels = in_channels
		self.out_channels = out_channels
		# Save KAND config for lazy init of message KAND
		self._grid_min = grid_min
		self._grid_max = grid_max
		self._num_grids = num_grids
		self._linspace = linspace
		self._trainable_grid = trainable_grid

		if kan:
			if bsplines:
				self.nn = KANLinear(
					in_channels,
					out_channels,
					grid_size=num_grids,
					grid_range=[grid_min, grid_max]
				)
			else:
				self.nn = KAND(
					in_channels,
					out_channels,
					grid_min,
					grid_max,
					num_grids,
					device=device,
					linspace = linspace,
					trainable_grid = trainable_grid
				)
		else:
			self.nn = MLP([in_channels, out_channels])

		# The input x is expected to already have in_channels dimensions
		self.skip = nn.Identity()

		# Content message via KAND over [x_j, edge_attr] -> in_channels
		# Lazy initialization when we know the edge_attr dimension
		self.msg_kand = None

		self.reset_parameters()

	def reset_parameters(self):
		super().reset_parameters()

	def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: Tensor = None, size: Size = None,) -> Tensor:
		if isinstance(x, Tensor):
			x = (x, x)
		
		agg = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
		res = self.skip(x[0]) + agg
		return self.nn(res)

	def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor = None) -> Tensor:
		# Build content message from [x_j, edge_attr]; if edge_attr missing, use zeros
		if edge_attr is None:
			edge_attr = torch.zeros(x_j.size(0), 1, device=x_j.device, dtype=x_j.dtype)
		# Lazy-init KAND with correct input size if not pre-initialized
		if self.msg_kand is None:
			in_dim = self.in_channels + edge_attr.size(1)
			self.msg_kand = KAND(
				in_dim,
				self.in_channels,
				self._grid_min,
				self._grid_max,
				self._num_grids,
				device=self.device,
				linspace=self._linspace,
				trainable_grid=self._trainable_grid,
				use_layernorm=False
			)
			self.msg_kand = self.msg_kand.to(x_j.device)
		z = torch.cat([x_j, edge_attr], dim=-1)
		m = self.msg_kand(z)
		return m

	def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
		if isinstance(adj_t, SparseTensor):
			adj_t = adj_t.set_value(None, layout=None)
		return spmm(adj_t, x[0], reduce=self.aggr)

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}(nn={self.nn})'