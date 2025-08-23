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

class KANG_MultiTask_Regression_SingleHead(nn.Module):
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
            use_global_features=False,
            use_self_loops=True
        ):
        super(KANG_MultiTask_Regression_SingleHead, self).__init__()
        self.dropout = dropout
        self.residuals = residuals
        self.use_global_features = use_global_features
        self.use_self_loops = use_self_loops
        self.num_tasks = num_tasks
        self.convs = nn.ModuleList()
        self._grid_min = grid_min
        self._grid_max = grid_max
        self._num_grids = num_grids
        self._device = device
        self._linspace = linspace
        self._trainable_grid = trainable_grid
        final_input_dim = hidden_channels
        if self.use_global_features:
            final_input_dim += get_global_feature_dim()
        self.convs.append(KANGConv(
            in_channels, hidden_channels, grid_min, grid_max, num_grids, device, aggr, kan,
            linspace=linspace, trainable_grid=trainable_grid, bsplines=bsplines
        ))
        for _ in range(num_layers-1):
            self.convs.append(KANGConv(
                hidden_channels, hidden_channels, grid_min, grid_max, num_grids, device, aggr, kan,
                linspace=linspace, trainable_grid=trainable_grid, bsplines=bsplines
            ))
        # Single head for all tasks: output shape [batch_size, num_tasks]
        if kan:
            if bsplines:
                self.head = KANLinear(final_input_dim, num_tasks, grid_size=num_grids, grid_range=[grid_min, grid_max])
            else:
                self.head = KAND(final_input_dim, num_tasks, grid_min, grid_max, num_grids, device=device, linspace=linspace, trainable_grid=trainable_grid)
        else:
            self.head = MLP([final_input_dim, num_tasks])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_channels, elementwise_affine=False, bias=False) for _ in range(num_layers)])

    def forward(self, x, edge_index, batch=None, global_features=None, edge_attr=None):
        x.requires_grad_(True)
        if self.use_self_loops:
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
        if batch is not None:
            x = global_mean_pool(x, batch)
        if self.use_global_features and global_features is not None:
            batch_size = x.size(0)
            global_features = global_features.view(batch_size, -1)
            x = torch.cat([x, global_features], dim=1)
        out = self.head(x)  # [batch_size, num_tasks]
        return out

    def encode(self, x, edge_index, batch=None, global_features=None, edge_attr=None):
        """
        Encode the graph into a shared molecular representation.
        Used for feature extraction and analysis.
        """
        x.requires_grad_(True)

        # Add self-loops and extend edge_attr for encoding
        if self.use_self_loops:
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
        if batch is not None:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # 3. Concatenate global features if available
        if self.use_global_features and global_features is not None:
            # Reshape global features to [batch_size, num_features]
            # PyTorch Geometric concatenates global features, so we need to reshape
            batch_size = x.size(0)
            global_features = global_features.view(batch_size, -1)
            x = torch.cat([x, global_features], dim=1)

        return x
    
    def predict_single_task(self, x, edge_index, batch=None, global_features=None, task_idx=0, edge_attr=None):
        """
        Predict for a single task. Useful for single-task evaluation.
        """
        encoded = self.encode(x, edge_index, batch, global_features, edge_attr)
        out = self.head(encoded)  # [batch_size, num_tasks]
        return out[:, task_idx]  # [batch_size]
    
    def get_task_predictions(self, x, edge_index, batch=None, global_features=None, task_indices=None, edge_attr=None):
        """
        Get predictions for specific tasks only.
        """
        encoded = self.encode(x, edge_index, batch, global_features, edge_attr)
        out = self.head(encoded)  # [batch_size, num_tasks]
        
        if task_indices is None:
            task_indices = list(range(self.num_tasks))
        
        return out[:, task_indices]  # [batch_size, num_selected_tasks]
    
    def decode(self, z, edge_label_index):
        """For compatibility with single-task models"""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        """For compatibility with single-task models"""
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
