import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class RelationalUnit(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super(RelationalUnit, self).__init__(aggr='add')  # Sum aggregation

        self.mlp = nn.Sequential(  # MLP shared across all node pairs
           nn.Linear(2 * in_channels, hidden_dim),
           nn.LayerNorm(hidden_dim),
           nn.ReLU(),
           nn.Dropout(0.5),
           nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x, edge_index):
        """
        x: Node feature matrix [num_nodes, in_channels]
        edge_index: Connectivity matrix [2, num_edges]
        """
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        """
        x_i: Features of the receiving node
        x_j: Features of the sending node
        """
        z = torch.cat([x_i, x_j], dim=-1)  # Concatenate player with it's neighbor
        return self.mlp(z)  
    
    def update(self, aggr_out):
        return aggr_out 

    # def aggregate(self, inputs, index): # need torch 2.6.*
    #     """
    #     Aggregate messages using summation.
    #     """
    #     return torch.scatter_add(inputs, 0, index, dim_size=inputs.size(0))

