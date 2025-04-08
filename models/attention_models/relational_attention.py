import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv 
from torch_geometric.utils import softmax

class RelationalUnit(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim=1024):
        super(RelationalUnit, self).__init__(aggr='add')  # Sum aggregation
        
        self.linear = nn.Linear(in_channels, hidden_dim)
        self.projection =  nn.Linear(hidden_dim, out_channels)

        self.mlp = nn.Sequential(  # MLP shared across all node pairs
           nn.Linear(hidden_dim, hidden_dim // 2),
           nn.ReLU(),
           nn.Dropout(0.5),
           nn.Linear(hidden_dim // 2, out_channels)
        )

    def forward(self, x, edge_index):
        """
        x: Node feature matrix [num_nodes, in_channels]
        edge_index: Connectivity matrix [2, num_edges]
        """
        batch_size, num_nodes, in_channels = x.shape
        x = x.view(batch_size * num_nodes, in_channels)

        x = self.linear(x)
        out = self.propagate(edge_index, x=x)

        out = out.view(batch_size, num_nodes, -1)
        return out


    def message(self, x_i, x_j, index):
        """
        x_i: Features of the receiving node
        x_j: Features of the sending node
        """
        z = self.mlp(x_i + x_j)
        e_ij = (self.projection(x_i) * z).sum(dim=-1, keepdim=True) # (b*num_edage, 1)
        
        # Normalize the attention scores with softmax over the destination nodes.
        a_ij = softmax(e_ij, index)  # (b*num_edage, 1)
        return a_ij * z
    
    def update(self, aggr_out):
        return aggr_out 


# class RelationalUnit(MessagePassing):
#     def __init__(self, in_channels, out_channels, heads=4):
#         super(RelationalUnit, self).__init__(aggr='add')  
#         self.gat = GATConv(
#             in_channels, 
#             out_channels, 
#             heads=heads, 
#             concat=False, 
#             negative_slope=0.2,
#             dropout=0.6
#         )

#     def forward(self, x, edge_index):
#         # https://github.com/pyg-team/pytorch_geometric/issues/2844
#         b, n, _ = x.shape
#         data_list = [Data(x=x_, edge_index=edge_index) for x_ in x] 
#         batch = Batch.from_data_list(data_list)
#         return self.gat(batch.x, batch.edge_index).view(b, n, -1)