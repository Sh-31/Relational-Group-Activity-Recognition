import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv 
from torch_geometric.utils import softmax

class BatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        # x is expected to be [B, N, C] where C is the feature dimension
        # Transpose to [B, C, N]
        x_permuted = x.transpose(1, 2)
        # Apply BatchNorm1d on the channel dimension (dim=1)
        x_bn = self.bn(x_permuted)
        # Transpose back to [B, N, C]
        return x_bn.transpose(1, 2)


class RelationalUnit(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden_dim=1024):
        super(RelationalUnit, self).__init__(aggr='add')  # Sum aggregation
          
        self.proj  = nn.Linear(in_channels,  out_channels)
        self.query = nn.Linear(in_channels,  hidden_dim)
        self.key   = nn.Linear(in_channels,  hidden_dim)
        
        self.attention_dropout = nn.Dropout(0.35)
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_channels, hidden_dim),
            BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_channels)
        )
       
        # gate mechanism to fuse new message and residual connection.
        self.keep_gate = nn.Sequential(   # keep gate
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Sigmoid()
        )

        self.forget_gate = nn.Sequential( # forget gate
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Sigmoid()
        )

        self.update_gate = nn.Sequential( # update gate
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Sigmoid()
        )

        self.act_drop = nn.Sequential(
             nn.ReLU(),
             nn.Dropout(0.25)
        )    


    def forward(self, x, edge_index):
        """
        x: Node feature matrix [num_nodes, in_channels]
        edge_index: Connectivity matrix [2, num_edges]
        """
        out = self.propagate(edge_index, x=x)
        out = self.attention_dropout(out)

        gate_input = torch.cat([x, out], dim=-1)
        
        k_gate = self.keep_gate(gate_input)
        f_gate = self.forget_gate(gate_input)
        u_gate = self.update_gate(gate_input)

        x = self.proj(x) # to align the dimension

        out = k_gate * x + f_gate * out
        out = u_gate * out

        return self.act_drop(out)

    def message(self, x_i, x_j, index, ptr, size_i):
        """
        x_i: Features of the receiving node
        x_j: Features of the sending node
        """
        query= self.query(x_i)
        key = self.key(x_j)       
        value = self.mlp(torch.cat([x_i , x_j], dim=-1))

        e_ij = (query* key).sum(dim=-1) / (key.size(-1) ** 0.5)    # (b, num_edage)
        a_ij = softmax(e_ij, index, ptr, num_nodes=size_i, dim=-1)  # Normalize the attention scores with softmax over the destination nodes.
        
        # print(f"x_j.shape: {x_j.shape}")
        # print(f"e_ij.shape: {e_ij.shape}")
        # print(f"a_ij.shape: {a_ij.shape}")
        # print(f"z.shape: {z.shape}")
        # print(a_ij)
        
        return a_ij.unsqueeze(-1) * value
    
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