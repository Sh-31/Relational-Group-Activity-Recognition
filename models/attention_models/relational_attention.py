import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax

class RelationalUnit(MessagePassing):
    """
    A single Relational Unit combining multi-head attention and a Feed-Forward Network (FFN).
    This unit processes node features in a graph-like structure, incorporating
    residual connections and Layer Normalization, similar to a Transformer block.
    """
    def __init__(self, in_channels, out_channels, num_heads=8, hidden_size=1024, dropout_rate=0.5):
        super(RelationalUnit, self).__init__(aggr='add') # Using 'add' aggregation for messages

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads 
        self.scale = self.head_dim ** 0.5

        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(2 * in_channels, in_channels)

        self.ln1 = nn.LayerNorm(in_channels)
        self.dr1 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_channels),
        )

        self.ln2 = nn.LayerNorm(out_channels)
        self.dr2 = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        
        x_att = self.propagate(edge_index, x=x)
        
        x_att = x + self.dr1(x_att) 
        x_att = self.ln1(x_att) 

        x_ffn = self.ffn(x_att)

        out = x_att + self.dr2(x_ffn) 
        out = self.ln2(out)

        return out

    def message(self, x_i, x_j, index, ptr, size_i):
        """
        x_i: Features of the receiving node
        x_j: Features of the sending node
        """
        batch, edges, _ = x_i.shape
        
        query= self.query(x_i).view(batch, edges, self.num_heads, self.head_dim).transpose(1, 2) # (batch, num_heads, num_edage, head_dim)
        key = self.key(x_j).view(batch, edges, self.num_heads, self.head_dim).transpose(1, 2)    # (batch, num_heads, num_edage, head_dim)  
        value = self.value(torch.cat([x_i , x_j], dim=-1)).view(batch, edges, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, num_heads, num_edage, head_dim)  

        e_ij = (query * key).sum(dim=-1) / self.scale # (batch, num_heads, num_edage)
        a_ij = softmax(e_ij, index, ptr, num_nodes=size_i, dim=-1)  # Normalize the attention scores with softmax over the destination nodes.

        return (a_ij.unsqueeze(-1) * value).view(batch, edges, self.num_heads * self.head_dim)
    
    def update(self, aggr_out):
        return aggr_out 
