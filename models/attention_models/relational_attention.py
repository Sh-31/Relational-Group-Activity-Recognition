import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv 

class RelationalUnit(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4):
        super(RelationalUnit, self).__init__(aggr='add')  
        self.gat = GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=0.5)

    def forward(self, x, edge_index):
        # https://github.com/pyg-team/pytorch_geometric/issues/2844
        b, n, _ = x.shape
        data_list = [Data(x=x_, edge_index=edge_index) for x_ in x] 
        batch = Batch.from_data_list(data_list)
        return self.gat(batch.x, batch.edge_index).view(b, n, -1)