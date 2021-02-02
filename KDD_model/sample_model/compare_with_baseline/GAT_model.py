import torch
import torch_geometric
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj


class GATModel(torch.nn.Module):
    def __init__(self, feature_num, output_num):
        super(GATModel, self).__init__()
        self.conv1 = torch_geometric.nn.GATConv(feature_num, 8, heads=8, dropout=0.6)
        self.conv2 = torch_geometric.nn.GATConv(8 * 8, output_num, heads=1, concat=False,
                                                dropout=0.6)

    def forward(self, x: Tensor, edge_index: Adj):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return x

# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
