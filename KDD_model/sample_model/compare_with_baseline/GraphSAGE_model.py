import torch
import torch_geometric
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj


class SAGEModel(torch.nn.Module):
    def __init__(self, feature_num, output_num):
        super(SAGEModel, self).__init__()
        self.conv1 = torch_geometric.nn.SAGEConv(feature_num, 16)
        self.conv2 = torch_geometric.nn.SAGEConv(16, output_num)

    def forward(self, x: Tensor, edge_index: Adj):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
