import torch
import torch_geometric
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj


class DropEdgeModel(torch.nn.Module):
    def __init__(self, feature_num, output_num):
        super(DropEdgeModel, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(feature_num, 16)
        self.conv2 = torch_geometric.nn.GCNConv(16, output_num)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0.3):
        if self.training:
            adj = drop_edge(edge_index, drop_rate)
            x = x * (1 / (1 - drop_rate))
        else:
            adj = edge_index
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


def drop_edge(edge_index: Tensor, drop_rate: float = 0.5) -> Tensor:
    dense_matrix = torch.squeeze(torch_geometric.utils.to_dense_adj(edge_index)).to(device=edge_index.device)
    dense_matrix = dense_matrix * (1 - drop_rate)
    dense_matrix = torch.bernoulli(dense_matrix).long()
    # use upper triangle
    adj = dense_matrix.cpu().numpy()
    adj = np.triu(adj)
    adj += adj.T
    dense_matrix = torch.from_numpy(adj).to(device=edge_index.device)
    edge_final = torch_geometric.utils.dense_to_sparse(torch.squeeze(dense_matrix, dim=0))[0]
    return edge_final
