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

    def forward(self):
        return


def drop_edge(edge_index: Tensor, drop_rate: float = 0.5) -> Tensor:
    dense_matrix = torch.squeeze(torch_geometric.utils.to_dense_adj(edge_index)).to(device=edge_index.device)
    dense_matrix = dense_matrix * (1 - drop_rate)
    dense_matrix = torch.bernoulli(dense_matrix).long()
    # use upper triangle
    adj = dense_matrix.numpy()
    adj = np.triu(adj)
    adj += adj.T
    dense_matrix = torch.from_numpy(adj).to(device=edge_index.device)
    edge_final = torch_geometric.utils.dense_to_sparse(torch.squeeze(dense_matrix, dim=0))[0]
    return edge_final


node_A = torch.LongTensor(
    [[0, 1, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0],
     [1, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0]])
edge_index = torch_geometric.utils.dense_to_sparse(torch.squeeze(node_A, dim=0))[0]

drop_edge(edge_index)
