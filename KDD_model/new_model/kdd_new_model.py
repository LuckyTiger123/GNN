import math
import torch
import torch_geometric
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Parameter


class FeatureSwitchLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True,
                 **kwargs):
        super(FeatureSwitchLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, activate_signal: Tensor = None, drop_rate: float = 0.1,
                edge_weight: OptTensor = None) -> Tensor:

        # add self loop
        if self.add_self_loops:
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # normalize
        if self.normalize:
            row, col = edge_index
            deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            edge_weight = None

        # message propagate
        out = self.propagate(edge_index, size=None, x=x, edge_weight=edge_weight, activate_signal=activate_signal,
                             drop_rate=drop_rate)

        # transform feature dimension
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor, activate_signal: Tensor, drop_rate: float) -> Tensor:
        # activate edge feature signal
        if activate_signal is None:
            activate_signal = torch.ones(x_j.size(), device=x_j.device)
            activate_signal = activate_signal - drop_rate
            activate_signal = torch.bernoulli(activate_signal).long()

        x_j = x_j.mul(activate_signal)
        return edge_weight.view(-1, 1) * x_j


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def add_and_remove_featureEdge(edge_index: Adj, feature_num: int, add_rate: float = 0.1, remove_rate: float = 0.1) -> (
        Tensor, Tensor):
    dense_matrix = torch.squeeze(torch_geometric.utils.to_dense_adj(edge_index))
    feature_edge_list = list()
    edge_final = torch.zeros(dense_matrix.size())
    for i in range(feature_num):
        item = dense_matrix.clone()
        item = item * (1 - add_rate) + add_rate
        item = torch.bernoulli(item)
        item = item * (1 - remove_rate)
        item = torch.bernoulli(item)
        feature_edge_list.append(item)
        edge_final += item
    sparse_A = torch_geometric.utils.dense_to_sparse(torch.squeeze(edge_final.long(), dim=0))[0]
    edge_activate = torch.zeros([sparse_A.size(1), feature_num])
    for i in range(sparse_A.size(1)):
        for j in range(feature_num):
            if feature_edge_list[j][sparse_A[0][i]][sparse_A[1][i]] != 0:
                edge_activate[i][j] = 1
    return sparse_A, edge_activate.long()


def add_edge(edge_index: Adj, add_rate: float = 0.1) -> Tensor:
    dense_matrix = torch.squeeze(torch_geometric.utils.to_dense_adj(edge_index)).to(device=edge_index.device)
    dense_matrix = dense_matrix * (1 - add_rate) + add_rate
    edge_final = torch_geometric.utils.dense_to_sparse(torch.squeeze(torch.bernoulli(dense_matrix).long(), dim=0))[0]
    return edge_final
