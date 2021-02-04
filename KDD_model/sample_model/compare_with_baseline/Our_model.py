import torch
import math
import random
import torch_geometric
import torch_sparse
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter


class NewSampleModelLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, k_hop: int = 3, sample_num: int = 4, **kwargs):
        super(NewSampleModelLayer, self).__init__(aggr='add', **kwargs)
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, 64, add_self_loops=add_self_loops, normalize=normalize,
                                                bias=bias)
        self.conv2 = torch_geometric.nn.GCNConv(64, 16, add_self_loops=add_self_loops, normalize=normalize, bias=bias)

        # self attributes
        self.k_hop_nerghbor = None
        self.edge_weight = None
        self.deg_inv_sqrt = None

        self.k_hop = k_hop
        self.sample_num = sample_num
        self.add_self_loops = add_self_loops

        # parameters
        self.probablity_coefficient = Parameter(Tensor(32, 1))
        self.weight = Parameter(Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # parameters reset
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.glorot(self.probablity_coefficient)
        self.glorot(self.weight)
        self.zeros(self.bias)

        # self attributes reset
        self.k_hop_nerghbor = None
        self.edge_weight = None
        self.deg_inv_sqrt = None

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def zeros(self, tensor):
        if tensor is not None:
            tensor.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0.4, add_edge_rate: float = 1,
                mask: Tensor = None):
        # adj generate
        if self.k_hop_nerghbor is None:
            self.k_hop_nerghbor = list()
            self.fill_k_hop_with_sparse_matrix(edge_index, x.size(0))

        # add self loop
        if self.add_self_loops:
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # normalize
        if self.edge_weight is None:
            row, col = edge_index
            deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
            self.deg_inv_sqrt = deg.pow(-0.5)
            self.edge_weight = self.deg_inv_sqrt[row] * self.deg_inv_sqrt[col]
            # print(self.edge_weight.size())

        # for eval
        if not self.training:
            # message propagate
            feature_result = self.propagate(edge_index, size=None, x=x, edge_weight=self.edge_weight)

        else:
            # generate GCN embedding of node
            half_embedding = self.conv1(x, edge_index)
            node_embedding = self.conv2(half_embedding, edge_index)

            # normal propagate
            feature_result = self.propagate(edge_index, size=None, x=x, edge_weight=self.edge_weight,
                                            drop_rate=drop_rate)

            # node-wise operation
            for i in range(x.size(0)):
                # mask judgment
                if mask is not None:
                    if not mask[i]:
                        continue

                if random.random() > add_edge_rate:
                    continue

                candidate_set = self.generate_candidate_set_for_node(i)

                feature_mix = torch.tensor([]).to(device=x.device)
                candidate_feature = torch.tensor([]).to(device=x.device)

                if candidate_set.size == 0:
                    continue

                # generate candidate feature mix
                for k in candidate_set:
                    item = torch.cat([node_embedding[i], node_embedding[k]])
                    candidate_feature = torch.cat([candidate_feature, torch.unsqueeze(x[k], 0)])
                    feature_mix = torch.cat([feature_mix, torch.unsqueeze(item, 0)])

                add_rate = torch.mm(feature_mix, self.probablity_coefficient).view(1, -1)
                add_rate = torch.cat([add_rate, torch.zeros_like(add_rate)], 0)
                add_rate = add_rate.softmax(dim=0)

                # gumbel softmax
                add_rate = F.gumbel_softmax(add_rate, tau=0.1, hard=False, dim=0)
                add_rate = torch.unsqueeze(add_rate[0], 0)

                nor_add = add_rate.clone()
                for l in range(candidate_set.size):
                    nor_add[0][l] /= (
                            self.deg_inv_sqrt[i] * (
                        self.deg_inv_sqrt[candidate_set[l]] if candidate_set[l] != -1 else 1))

                add_feature = torch.mm(nor_add, candidate_feature)
                feature_result[i] += torch.squeeze(add_feature)

        # transform feature dimension
        out = torch.matmul(feature_result, self.weight)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor, drop_rate: float = 0) -> Tensor:
        # random drop feature
        # expand other dimension
        x_j = x_j.mul(torch.bernoulli(torch.ones_like(x_j) - drop_rate).long()) * (1 / (1 - drop_rate))
        return edge_weight.view(-1, 1) * x_j

    def fill_k_hop_with_sparse_matrix(self, edge_index: Adj, node_num: int):
        k_hop_neighbor = list(set() for i in range(node_num))
        one_hop_neighbor = list(set() for i in range(node_num))

        for i in range(edge_index.size(-1)):
            one_hop_neighbor[edge_index[0][i]].add(int(edge_index[1][i]))

        two_hop_neighbor = torch_sparse.spspmm(edge_index, torch.ones(edge_index.size(-1)).to(device=edge_index.device),
                                               edge_index, torch.ones(edge_index.size(-1)).to(device=edge_index.device),
                                               node_num, node_num, node_num)[0]
        for i in range(two_hop_neighbor.size(-1)):
            k_hop_neighbor[two_hop_neighbor[0][i]].add(int(two_hop_neighbor[1][i]))

        cur_matrix = two_hop_neighbor

        for k in range(2, self.k_hop):
            cur_matrix = \
                torch_sparse.spspmm(cur_matrix, torch.ones(cur_matrix.size(-1)).to(device=edge_index.device),
                                    edge_index,
                                    torch.ones(edge_index.size(-1)).to(device=edge_index.device),
                                    node_num, node_num, node_num)[0]
            for i in range(cur_matrix.size(-1)):
                k_hop_neighbor[cur_matrix[0][i]].add(int(cur_matrix[1][i]))

        # self.k_hop_nerghbor = list()
        for i in range(node_num):
            self.k_hop_nerghbor.append(np.array(list(k_hop_neighbor[i] - one_hop_neighbor[i] - {i})))
        return

    def generate_candidate_set_for_node(self, node: int) -> np.ndarray:
        if self.sample_num > self.k_hop_nerghbor[node].size:
            return self.k_hop_nerghbor[node]
        else:
            return np.random.choice(self.k_hop_nerghbor[node], self.sample_num, replace=False)


# 2 layer class
class TwoLayerModel(torch.nn.Module):
    def __init__(self, feature_num, output_num):
        super(TwoLayerModel, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(feature_num, 32)
        self.conv2 = NewSampleModelLayer(32, output_num)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0.1, add_rate: float = 0.1,
                mask: Tensor = None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, drop_rate, add_rate, mask=mask)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
