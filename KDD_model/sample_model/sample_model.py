import gc
import torch
import math
import random
import torch_geometric
import time
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter


class SampleModel(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, k_hop: int = 3, sample_num: int = 4, **kwargs):
        super(SampleModel, self).__init__(aggr='add', **kwargs)

        self.conv1 = torch_geometric.nn.GCNConv(in_channels, 64, add_self_loops=add_self_loops, normalize=normalize,
                                                bias=bias)
        self.conv2 = torch_geometric.nn.GCNConv(64, 16, add_self_loops=add_self_loops, normalize=normalize, bias=bias)

        self.adj_list = None

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
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.glorot(self.probablity_coefficient)
        self.glorot(self.weight)
        self.zeros(self.bias)

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def zeros(self, tensor):
        if tensor is not None:
            tensor.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0.4, add_edge_rate: float = 0.4,
                candidate_type: int = 0, mask: Tensor = None):
        # add self loop
        if self.add_self_loops:
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # normalize
        row, col = edge_index
        deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # for eval
        if not self.training:
            # message propagate
            feature_result = self.propagate(edge_index, size=None, x=x, edge_weight=edge_weight)

        else:
            # adjacency list
            self.adj_list = list(list() for i in range(x.size(0)))
            self.fill_adj_list(edge_index)

            # generate GCN embedding of node
            half_embedding = self.conv1(x, edge_index)
            node_embedding = self.conv2(half_embedding, edge_index)

            # normal propagate
            feature_result = self.propagate(edge_index, size=None, x=x, edge_weight=edge_weight, drop_rate=drop_rate)

            # node-wise operation
            for i in range(x.size(0)):
                # mask judgment
                if mask is not None:
                    if not mask[i]:
                        continue

                self_feature = torch.zeros([1, x.size(1)]).to(device=x.device)

                if candidate_type == 0:
                    candidate_set = self.roughly_generate_candidate_set(i, add_rate=add_edge_rate)
                else:
                    candidate_set = self.generate_candidate_set(i, add_rate=add_edge_rate)

                # add -1 to make the candidate set have sample_num elements.
                if len(candidate_set) < self.sample_num:
                    for j in range(self.sample_num - len(candidate_set)):
                        candidate_set.append(-1)

                feature_mix = torch.tensor([]).to(device=x.device)
                candidate_feature = torch.tensor([]).to(device=x.device)

                # generate candidate feature mix
                for k in candidate_set:
                    item = torch.cat(
                        [node_embedding[i], node_embedding[k] if k != -1 else torch.zeros_like(node_embedding[i])])
                    candidate_feature = torch.cat([candidate_feature,
                                                   torch.unsqueeze(x[k] if k != -1 else torch.zeros_like(x[i]), 0)])
                    feature_mix = torch.cat([feature_mix, torch.unsqueeze(item, 0)])

                # candidate feature drop
                dropout_matrix = torch.ones_like(candidate_feature, device=x.device) - drop_rate
                candidate_feature = candidate_feature.mul(torch.bernoulli(dropout_matrix).long())
                del dropout_matrix

                add_rate = torch.mm(feature_mix, self.probablity_coefficient).view(1, -1)
                add_rate = torch.cat([add_rate, torch.zeros_like(add_rate)], 0)

                # gumbel softmax
                add_rate = F.gumbel_softmax(add_rate, tau=0.1, hard=False, dim=0)
                add_rate = torch.unsqueeze(add_rate[0], 0)

                nor_add = add_rate.clone()
                for l in range(self.sample_num):
                    # nor_add[0][l] /= (
                    #         pow(len(self.adj_list[i]), 0.5) * (
                    #     pow(len(self.adj_list[candidate_set[l]]), 0.5) if candidate_set[l] != -1 else 1))

                    nor_add[0][l] /= (
                            deg_inv_sqrt[i] * (deg_inv_sqrt[candidate_set[l]] if candidate_set[l] != -1 else 1))

                add_feature = torch.mm(nor_add, candidate_feature)
                self_feature += add_feature
                # feature_result = torch.cat([feature_result, self_feature], 0)
                feature_result[i] += torch.squeeze(self_feature, dim=0)
                # call gc
                gc.collect()
                # print('{} node has caled.'.format(i))
                # print('--------------------------------------')

        # transform feature dimension
        out = torch.matmul(feature_result, self.weight)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor, drop_rate: float = 0) -> Tensor:
        # random drop feature
        x_j = x_j.mul(torch.bernoulli(torch.ones_like(x_j) - drop_rate).long())
        return edge_weight.view(-1, 1) * x_j

    def fill_adj_list(self, edge_index: Adj):
        for i in range(edge_index.size(1)):
            self.adj_list[edge_index[1][i]].append(int(edge_index[0][i]))

    def generate_candidate_set(self, node: int, add_rate: float = 0.1) -> list:
        one_hop_neighbor = set(self.adj_list[node])
        k_hop_candidates = one_hop_neighbor.copy()
        next_queue = one_hop_neighbor.copy()
        for i in range(self.k_hop):
            cur_candidate_set = set()
            for item in next_queue:
                cur_candidate_set = cur_candidate_set | set(self.adj_list[item])
            k_hop_candidates = k_hop_candidates | cur_candidate_set
            next_queue = cur_candidate_set
        result = list(k_hop_candidates - one_hop_neighbor - {node})
        if len(result) > self.sample_num:
            result = random.sample(result, self.sample_num)
        for i in range(len(result)):
            if random.random() > add_rate:
                result[i] = -1
        return result

    def roughly_generate_candidate_set(self, node: int, add_rate: float = 0.1) -> list:
        flag_list = list(0 for i in range(len(self.adj_list)))

        # dyeing the neighbor nodes
        for item in self.adj_list[node]:
            flag_list[item] = 1

        result = list()
        # generate k-hop sample
        for i in range(self.sample_num):
            cur_node = node
            for j in range(self.k_hop):
                cur_node = random.sample(self.adj_list[cur_node], 1)[0]
                if random.random() > add_rate:
                    break
            if flag_list[cur_node] != 1:
                result.append(cur_node)
                flag_list[cur_node] = 1

        return result

# device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
#
# node_A = torch.LongTensor(
#     [[0, 1, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0],
#      [1, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0]])
# edge_indexk = torch_geometric.utils.dense_to_sparse(torch.squeeze(node_A, dim=0))[0]
# feature_M = torch.Tensor(
#     [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]])
# model = SampleModel(3, 2)
#
# model.train()
# out = model(feature_M, edge_indexk)
# out = F.softmax(out, dim=1)
# loss = F.cross_entropy(out, torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]))
# #
# print(model.conv1.weight.grad)
# print(model.conv2.weight.grad)
#
# print(model.weight.grad)
# print(model.bias.grad)
#
# print(model.probablity_coefficient.grad)
# #
# torch.autograd.set_detect_anomaly(True)
# loss.backward()
# #
# print(model.conv1.weight.grad)
# print(model.conv2.weight.grad)
# print(model.weight.grad)
# print(model.bias.grad)
# print(model.probablity_coefficient.grad)
