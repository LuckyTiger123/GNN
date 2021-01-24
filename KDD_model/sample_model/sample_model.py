import gc
import torch
import math
import random
import torch_geometric
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Parameter


class SampleModel(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, k_hop: int = 3, sample_num: int = 4, **kwargs):
        super(SampleModel, self).__init__()

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

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0.4):
        # add self loop
        if self.add_self_loops:
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # adjacency list
        self.adj_list = list(list() for i in range(x.size(0)))
        self.fill_adj_list(edge_index)

        # generate GCN embedding of node
        half_embedding = self.conv1(x, edge_index)
        node_embedding = self.conv2(half_embedding, edge_index)

        feature_result = torch.tensor([]).to(device=x.device)

        # node-wise operation
        for i in range(x.size(0)):
            self_feature = torch.zeros(x.size(1)).to(device=x.device)
            feature_agg = torch.tensor([]).to(device=x.device)
            # print(feature_agg)
            for j in self.adj_list[i]:
                feature_agg = torch.cat([feature_agg, torch.unsqueeze(x[j], dim=0)], 0)

            # random drop
            if self.training:
                dropout_matrix = torch.ones_like(feature_agg, device=x.device) - drop_rate
                feature_agg = feature_agg.mul(torch.bernoulli(dropout_matrix).long())
                del dropout_matrix

            # self agg
            for j in range(len(self.adj_list[i])):
                self_feature += feature_agg[j] / (
                        pow(len(self.adj_list[i]), 0.5) * pow(len(self.adj_list[self.adj_list[i][j]]), 0.5))
            del feature_agg
            self_feature = torch.unsqueeze(self_feature, 0)

            # candidate set
            if self.training:
                candidate_set = self.generate_candidate_set(i)

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

                add_rate = torch.mm(feature_mix, self.probablity_coefficient)
                # add_rate = F.softmax(add_rate, dim=0).view(1, -1)

                # gumbel softmax
                add_rate = F.gumbel_softmax(add_rate, tau=0.1, hard=False, dim=0).view(1, -1)

                for l in range(self.sample_num):
                    add_rate[0][l] /= (
                            pow(len(self.adj_list[i]), 0.5) * (
                        pow(len(self.adj_list[candidate_set[l]]), 0.5) if candidate_set[l] != -1 else 1))

                add_feature = torch.mm(add_rate, candidate_feature)
                self_feature += add_feature

            feature_result = torch.cat([feature_result, self_feature], 0)

            # call gc
            gc.collect()
            print('{} node has caled.'.format(i))

        # transform feature dimension
        out = torch.matmul(feature_result, self.weight)

        if self.bias is not None:
            out += self.bias

        return out

    def fill_adj_list(self, edge_index: Adj):
        for i in range(edge_index.size(1)):
            self.adj_list[edge_index[1][i]].append(int(edge_index[0][i]))

    def generate_candidate_set(self, node: int) -> list:
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
        del one_hop_neighbor
        del k_hop_candidates
        del next_queue
        if len(result) < self.sample_num:
            return result
        else:
            return random.sample(result, self.sample_num)
            # return result

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#
# node_A = torch.LongTensor(
#     [[0, 1, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0],
#      [1, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0]])
# edge_indexk = torch_geometric.utils.dense_to_sparse(torch.squeeze(node_A, dim=0))[0]
# feature_M = torch.Tensor(
#     [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]])
# model = SampleModel(3, 2)
# model.train()
# out = model(feature_M, edge_indexk)
# print(out)
