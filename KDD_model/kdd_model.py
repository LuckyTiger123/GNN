import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, feature_num, output_num):
        super(Model, self).__init__()
        # init feature coefficient
        A = torch.ones(feature_num, 1, requires_grad=True)
        self.A = torch.nn.Parameter(A)
        self.register_parameter('FeatureCoefficient', self.A)

        # init weight coefficient
        B = torch.zeros(feature_num, output_num, requires_grad=True)
        self.B = torch.nn.Parameter(B)
        self.register_parameter('WeightCoefficient', self.B)

    def forward(self, node_adjacency, feature_map, fe_mask_rate):
        fsw = torch.mm(self.A, node_adjacency)
        fsw = F.dropout(fsw, p=fe_mask_rate, training=True)  # TODO
        agg_feature = torch.zeros(1, fsw.shape[0], requires_grad=True)
        for i in range(fsw.shape[0]):
            item1 = torch.unsqueeze(fsw[i], 0)
            item2 = torch.unsqueeze(feature_map.T[i], 0)
            sum = torch.squeeze(torch.mm(item1, item2.T), 0)
            agg_feature[0][i] = sum
        result = torch.mm(agg_feature, self.B)
        return result


# test model
# model = Model(3, 4)
# node_A = torch.Tensor([0, 1, 1, 0])
# node_a = torch.unsqueeze(node_A, 0)
# feature_M = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# model(node_a, feature_M, 0.1)
