import torch
import numpy as np
import torch.nn.functional as F


def consis_loss(logps, temp, lam):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)

    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return lam * loss


def propagate(feature, A, order):
    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        y.add_(x)
    return y.div_(order + 1.0).detach_()


def rand_prop(A, features, dropnode_rate, order, training):
    n = features.shape[0]
    drop_rate = dropnode_rate
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    if training:
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        features = masks.cuda() * features
    else:
        features = features * (1. - drop_rate)
    features = propagate(features, A, order)
    return features


class GRANDModel(torch.nn.Module):
    def __init__(self, feature_num, output_num, input_droprate, hidden_droprate):
        super(GRANDModel, self).__init__()
        self.layer1 = torch.nn.Linear(feature_num, 32)
        self.layer2 = torch.nn.Linear(32, output_num)
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate

    def forward(self, x):
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)
        return x

    def reset_parameters(self, input_droprate, hidden_droprate):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate

# Cora:python train_grand.py --lam 1.0 --tem 0.5 --order 8 --sample 4 --dataset cora --input_droprate 0.5
# --hidden_droprate 0.5 --hidden 32 --lr 0.01 --patience 200 --seed 100 --dropnode_rate 0.5  --cuda_device 2

# citeseer:python train_grand.py --lam 0.7 --tem 0.3 --order 2 --sample 2 --dataset citeseer --input_droprate 0.0
# --hidden_droprate 0.2 --hidden 32 --lr 0.01 --patience 200 --seed 100 --dropnode_rate 0.5  --cuda_device 0

# pubmed:python train_grand.py --lam 1.0 --tem 0.2 --order 5 --sample 4 --dataset pubmed --input_droprate 0.6
# --hidden_droprate 0.8 --hidden 32 --seed 100 --dropnode_rate 0.5 --patience 100 --lr 0.2  --use_bn
