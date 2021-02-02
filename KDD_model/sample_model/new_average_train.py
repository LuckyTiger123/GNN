import torch
import argparse
import torch_geometric
import torch.nn.functional as F
import pandas as pd
from torch import Tensor
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
# import KDD_model.sample_model.new_sample_model as NSM
# import KDD_model.sample_model.utils as utils

import new_sample_model as NSM
import utils as utils

# terminal flag
parser = argparse.ArgumentParser(description='Specify noise parameters and add and subtract edge parameters')
parser.add_argument('-t', '--type', choices=['add', 'drop', 'flip', 'aflip'], default='flip')
parser.add_argument('-r', '--rate', type=float, default=0.2)
parser.add_argument('-fr', '--feature_rate', type=float, default=0)
parser.add_argument('-a', '--attenuate', type=int, default=2)
parser.add_argument('-d', '--drop', type=int, default=2)
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-e', '--epoch', type=int, default=200)
parser.add_argument('-l', '--learning_rate', type=float, default=0.005)
# parser.add_argument('-f', '--form', type=int, choices=[0, 1], default=0)
parser.add_argument('-rd', '--round', type=int, default=10)
parser.add_argument('-ab', '--add_base', type=float, default=0.1)
parser.add_argument('-db', '--drop_base', type=float, default=0.1)
args = parser.parse_args()


# 2 layer class
class TwoLayerModel(torch.nn.Module):
    def __init__(self, feature_num, output_num):
        super(TwoLayerModel, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(feature_num, 32)
        self.conv2 = NSM.NewSampleModel(32, output_num)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0.4, attenuate_rate: float = 0.4,
                mask: Tensor = None):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index, drop_rate, attenuate_rate, mask=mask)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/Cora", name="Cora", transform=T.NormalizeFeatures())
data = dataset[0].to(device=device)


# build noise graph
def build_noise_graph():
    # build noisy graph
    if args.type == 'add':
        noisy_graph = utils.add_related_edge(data.edge_index, add_rate=args.rate)
    elif args.type == 'drop':
        noisy_graph = utils.drop_related_edge(data.edge_index, remove_rate=args.rate)
    elif args.type == 'flip':
        noisy_graph = utils.flip_related_edge(data.edge_index, flip_rate=args.rate)
    elif args.type == 'aflip':
        noisy_graph = utils.average_flip_related_edge(data.edge_index, flip_rate=args.rate)
    return noisy_graph


# model init
model = TwoLayerModel(dataset.num_features, dataset.num_classes).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

# change base
attenuate_base = args.add_base
drop_base = args.drop_base

# total result
best_val = list(0 for item in range(args.attenuate * args.drop))
test_result = list(0 for item in range(args.attenuate * args.drop))
train_loss = list(0 for item in range(args.attenuate * args.drop))

# train
for round_num in range(args.round):
    noisy_graph = build_noise_graph()
    feature_x = utils.flip_feature_for_cora(data.x, flip_rate=args.feature_rate)

    for m in range(args.attenuate):
        attenuate_rate = attenuate_base * m
        for n in range(args.drop):
            best_validate_rate = 0
            test_rate_under_best_validate = 0
            train_loss_under_best_validate = 0

            drop_rate = drop_base * n

            model.reset_parameters()
            optimizer.zero_grad()

            count = m * args.drop + n

            # train
            for epoch in range(args.epoch):
                model.train()
                optimizer.zero_grad()
                out = model(feature_x, noisy_graph, drop_rate, attenuate_rate, mask=data.train_mask)
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                loss.backward()

                # validate set
                model.eval()
                with torch.no_grad():
                    _, pred = model(feature_x, noisy_graph).max(dim=1)
                    validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
                    validate_acc = validate_correct / int(data.val_mask.sum())
                    if validate_acc > best_validate_rate:
                        best_validate_rate = validate_acc
                        test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                        test_acc = test_correct / int(data.test_mask.sum())
                        test_rate_under_best_validate = test_acc
                        train_loss_under_best_validate = round(float(loss), 4)

                print(
                    'on the {} round, for the {} model, on the {} epoch, the loss is {}'.format(round_num, count, epoch,
                                                                                                round(float(loss), 4)))
                optimizer.step()

            print('the best acc on validate set is {}'.format(best_validate_rate))
            print('the test acc rate is {}'.format(test_rate_under_best_validate))

            best_val[count] += best_validate_rate
            test_result[count] += test_rate_under_best_validate
            train_loss[count] += train_loss_under_best_validate

# write to file
agg_table = pd.read_excel('/home/luckytiger/GNN/KDD_model/model_result/agg_result/feature_change_average_result.xlsx')
for m in range(args.attenuate):
    attenuate_rate = attenuate_base * m
    for n in range(args.drop):
        drop_rate = drop_base * n
        count = m * args.drop + n

        agg_table.loc[agg_table.shape[0]] = {'type': args.type, 'edge_rate': args.rate,
                                             'feature_rate': args.feature_rate, 'attenuate': attenuate_rate,
                                             'drop': drop_rate, 'lr': args.learning_rate, 'epoch': args.epoch,
                                             'acc_on_test': test_result[count] / args.round,
                                             'acc_on_val': best_val[count] / args.round,
                                             'loss_on_train': train_loss[count] / args.round,
                                             'sample_round': args.round}

agg_table = agg_table[
    ['type', 'edge_rate', 'feature_rate', 'attenuate', 'drop', 'lr', 'epoch', 'acc_on_test', 'acc_on_val',
     'loss_on_train', 'sample_round']]
agg_table.to_excel('/home/luckytiger/GNN/KDD_model/model_result/agg_result/feature_change_average_result.xlsx')

print('mission complete!')
