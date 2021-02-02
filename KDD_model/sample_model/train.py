import torch
import argparse
import torch.nn.functional as F
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import KDD_model.sample_model.sample_model as SM
import KDD_model.sample_model.utils as utils
# import sample_model as SM
# import utils as utils

# terminal flag
parser = argparse.ArgumentParser(description='Specify noise parameters and add and subtract edge parameters')
parser.add_argument('-t', '--type', choices=['add', 'drop', 'flip'], default='flip')
parser.add_argument('-r', '--rate', type=float, default=0.2)
parser.add_argument('-a', '--attenuate', type=float, default=0.4)
parser.add_argument('-d', '--drop', type=float, default=0.4)
parser.add_argument('-c', '--cuda', type=int, default=4)
parser.add_argument('-e', '--epoch', type=int, default=100)
parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
parser.add_argument('-f', '--form', type=int, choices=[0, 1], default=0)
args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/Cora", name="Cora", transform=T.NormalizeFeatures())
data = dataset[0].to(device=device)

# build noisy graph
noisy_graph = torch.tensor([], device=device)
if args.type == 'add':
    noisy_graph = utils.add_related_edge(data.edge_index, add_rate=args.rate)
elif args.type == 'drop':
    noisy_graph = utils.drop_related_edge(data.edge_index, remove_rate=args.rate)
elif args.type == 'flip':
    noisy_graph = utils.flip_related_edge(data.edge_index, flip_rate=args.rate)

# model init
model = SM.SampleModel(dataset.num_features, dataset.num_classes).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

best_validate_rate = 0
test_rate_under_best_validate = 0
# train
for epoch in range(args.epoch):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, noisy_graph, args.drop, args.attenuate, args.form, mask=data.train_mask)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()

    # validate set
    model.eval()
    with torch.no_grad():
        _, pred = model(data.x, noisy_graph).max(dim=1)

        validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        validate_acc = validate_correct / int(data.val_mask.sum())
        if validate_acc > best_validate_rate:
            best_validate_rate = validate_acc
            test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            test_acc = test_correct / int(data.test_mask.sum())
            test_rate_under_best_validate = test_acc

    print('on the {} epoch, the loss is {}'.format(epoch, round(float(loss), 4)))
    optimizer.step()

print('the best acc on validate set is {}'.format(best_validate_rate))
print('the test acc rate is {}'.format(test_rate_under_best_validate))

# write to file
agg_table = pd.read_excel('/home/luckytiger/GNN/KDD_model/model_result/agg_result/agg_result.xlsx')
agg_table.loc[agg_table.shape[0]] = {'type': args.type, 'rate': args.rate, 'attenuate': args.attenuate,
                                     'drop': args.drop, 'lr': args.learning_rate, 'epoch': args.epoch,
                                     'acc': test_rate_under_best_validate}
agg_table = agg_table[['type', 'rate', 'attenuate', 'drop', 'lr', 'epoch', 'acc']]
agg_table.to_excel('/home/luckytiger/GNN/KDD_model/model_result/agg_result/agg_result.xlsx')

print('mission complete!')
