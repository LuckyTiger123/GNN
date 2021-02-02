import torch
import argparse
import torch_geometric
import torch.nn.functional as F
import pandas as pd
from torch import Tensor
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

import KDD_model.sample_model.compare_with_baseline.utils as utils
import KDD_model.sample_model.compare_with_baseline.DropEdge_model as DE
import KDD_model.sample_model.compare_with_baseline.GAT_model as GAT
import KDD_model.sample_model.compare_with_baseline.GCN_model as GCN
import KDD_model.sample_model.compare_with_baseline.GraphSAGE_model as SAGEM
import KDD_model.sample_model.compare_with_baseline.Ours_model as OM

# terminal flag
parser = argparse.ArgumentParser(description='Specify noise parameters and add and subtract edge parameters')
parser.add_argument('-ds', '--dataset', choices=['Cora', 'CiteSeer', 'PubMed'], default='Cora')
parser.add_argument('-t', '--type', choices=['add', 'drop', 'flip', 'aflip'], default='flip')
parser.add_argument('-r', '--rate', type=float, default=0.2)
parser.add_argument('-fr', '--feature_rate', type=float, default=0)
parser.add_argument('-a', '--attenuate', type=int, default=2)
parser.add_argument('-d', '--drop', type=int, default=2)
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-e', '--epoch', type=int, default=150)
parser.add_argument('-l', '--learning_rate', type=float, default=0.005)
parser.add_argument('-rd', '--round', type=int, default=10)
parser.add_argument('-ab', '--add_base', type=float, default=0.1)
parser.add_argument('-db', '--drop_base', type=float, default=0.1)
args = parser.parse_args()

# device selection
device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/{}".format(args.dataset), name=args.dataset,
                    transform=T.NormalizeFeatures())
data = dataset[0].to(device=device)


# build noise graph
def build_noise_graph():
    # build noisy graph
    if args.type == 'add':
        ng = utils.add_related_edge(data.edge_index, add_rate=args.rate)
    elif args.type == 'drop':
        ng = utils.drop_related_edge(data.edge_index, remove_rate=args.rate)
    elif args.type == 'flip':
        ng = utils.flip_related_edge(data.edge_index, flip_rate=args.rate)
    elif args.type == 'aflip':
        ng = utils.average_flip_related_edge(data.edge_index, flip_rate=args.rate)
    return ng


de_model = DE.DropEdgeModel(dataset.num_features, dataset.num_classes).to(device=device)
