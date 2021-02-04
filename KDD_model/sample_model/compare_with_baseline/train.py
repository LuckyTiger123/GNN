import torch
import argparse
import torch_geometric
import torch.nn.functional as F
import pandas as pd
from torch import Tensor
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from deeprobust.graph.defense import GCN

import KDD_model.sample_model.compare_with_baseline.Our_model as OM
import KDD_model.sample_model.compare_with_baseline.utils as utils
import KDD_model.sample_model.compare_with_baseline.DropEdge_model as DE
import KDD_model.sample_model.compare_with_baseline.GAT_model as GAT
import KDD_model.sample_model.compare_with_baseline.GCN_model as GCNM
import KDD_model.sample_model.compare_with_baseline.GraphSAGE_model as SAGEM
import KDD_model.sample_model.compare_with_baseline.GRAND_model as GRAND
import KDD_model.sample_model.compare_with_baseline.ProGNN_model as ProGNN
import KDD_model.sample_model.compare_with_baseline.RGCN_model as RGCN

# terminal flag
parser = argparse.ArgumentParser(description='Specify noise parameters and add and subtract edge parameters')
parser.add_argument('-ds', '--dataset', choices=['Cora', 'CiteSeer', 'PubMed'], default='Cora')
parser.add_argument('-t', '--type', choices=['add', 'drop', 'flip', 'aflip'], default='aflip')
parser.add_argument('-r', '--rate', type=float, default=0)
parser.add_argument('-fr', '--feature_rate', type=float, default=0)
# parser.add_argument('-a', '--attenuate', type=int, default=2)
# parser.add_argument('-d', '--drop', type=int, default=2)
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-e', '--epoch', type=int, default=200)
parser.add_argument('-l', '--learning_rate', type=float, default=0.01)
parser.add_argument('-rd', '--round', type=int, default=10)
parser.add_argument('-ab', '--add_base', type=float, default=0.1)
parser.add_argument('-db', '--drop_base', type=float, default=0.1)
args = parser.parse_args()

# device selection
device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/{}".format(args.dataset), name=args.dataset,
                    transform=T.NormalizeFeatures())
data = dataset[0].to(device=device)


# build noise graph
def build_noise_graph(random_seed: int = 0):
    # build noisy graph
    if args.type == 'add':
        ng = utils.add_related_edge(data.edge_index, add_rate=args.rate)
    elif args.type == 'drop':
        ng = utils.drop_related_edge(data.edge_index, remove_rate=args.rate)
    elif args.type == 'flip':
        ng = utils.flip_related_edge(data.edge_index, flip_rate=args.rate)
    elif args.type == 'aflip':
        ng = utils.average_flip_related_edge_with_sparse_matrix(data.edge_index, data.x.size(0),
                                                                random_seed=random_seed, flip_rate=args.rate)
    return ng


# train a DropEdge Model
def train_de_model(x, edge_index, k):
    if args.dataset == 'Cora':
        drop_rate = 0.3
    elif args.dataset == 'CiteSeer':
        drop_rate = 0.95
    elif args.dataset == 'PubMed':
        drop_rate = 0.7

    model = DE.DropEdgeModel(dataset.num_features, dataset.num_classes).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.005)

    best_validate_rate = 0
    test_rate_under_best_validate = 0
    train_loss_under_best_validate = 0

    # train
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index, drop_rate)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        # validate set
        model.eval()
        with torch.no_grad():
            _, pred = model(x, edge_index).max(dim=1)
            validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            validate_acc = validate_correct / int(data.val_mask.sum())
            if validate_acc > best_validate_rate:
                best_validate_rate = validate_acc
                test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                test_acc = test_correct / int(data.test_mask.sum())
                test_rate_under_best_validate = test_acc
                train_loss_under_best_validate = round(float(loss), 4)

        print('for the {} th DropEdge model and the {} epoch, the loss is {}.'.format(k, epoch, float(loss)))
        optimizer.step()

    return best_validate_rate, test_rate_under_best_validate, train_loss_under_best_validate


# train a GAT Model
def train_GAT_model(x, edge_index, k):
    model = GAT.GATModel(dataset.num_features, dataset.num_classes).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    best_validate_rate = 0
    test_rate_under_best_validate = 0
    train_loss_under_best_validate = 0

    # train
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        # validate set
        model.eval()
        with torch.no_grad():
            _, pred = model(x, edge_index).max(dim=1)
            validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            validate_acc = validate_correct / int(data.val_mask.sum())
            if validate_acc > best_validate_rate:
                best_validate_rate = validate_acc
                test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                test_acc = test_correct / int(data.test_mask.sum())
                test_rate_under_best_validate = test_acc
                train_loss_under_best_validate = round(float(loss), 4)

        print('for the {} th GAT model and the {} epoch, the loss is {}.'.format(k, epoch, float(loss)))
        optimizer.step()

    return best_validate_rate, test_rate_under_best_validate, train_loss_under_best_validate


# train a GCN model
def train_GCN_model(x, edge_index, k):
    model = GCNM.GCNModel(dataset.num_features, dataset.num_classes).to(device=device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=args.learning_rate)

    best_validate_rate = 0
    test_rate_under_best_validate = 0
    train_loss_under_best_validate = 0

    # train
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        # validate set
        model.eval()
        with torch.no_grad():
            _, pred = model(x, edge_index).max(dim=1)
            validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            validate_acc = validate_correct / int(data.val_mask.sum())
            if validate_acc > best_validate_rate:
                best_validate_rate = validate_acc
                test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                test_acc = test_correct / int(data.test_mask.sum())
                test_rate_under_best_validate = test_acc
                train_loss_under_best_validate = round(float(loss), 4)

        print('for the {} th GCN model and the {} epoch, the loss is {}.'.format(k, epoch, float(loss)))
        optimizer.step()

    return best_validate_rate, test_rate_under_best_validate, train_loss_under_best_validate


# train a SAGE model
def train_SAGE_model(x, edge_index, k):
    model = SAGEM.SAGEModel(dataset.num_features, dataset.num_classes).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    best_validate_rate = 0
    test_rate_under_best_validate = 0
    train_loss_under_best_validate = 0

    # train
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        # validate set
        model.eval()
        with torch.no_grad():
            _, pred = model(x, edge_index).max(dim=1)
            validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            validate_acc = validate_correct / int(data.val_mask.sum())
            if validate_acc > best_validate_rate:
                best_validate_rate = validate_acc
                test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                test_acc = test_correct / int(data.test_mask.sum())
                test_rate_under_best_validate = test_acc
                train_loss_under_best_validate = round(float(loss), 4)

        print('for the {} th SAGE model and the {} epoch, the loss is {}.'.format(k, epoch, float(loss)))
        optimizer.step()

    return best_validate_rate, test_rate_under_best_validate, train_loss_under_best_validate


# train a GRAND model
def train_GRAND_model(x, edge_index, k):
    if args.dataset == 'Cora':
        input_droprate = 0.5
        hidden_droprate = 0.5
        dropnode_rate = 0.5
        K = 4
        lam = 1.0
        tem = 0.5
        order = 8
    elif args.dataset == 'CiteSeer':
        input_droprate = 0
        hidden_droprate = 0.2
        dropnode_rate = 0.5
        K = 2
        lam = 0.7
        tem = 0.3
        order = 2
    elif args.dataset == 'PubMed':
        input_droprate = 0.6
        hidden_droprate = 0.8
        dropnode_rate = 0.5
        K = 4
        lam = 1.0
        tem = 0.2
        order = 5

    edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=data.x.size(0))
    row, col = edge_index
    deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    model = GRAND.GRANDModel(dataset.num_features, dataset.num_classes, input_droprate=input_droprate,
                             hidden_droprate=hidden_droprate).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    sparse_tensor = torch.sparse.FloatTensor(edge_index, edge_weight,
                                             torch.Size([feature_x.size(0), feature_x.size(0)])).float()

    best_validate_rate = 0
    test_rate_under_best_validate = 0
    train_loss_under_best_validate = 0

    # train
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()

        X_list = []
        for i in range(K):
            X_list.append(GRAND.rand_prop(sparse_tensor, x, dropnode_rate, order, training=True))

        output_list = []
        for i in range(K):
            output_list.append(torch.log_softmax(model(X_list[i]), dim=-1))

        loss_train = 0.
        for i in range(K):
            loss_train += F.nll_loss(output_list[i][data.train_mask], data.y[data.train_mask])

        loss_train = loss_train / K
        loss_consis = GRAND.consis_loss(output_list, tem, lam)
        loss_train = loss_train + loss_consis
        loss_train.backward()

        # validate set
        model.eval()
        with torch.no_grad():
            X = x
            X = GRAND.rand_prop(sparse_tensor, X, dropnode_rate, order, training=False)
            _, pred = model(X).max(dim=1)
            validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            validate_acc = validate_correct / int(data.val_mask.sum())
            if validate_acc > best_validate_rate:
                best_validate_rate = validate_acc
                test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                test_acc = test_correct / int(data.test_mask.sum())
                test_rate_under_best_validate = test_acc
                train_loss_under_best_validate = round(float(loss_train), 4)

        print('for the {} th GRAND model and the {} epoch, the loss is {}.'.format(k, epoch, float(loss_train)))
        optimizer.step()

    return best_validate_rate, test_rate_under_best_validate, train_loss_under_best_validate


# train a proGNN model
def train_proGNN_model(x, edge_index, k):
    data_item = utils.process_data_for_nettack_on_noisy_graph(x, edge_index, data).to(device)
    adj, features, labels = data_item.adj, data_item.features, data_item.labels
    idx_train, idx_val, idx_test = data_item.idx_train, data_item.idx_val, data_item.idx_test

    gcn_model = GCN(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
    adj, features, labels = utils.preprocess(adj, features, labels, preprocess_adj=False, device=device)
    model = ProGNN.ProGNN(model=gcn_model, device=device)
    model.fit(features, adj, labels, idx_train, idx_val, idx_test, train_iters=args.epoch)
    return model.test_acc_under_best_val


# train a RGCN model
def train_RGCN_model(x, edge_index, k):
    data_item = utils.process_data_for_nettack_on_noisy_graph(x, edge_index, data).to(device)
    adj, features, labels = data_item.adj, data_item.features, data_item.labels
    idx_train, idx_val, idx_test = data_item.idx_train, data_item.idx_val, data_item.idx_test
    adj, features, labels = utils.preprocess(adj, features, labels, preprocess_adj=False, device=device)
    model = RGCN.RGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16,
                      device=device).to(device=device)
    model.fit(features, adj, labels, idx_train, idx_val, idx_test, train_iters=args.epoch)
    return model.test_acc


# train our model
def train_our_model(x, edge_index, k):
    model = OM.TwoLayerModel(dataset.num_features, dataset.num_classes).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    if args.dataset == 'Cora':
        drop_rate = 0.1
        add_rate = 0.1
    elif args.dataset == 'CiteSeer':
        drop_rate = 0.1
        add_rate = 0.1
    elif args.dataset == 'PubMed':
        drop_rate = 0.1
        add_rate = 0.1

    best_validate_rate = 0
    test_rate_under_best_validate = 0
    train_loss_under_best_validate = 0

    # train
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index, drop_rate, add_rate)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        # validate set
        model.eval()
        with torch.no_grad():
            _, pred = model(x, edge_index).max(dim=1)
            validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            validate_acc = validate_correct / int(data.val_mask.sum())
            if validate_acc > best_validate_rate:
                best_validate_rate = validate_acc
                test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                test_acc = test_correct / int(data.test_mask.sum())
                test_rate_under_best_validate = test_acc
                train_loss_under_best_validate = round(float(loss), 4)

        print('for the {} th our model and the {} epoch, the loss is {}.'.format(k, epoch, float(loss)))
        optimizer.step()

    return best_validate_rate, test_rate_under_best_validate, train_loss_under_best_validate


test_acc_sample_result = list(list() for i in range(8))
val_acc_sample_result = list(list() for i in range(8))
train_loss_sample_result = list(list() for i in range(8))

# train
for round_num in range(args.round):
    noisy_graph = build_noise_graph(round_num)
    feature_x = utils.flip_feature_for_dataset(data.x, flip_rate=args.feature_rate)
    



# data operation
noisy_graph = build_noise_graph()

feature_x = utils.flip_feature_for_dataset(data.x, flip_rate=args.feature_rate)

# noisy_graph, _ = torch_geometric.utils.add_self_loops(noisy_graph, num_nodes=data.x.size(0))
# row, col = noisy_graph
# deg = torch_geometric.utils.degree(col, feature_x.size(0), dtype=feature_x.dtype)
# deg_inv_sqrt = deg.pow(-0.5)
# edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#
# print(edge_weight.size())
# print(edge_weight)
#
# sparse_tensor = torch.sparse.FloatTensor(noisy_graph, edge_weight,
#                                          torch.Size([feature_x.size(0), feature_x.size(0)])).float()

# a = torch.spmm(sparse_tensor, feature_x) - torch.mm(
#     torch.squeeze(torch_geometric.utils.to_dense_adj(noisy_graph[0])).float().to(device=device), feature_x)

a = train_GRAND_model(feature_x, noisy_graph, 1)
print(a)
