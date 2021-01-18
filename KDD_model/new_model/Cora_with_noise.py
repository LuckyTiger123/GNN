import torch
import pandas as pd
import torch_geometric.transforms as T
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.typing import Adj
import KDD_model.new_model.kdd_new_model as FS

# device init
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/Cora", name="Cora", transform=T.NormalizeFeatures())
data = dataset[0].to(device=device)

# init change rate and change type
graph_add_rate = 0.00005 * 15
graph_remove_rate = 0.5

# add noise
noisy_edge_index = FS.add_noise_edge(data.edge_index, add_rate=graph_add_rate, remove_rate=graph_remove_rate)


# 2 layer model
class TwoLayerFSL(torch.nn.Module):
    def __init__(self, feature_num, output_num):
        super(TwoLayerFSL, self).__init__()
        self.layer_1 = FS.FeatureSwitchLayer(feature_num, 32)
        self.layer_2 = FS.FeatureSwitchLayer(32, output_num)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0.1):
        x = self.layer_1(x, edge_index, drop_rate=drop_rate)
        x = F.elu(x)
        x = self.layer_2(x, edge_index, drop_rate=drop_rate)
        return F.log_softmax(x, dim=1)


# init data grid
add_rate_turn = 0
if add_rate_turn == 0:
    acc_collect = pd.DataFrame(
        columns=['drop_rate\\add_lambda(0.00005)', '0', '1', '2', '4', '8', '16', '32'])
    for i in range(9):
        acc_collect.loc[acc_collect.shape[0]] = {'drop_rate\\add_lambda(0.00005)': 0.1 * i}
    loss_on_train_collect = pd.DataFrame(
        columns=['drop_rate\\add_lambda(0.00005)', '0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'])
    for i in range(9):
        loss_on_train_collect.loc[loss_on_train_collect.shape[0]] = {'drop_rate\\add_lambda(0.00005)': 0.1 * i}
else:
    acc_collect = pd.read_excel(
        '/home/luckytiger/GNN/KDD_model/model_result/add_and_remove_edge/acc_{}_{}.xlsx'.format(graph_remove_rate,
                                                                                                graph_add_rate))
    loss_on_train_collect = pd.read_excel(
        '/home/luckytiger/GNN/KDD_model/model_result/add_and_remove_edge/loss_{}_{}.xlsx'.format(graph_remove_rate,
                                                                                                 graph_add_rate))

# init add rate and drop rate
add_rate = 0
drop_rate = 0

# define epoch num
epoch_num = 300

# train and test the model
# init model and optimizer
model = TwoLayerFSL(dataset.num_features, dataset.num_classes).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# add rate cycle
for i in range(7):
    # with add rate
    add_item = 0
    if i != 0:
        add_item = pow(2, i - 1)
    add_rate = add_item * 0.00005
    for j in range(9):
        # with drop rate 0.1 * i
        drop_rate = 0.1 * j
        print('-------------------------------')

        # model reset
        optimizer.zero_grad()
        model.layer_1.reset_parameters()
        model.layer_2.reset_parameters()

        best_validate_loss = 10
        best_entropy = 0
        best_acc_in_test = 0

        # model train and select
        for epoch in range(300):
            model.train()
            edge_index = FS.add_edge(noisy_edge_index, add_rate=add_rate)
            optimizer.zero_grad()
            out = model(data.x, edge_index, drop_rate=drop_rate)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()

            # validate set
            model.eval()
            with torch.no_grad():
                val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
                if float(val_loss) < best_validate_loss:
                    # train set
                    _, pred = model(data.x, noisy_edge_index, drop_rate=0).max(dim=1)
                    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
                    acc = correct / int(data.test_mask.sum())

                    # update the data grid
                    loss_on_train_collect.loc[j, str(add_item)] = float(loss)
                    acc_collect.loc[j, str(add_item)] = float(acc)

                    # for display
                    best_validate_loss = float(val_loss)
                    best_entropy = float(loss)
                    best_acc_in_test = float(acc)

            optimizer.step()
        print('with {} add rate and {} drop rate, the loss on train is {}. the acc on test is {}.'.format(add_rate,
                                                                                                          drop_rate,
                                                                                                          best_entropy,
                                                                                                          best_acc_in_test))

        print('-------------------------------')

acc_collect.to_excel(
    '/home/luckytiger/GNN/KDD_model/model_result/add_and_remove_edge/acc_{}_{}.xlsx'.format(graph_remove_rate,
                                                                                            graph_add_rate))
loss_on_train_collect.to_excel(
    '/home/luckytiger/GNN/KDD_model/model_result/add_and_remove_edge/loss_{}_{}.xlsx'.format(graph_remove_rate,
                                                                                             graph_add_rate))
print('mission completed!')
