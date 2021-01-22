import torch
import pandas as pd
import torch_geometric.transforms as T
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.typing import Adj
import KDD_model.new_model.kdd_new_model as FS

# device init
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/Cora", name="Cora", transform=T.NormalizeFeatures())
data = dataset[0].to(device=device)

# svae file path
save_path = '/home/luckytiger/GNN/KDD_model/model_result/average_result_short'


# model
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


# graph change rate base
graph_add_base = 0.00005 * 3
graph_remove_base = 0.1

# train and test the model
# init model and optimizer
model = TwoLayerFSL(dataset.num_features, dataset.num_classes).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# out cycle
# noise add edge multiplier
for i in range(3):
    graph_add_lambda = graph_add_base * i
    # noise remove edge multiplier
    for j in range(3):
        graph_remove_lambda = graph_remove_base * j

        # add noise
        noisy_edge_index = FS.add_noise_edge(data.edge_index, add_rate=graph_add_lambda,
                                             remove_rate=graph_remove_lambda)

        # collect result
        acc_collect = pd.DataFrame(
            columns=['drop_rate\\add_rate', '0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
        for k in range(7):
            acc_collect.loc[acc_collect.shape[0]] = {'drop_rate\\add_rate': 0.1 * k}
        loss_on_train_collect = pd.DataFrame(
            columns=['drop_rate\\add_rate', '0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
        for k in range(7):
            loss_on_train_collect.loc[loss_on_train_collect.shape[0]] = {'drop_rate\\add_rate': 0.1 * k}

        # init add rate and drop rate
        add_rate = 0
        drop_rate = 0

        # define epoch num and round num
        epoch_num = 300
        round_num = 5

        # add rate cycle
        for l in range(3):
            add_rate = round(0.1 * l, 1)
            # drop rate cycle
            for m in range(3):
                drop_rate = round(0.1 * m, 1)

                # average result
                average_acc_on_test = 0
                average_loss_on_train = 0

                # round cycle
                for n in range(round_num):
                    # model reset
                    optimizer.zero_grad()
                    model.layer_1.reset_parameters()
                    model.layer_2.reset_parameters()

                    best_validate_loss = 10
                    best_entropy = 0
                    best_acc_in_test = 0

                    # model train and select
                    for epoch in range(epoch_num):
                        model.train()
                        edge_index = FS.add_related_edge(noisy_edge_index, add_rate=add_rate, random_seed=epoch)
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

                                # for display
                                best_validate_loss = float(val_loss)
                                best_entropy = float(loss)
                                best_acc_in_test = float(acc)

                        optimizer.step()
                    average_acc_on_test += best_acc_in_test
                    average_loss_on_train += best_validate_loss

                # cal avg
                average_acc_on_test = round(average_acc_on_test / round_num, 4)
                average_loss_on_train = round(average_loss_on_train / round_num, 4)

                # record in data grid
                acc_collect.loc[m, str(add_rate)] = average_acc_on_test
                loss_on_train_collect.loc[m, str(add_rate)] = average_loss_on_train

                # terminal display
                print('the add rate {} and drop rate {} has been over.'.format(add_rate, drop_rate))
                print('the acc is {},the loss on train is {}'.format(average_acc_on_test, average_loss_on_train))
                print('-----------------------------------------------------')

        # save file
        if graph_add_lambda == 0 and graph_remove_lambda == 0:
            acc_collect.to_excel(
                '{}/clean/clean_graph_acc.xlsx'.format(save_path))
            loss_on_train_collect.to_excel(
                '{}/clean/clean_graph_loss.xlsx'.format(save_path))
        elif graph_add_lambda != 0 and graph_remove_lambda != 0:
            acc_collect.to_excel(
                '{}/add&remove/A{}_D{}_acc.xlsx'.format(
                    save_path, round(graph_add_lambda, 5), round(graph_remove_lambda, 1)))
            loss_on_train_collect.to_excel(
                '{}/add&remove/A{}_D{}_loss.xlsx'.format(
                    save_path, round(graph_add_lambda, 5), round(graph_remove_lambda, 1)))
        elif graph_add_lambda != 0:
            acc_collect.to_excel('{}/add/A{}_acc.xlsx'.format(save_path,
                                                              round(graph_add_lambda, 5)))
            loss_on_train_collect.to_excel(
                '{}/add/A{}_loss.xlsx'.format(
                    save_path, round(graph_add_lambda, 5)))
        else:
            acc_collect.to_excel(
                '{}/remove/D{}_acc.xlsx'.format(
                    save_path, round(graph_remove_lambda, 1)))
            loss_on_train_collect.to_excel(
                '{}/remove/D{}_loss.xlsx'.format(
                    save_path, round(graph_remove_lambda, 1)))

        print('the origin graph with {} add lambda and {} remove lambda has saved successfully'.format(graph_add_lambda,
                                                                                                       graph_remove_lambda))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

print('mission completed!')
