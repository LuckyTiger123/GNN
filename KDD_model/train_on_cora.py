import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
import torch.nn.functional as F

# device init
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/Cora", name="Cora", transform=T.NormalizeFeatures())
data = dataset[0].to(device=device)

# transform to adj matrix
adj_matrix = SparseTensor.from_edge_index(data.edge_index).to_dense().to(device=device)

# model
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
        fsw = F.dropout(fsw, p=fe_mask_rate, training=self.training)  # TODO
        agg_feature = torch.zeros(1, fsw.shape[0], device=device)
        for i in range(fsw.shape[0]):
            item1 = torch.unsqueeze(fsw[i], 0)
            item2 = torch.unsqueeze(feature_map.T[i], 0)
            feature_sum = torch.squeeze(torch.mm(item1, item2.T), 0)
            agg_feature[0][i] = feature_sum
        result = torch.mm(agg_feature, self.B)
        return F.log_softmax(result, dim=1)


# model and optimizer and fe_mask_rate
model = Model(dataset.num_features, dataset.num_classes).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
fe_mask_rate = 0.1

# train the model
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    total_loss = torch.zeros(1).to(device=device)
    for i in range(adj_matrix[data.train_mask].shape[0]):
        out = model(torch.unsqueeze(adj_matrix[data.train_mask][i], 0).to(device=device), data.x, fe_mask_rate)
        loss = F.nll_loss(out, torch.unsqueeze(data.y[data.train_mask][i], 0).to(device=device))
        total_loss += loss
    total_loss = total_loss / adj_matrix[data.train_mask].shape[0]
    print('after {} epoch, the total loss is {}.'.format(epoch, float(total_loss)))
    total_loss.backward()
    optimizer.step()
    # break

# test on the dataset
model.eval()
total = 0
acc = 0
for i in range(adj_matrix[data.test_mask].shape[0]):
    _, pred = model(torch.unsqueeze(adj_matrix[data.test_mask][i], 0).to(device=device), data.x, fe_mask_rate).max(
        dim=1)
    if pred == data.y[data.test_mask][i]:
        acc += 1
        print('the {} test data is correct!'.format(i))
    else:
        print('the {} test data is failed!'.format(i))
    total += 1

print('the acc rate on test set is {}'.format(acc / total))
print('Mission success!')
