import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as N

dataset = Planetoid(root="/home/trfang/datasets/Cora", name="Cora")


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = N.GATConv(dataset.num_node_features, 8, heads=8, dropout=0.6)
        self.conv2 = N.GATConv(64, dataset.num_classes, heads=1, dropout=0.6)
        # self.conv1 = N.GCNConv(dataset.num_node_features, 64)
        # self.conv2 = N.GCNConv(64, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.6, training=self.training)
        eluLayer = torch.nn.ELU()
        x = eluLayer(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        # return F.softmax(x, dim=1)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    # loss_fn = torch.nn.MSELoss()
    # loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))
