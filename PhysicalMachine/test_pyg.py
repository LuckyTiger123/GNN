import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as N

dataset = Planetoid(root="/home/trfang/datasets/Cora", name="Cora")


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = N.GATConv(dataset.num_node_features, 8)
        self.conv2 = N.GATConv(dataset.num_node_features, 8)
        self.conv3 = N.GATConv(dataset.num_node_features, 8)
        self.conv4 = N.GATConv(dataset.num_node_features, 8)
        self.conv5 = N.GATConv(dataset.num_node_features, 8)
        self.conv6 = N.GATConv(dataset.num_node_features, 8)
        self.conv7 = N.GATConv(dataset.num_node_features, 8)
        self.conv8 = N.GATConv(dataset.num_node_features, 8)
        self.conv9 = N.GATConv(8, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = F.dropout(x, 0.6)
        x1 = self.conv1(x1, edge_index)
        x2 = F.dropout(x, 0.6)
        x2 = self.conv2(x2, edge_index)
        x3 = F.dropout(x, 0.6)
        x3 = self.conv3(x3, edge_index)
        x4 = F.dropout(x, 0.6)
        x4 = self.conv4(x4, edge_index)
        x5 = F.dropout(x, 0.6)
        x5 = self.conv5(x5, edge_index)
        x6 = F.dropout(x, 0.6)
        x6 = self.conv6(x6, edge_index)
        x7 = F.dropout(x, 0.6)
        x7 = self.conv7(x7, edge_index)
        x8 = F.dropout(x, 0.6)
        x8 = self.conv8(x8, edge_index)
        x = 1 / 8 * (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8)
        eluLayer = torch.nn.ELU()
        x = eluLayer(x)
        x = F.dropout(x, 0.6)
        x = self.conv9(x, edge_index)
        # return F.softmax(x, dim=1)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, weight_decay=5e-4)

model.train()
for epoch in range(500):
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
