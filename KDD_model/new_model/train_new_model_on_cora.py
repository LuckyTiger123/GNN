import torch
import torch_geometric
import torch_geometric.transforms as T
from torch import Tensor
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.typing import Adj
import KDD_model.new_model.kdd_new_model as FS

# device init
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/Cora", name="Cora", transform=T.NormalizeFeatures())
data = dataset[0].to(device=device)

# init add rate and drop rate
add_rate = 0
drop_rate = 0.4


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


# init model and optimizer
model = TwoLayerFSL(dataset.num_features, dataset.num_classes).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# model train
model.train()
for epoch in range(300):
    edge_index = FS.add_edge(data.edge_index, add_rate=add_rate)
    optimizer.zero_grad()
    out = model(data.x, edge_index, drop_rate=drop_rate)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    print('on the {} epoch, the loss is {}'.format(epoch, float(loss)))
    loss.backward()
    optimizer.step()

# model test
model.eval()
_, pred = model(data.x, data.edge_index, drop_rate=0).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))
