import torch
import math
import random
import torch_geometric
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Parameter
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import KDD_model.sample_model.sample_model as SM

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/Cora", name="Cora", transform=T.NormalizeFeatures())
data = dataset[0].to(device=device)

# model init
model = SM.SampleModel(dataset.num_features, dataset.num_classes).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# train
model.train()
out = model(data.x, data.edge_index)

