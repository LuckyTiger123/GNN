import math
import torch
import random
import torch_geometric
from torch import Tensor

adj = torch.tensor([[0, 1, 0.1, 0.1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.float, requires_grad=True)
filter_matrix = torch.where(adj > 0.5, adj, torch.zeros_like(adj))
print(filter_matrix)
