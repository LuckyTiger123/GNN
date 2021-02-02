import torch
import math
import random
import torch_geometric
import time
import torch_sparse
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter


class GCNModel(torch.nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv()
