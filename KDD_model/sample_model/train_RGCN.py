import os, sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from compare_with_baseline.baseline_utils import preprocess, process_data_for_nettack
from compare_with_baseline.RGCN_model import RGCN


seed = 0
np.random.rand(seed)
torch.manual_seed(seed)

is_cuda = "cuda:1" if torch.cuda.is_available() else "cpu"
device = torch.device(is_cuda if isinstance(is_cuda, str) else "cuda" if is_cuda else "cpu")

data_type = "Cora"
path = os.path.join(os.path.dirname(os.path.realpath("__file__")), '..', 'data', data_type)
dataset = Planetoid(path, data_type, transform=T.NormalizeFeatures())
data = dataset[0]
data = process_data_for_nettack(data).to(device)    # torch.tensor() -> spicy.sparse.csr_matrix

adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, device=device)

rgcn_model = RGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nclass=labels.max()+1, nhid=16, device=device)
rgcn_model = rgcn_model.to(device)
rgcn_model.fit(features, adj, labels, idx_train, idx_val, train_iters=1)


"""
def test(self, idx_test):
    ''' test performance '''
    output = self.output
    loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
    acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()
"""