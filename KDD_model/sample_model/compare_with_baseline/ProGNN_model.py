import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

from deeprobust.graph.defense.pgd import PGD, prox_operators

from baseline_utils import accuracy


class ProGNN:
    """ ProGNN (Properties Graph Neural Network). See more details in Graph Structure Learning for Robust Graph Neural Networks, KDD 2020, https://arxiv.org/abs/2005.10203.
        See details in https://github.com/ChandlerBang/Pro-GNN.

    Parameters
    ----------
    model:
        model: The backbone GNN model in ProGNN
    args:
        model configs
    device: str
        'cpu' or 'cuda'.
    """

    def __init__(self, model, device):
        super(ProGNN, self).__init__()
        self.device = device
        self.model = model.to(device)
        self.features = None
        self.labels = None
        self.adj = None

        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.test_acc_under_best_val = 0

        # params
        self.symmetric = False
        self.lr = 0.01
        self.lr_adj = 0.01
        self.weight_decay = 5e-4
        self.alpha = 5e-4
        self.beta = 2
        self.gamma = 1
        self.lambda_ = 1e-3
        self.phi = 0
        self.inner_steps = 2
        self.outer_steps = 1

    def fit(self, features, adj, labels, idx_train, idx_val, idx_test, train_iters=600):
        """Train Pro-GNN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """
        self.features = features
        self.labels = labels
        self.adj = adj

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr, weight_decay=self.weight_decay)
        estimator = EstimateAdj(adj, symmetric=self.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(),
                                       momentum=0.9, lr=self.lr_adj)

        self.optimizer_l1 = PGD(estimator.parameters(),
                                proxs=[prox_operators.prox_l1],
                                lr=self.lr_adj, alphas=[self.alpha])

        warnings.warn(
            "If you find the nuclear proximal operator runs too slow, you can modify line 77 to use "
            "prox_operators.prox_nuclear_cuda instead of prox_operators.prox_nuclear to perform the proximal on GPU. "
            "See details in https://github.com/ChandlerBang/Pro-GNN/issues/1")
        self.optimizer_nuclear = PGD(estimator.parameters(),
                                     proxs=[prox_operators.prox_nuclear_cuda],
                                     lr=self.lr_adj, alphas=[self.beta])

        # Train model
        for epoch in range(train_iters):
            for i in range(int(self.outer_steps)):
                self._train_adj(epoch, features, adj, labels, idx_train, idx_val, idx_test)

            for i in range(int(self.inner_steps)):
                self._train_gcn(epoch, features, estimator.estimated_adj, labels, idx_train, idx_val, idx_test)

        print("Optimization Finished!")

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

    def _train_gcn(self, epoch, features, adj, labels, idx_train, idx_val, idx_test):
        estimator = self.estimator
        adj = estimator.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately, deactivates dropout during validation run.
        self.model.eval()
        output = self.model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            self.test_acc_under_best_val = acc_test

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.weights = deepcopy(self.model.state_dict())

    def _train_adj(self, epoch, features, adj, labels, idx_train, idx_val, idx_test):
        estimator = self.estimator
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        normalized_adj = estimator.normalize()

        if self.lambda_:
            loss_smooth_feat = self.feature_smoothing(estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        output = self.model(features, normalized_adj)
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        loss_symmetric = torch.norm(estimator.estimated_adj - estimator.estimated_adj.t(), p="fro")

        loss_diffiential = loss_fro + self.gamma * loss_gcn + self.lambda_ * loss_smooth_feat + self.phi * loss_symmetric
        loss_diffiential.backward()

        self.optimizer_adj.step()
        loss_nuclear = 0 * loss_fro
        if self.beta != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = loss_fro \
                     + self.gamma * loss_gcn \
                     + self.alpha * loss_l1 \
                     + self.beta * loss_nuclear \
                     + self.phi * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(
            estimator.estimated_adj.data, min=0, max=1))

        # Evaluate validation set performance separately, deactivates dropout during validation run.
        self.model.eval()
        normalized_adj = estimator.normalize()
        output = self.model(features, normalized_adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        print('ProGNN: ',
              'Epoch: {:04d}'.format(epoch + 1),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            self.test_acc_under_best_val = acc_test

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.weights = deepcopy(self.model.state_dict())
            self.test_acc_under_best_val = acc_test

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj) / 2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
