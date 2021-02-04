import torch
import random
import torch_geometric
import torch_sparse
import numpy as np
from torch import Tensor
import scipy.sparse as sp
from torch.autograd import Variable
from numbers import Number
from torch_geometric.typing import Adj, OptTensor


# noise generate
# add noisy edge change_type {1,2,3} means operation {add edge/remove edge/add and remove edge}
def add_noise_edge(edge_index: Adj, add_rate: float = 0.00005, remove_rate: float = 0.1) -> Tensor:
    dense_matrix = torch.squeeze(torch_geometric.utils.to_dense_adj(edge_index)).to(device=edge_index.device)
    dense_matrix = dense_matrix * (1 - remove_rate)
    dense_matrix = torch.bernoulli(dense_matrix).long()
    dense_matrix = dense_matrix * (1 - add_rate) + add_rate
    dense_matrix = torch.bernoulli(dense_matrix).long()
    edge_final = torch_geometric.utils.dense_to_sparse(torch.squeeze(dense_matrix, dim=0))[0]
    return edge_final


# add edge with parameter relates to existing edges
def add_related_edge(edge_index: Adj, add_rate: float = 0.1, random_seed: int = 0) -> Tensor:
    edge_num = int(edge_index.size(1))
    add_num = int(edge_num * add_rate / 2)
    dense_matrix = torch.squeeze(torch_geometric.utils.to_dense_adj(edge_index)).to(device=edge_index.device)
    edge_num = dense_matrix.size(0)
    random.seed(random_seed)
    for i in range(add_num):
        source_node = random.randint(0, edge_num - 1)
        des_node = random.randint(0, edge_num - 1)
        dense_matrix[source_node][des_node] = 1
        dense_matrix[des_node][source_node] = 1
    edge_final = torch_geometric.utils.dense_to_sparse(dense_matrix)[0]
    return edge_final


# drop edge with parameter relates to existing edges
def drop_related_edge(edge_index: Adj, remove_rate: float = 0.1) -> Tensor:
    dense_matrix = torch.squeeze(torch_geometric.utils.to_dense_adj(edge_index)).to(device=edge_index.device)
    dense_matrix = dense_matrix * (1 - remove_rate)
    dense_matrix = torch.bernoulli(dense_matrix).long()
    # use upper triangle
    adj = dense_matrix.cpu().numpy()
    adj = np.triu(adj)
    adj += adj.T
    dense_matrix = torch.from_numpy(adj).to(device=edge_index.device)
    edge_final = torch_geometric.utils.dense_to_sparse(torch.squeeze(dense_matrix, dim=0))[0]
    return edge_final


# flip edge with parameter relates to existing edges
def flip_related_edge(edge_index: Adj, flip_rate: float = 0.1, random_seed: int = 0) -> Tensor:
    edge_num = int(edge_index.size(1))
    flip_num = int(edge_num * flip_rate / 2)
    dense_matrix = torch.squeeze(torch_geometric.utils.to_dense_adj(edge_index)).long().to(device=edge_index.device)
    edge_num = dense_matrix.size(0)
    random.seed(random_seed)
    for i in range(flip_num):
        source_node = random.randint(0, edge_num - 1)
        des_node = random.randint(0, edge_num - 1)
        dense_matrix[source_node][des_node] = dense_matrix[source_node][des_node] ^ 1
        dense_matrix[des_node][source_node] = dense_matrix[des_node][source_node] ^ 1
    edge_final = torch_geometric.utils.dense_to_sparse(dense_matrix)[0]
    return edge_final


# flip edge with parameter relates to existing edges
def average_flip_related_edge(edge_index: Adj, flip_rate: float = 0.1, random_seed: int = 0) -> Tensor:
    edge_num = int(edge_index.size(1))
    flip_add_num = int(edge_num * flip_rate / 4)
    dense_matrix = torch.squeeze(torch_geometric.utils.to_dense_adj(edge_index)).long().to(device=edge_index.device)
    edge_num = dense_matrix.size(0)
    random.seed(random_seed)
    dense_matrix = dense_matrix * (1 - flip_rate / 2)
    dense_matrix = torch.bernoulli(dense_matrix).long()
    # use upper triangle
    adj = dense_matrix.cpu().numpy()
    adj = np.triu(adj)
    adj += adj.T
    dense_matrix = torch.from_numpy(adj).to(device=edge_index.device)
    for i in range(flip_add_num):
        source_node = random.randint(0, edge_num - 1)
        des_node = random.randint(0, edge_num - 1)
        dense_matrix[source_node][des_node] = 1
        dense_matrix[des_node][source_node] = 1
    edge_final = torch_geometric.utils.dense_to_sparse(dense_matrix)[0]
    return edge_final


# flip sparse matrix
def average_flip_related_edge_with_sparse_matrix(edge_index: Adj, node_num: int, flip_rate: float = 0.5,
                                                 random_seed: int = 0) -> Tensor:
    random.seed(random_seed)
    edge_num = int(edge_index.size(1))
    change_edge_num = int(edge_num * flip_rate / 4)
    edge_set = set()
    for i in range(edge_index.size(-1)):
        if edge_index[0][i] > edge_index[1][i]:
            edge_set.add(int(edge_index[1][i]) * 100000 + int(edge_index[0][i]))
        else:
            edge_set.add(int(edge_index[0][i]) * 100000 + int(edge_index[1][i]))

    edge_set = set(random.sample(list(edge_set), int(edge_num / 2 - change_edge_num)))

    i = 0
    while i < change_edge_num:
        source_node = random.randint(0, node_num - 1)
        des_node = random.randint(0, node_num - 1)
        if source_node == des_node:
            continue
        if source_node > des_node:
            item = des_node * 100000 + source_node
        else:
            item = source_node * 100000 + des_node

        if item not in edge_set:
            edge_set.add(item)
            i += 1

    np_edge_index = np.array([[int(item / 100000), int(item % 100000)] for item in edge_set])
    np_edge_index_invert = np.array([[int(item % 100000), int(item / 100000)] for item in edge_set])
    final_edge_index = torch.from_numpy(np.append(np_edge_index, np_edge_index_invert, axis=0)).T.to(
        device=edge_index.device)

    final_edge_index = \
        torch_sparse.spspmm(final_edge_index, torch.ones(final_edge_index.size(-1)).to(device=edge_index.device),
                            torch.tensor([[i for i in range(node_num)], [i for i in range(node_num)]]).to(
                                device=edge_index.device),
                            torch.ones(node_num).to(device=edge_index.device), node_num, node_num, node_num,
                            coalesced=True)[0]

    return final_edge_index


# flip feature in cora with a certain rate
def flip_feature_for_dataset(x: Tensor, flip_rate: float = 0.01) -> Tensor:
    # we use two step to flip: step1 selects the flip node, step2 selects the flip dimension
    if flip_rate == 0:
        return x
    step_flip_rate = pow(flip_rate, 0.5)
    change_node = torch.diag(torch.bernoulli(torch.zeros(x.size(0)) + step_flip_rate)).to(device=x.device)
    change_matrix = torch.bernoulli(
        torch.mm(change_node, torch.ones_like(x, device=x.device)) * step_flip_rate).long().to(
        device=x.device)
    result = x.long() ^ change_matrix
    return result.float()


def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.Tensor) or isinstance(array, torch.FloatTensor) or isinstance(array,
                                                                                                 torch.LongTensor) or isinstance(
            array, torch.ByteTensor) or \
                isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(
            array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == (1,):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list


def edge_index_2_csr(edge_index, size):
    """Edge index (PyG COO format) transformed to csr format."""
    csr_matrix = sp.csr_matrix(
        (np.ones(edge_index.shape[1]), to_np_array(edge_index)),
        shape=(size, size))
    return csr_matrix


def process_data_for_nettack_on_noisy_graph(x, edge_index, data):
    data.features = sp.csr_matrix(to_np_array(x))
    data.adj = edge_index_2_csr(edge_index, size=data.x.shape[0])
    data.labels = to_np_array(data.y)
    data.idx_train = np.where(to_np_array(data.train_mask))[0]
    data.idx_val = np.where(to_np_array(data.val_mask))[0]
    data.idx_test = np.where(to_np_array(data.test_mask))[0]
    return data


def normalize_feature(mx):
    """Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(sparse_mx.shape))


def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor, and normalize the input data.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    preprocess_adj : bool
        whether to normalize the adjacency matrix
    preprocess_feature : bool
        whether to normalize the feature matrix
    sparse : bool
       whether to return sparse tensor
    device : str
        'cpu' or 'cuda'
    """

    if preprocess_adj:
        adj_norm = normalize_adj(adj)

    if preprocess_feature:
        features = normalize_feature(features)

    labels = torch.LongTensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features.todense()))
        adj = torch.FloatTensor(adj.todense())
    return adj.to(device), features.to(device), labels.to(device)
