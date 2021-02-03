import numpy as np
import scipy.sparse as sp
import torch
from torch.autograd import Variable
from numbers import Number


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


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


def process_data_for_nettack(data):
    data.features = sp.csr_matrix(to_np_array(data.x))
    data.adj = edge_index_2_csr(data.edge_index, size=data.x.shape[0])
    data.labels = to_np_array(data.y)
    data.idx_train = np.where(to_np_array(data.train_mask))[0]
    data.idx_val = np.where(to_np_array(data.val_mask))[0]
    data.idx_test = np.where(to_np_array(data.test_mask))[0]
    return data


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


def to_tensor(adj, features, labels=None, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features))

    if labels is None:
        return adj.to(device), features.to(device)
    else:
        labels = torch.LongTensor(labels)
        return adj.to(device), features.to(device), labels.to(device)
