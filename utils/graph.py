from typing import Union

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import eigsh
from torch import sparse


def normalized_laplacian(w: np.ndarray):
    w = sp.coo_matrix(w)
    d = np.array(w.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(w.shape[0]) - w.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt).tocoo()


def random_walk_matrix(w):
    w = sp.coo_matrix(w)
    d = np.array(w.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(w).tocoo()


def scaled_laplacian(w: np.ndarray, lambda_max: Union[float, None] = 2., undirected: bool = True):
    if undirected:
        w = np.maximum(w, w.T)
    l = normalized_laplacian(w)
    if lambda_max is None:
        lambda_max, _ = eigsh(l, 1, which='LM')
        lambda_max = lambda_max[0]
    m, _ = l.shape
    i = sp.identity(m, format='coo', dtype=l.dtype)
    l = (2 / lambda_max * l) - i
    return l.tocoo()


def cheb_poly_approx(lp, k_hop, n):
    """
    Chebyshev polynomials approximation function.
    :param lp: np.ndarray, [n_route, n_route], graph Laplacian.
    :param k_hop: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks, n_route].
    """
    l0, l1 = np.identity(n), np.copy(lp)

    if k_hop > 1:
        l_list = [np.copy(l0), np.copy(l1)]
        for i in range(k_hop - 2):
            ln = 2 * np.matmul(lp, l1) - l0
            l_list.append(np.copy(ln))
            l0, l1 = np.copy(l1), np.copy(ln)
        # L_lsit Ks, [n, n], [n, Ks, n]
        return np.stack(l_list, axis=1)
    elif k_hop == 1:
        return l0.reshape((n, 1, n))
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{k_hop}".')


def first_approx(w, n):
    """
    1st-order approximation function.
    :param w: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    """
    a = w + np.identity(n)
    d = np.sum(a, axis=1)
    sinv_d = np.sqrt(np.linalg.inv(np.diag(d)))
    # refer to Eq.5
    return np.identity(n) + np.matmul(np.matmul(sinv_d, a), sinv_d)


def convert_scipy_to_torch_sparse(w: sp.coo_matrix):
    """
    build pytorch sparse tensor from scipy sparse matrix
    reference: https://stackoverflow.com/questions/50665141
    :return:
    """
    shape = w.shape
    i = torch.LongTensor(np.vstack((w.row, w.col)).astype(int))
    v = torch.FloatTensor(w.data)
    return sparse.FloatTensor(i, v, torch.Size(shape))
