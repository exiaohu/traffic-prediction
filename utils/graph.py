import numpy as np
from scipy.sparse.linalg import eigs


def scaled_laplacian(w):
    """
    Normalized graph Laplacian function.
    :param w: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.ndarray, [n_route, n_route].
    """
    # d ->  diagonal degree matrix
    n, d = w.shape[0], np.sum(w, axis=1)
    # L -> graph Laplacian
    lp = -w
    lp[np.diag_indices_from(lp)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                lp[i, j] = lp[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(lp, k=1, which='LR')[0][0].real
    return 2 * lp / lambda_max - np.identity(n)


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
