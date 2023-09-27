from __future__ import absolute_import

import torch
import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=False, eye=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    if eye:
        adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    else:
        adj_mx = normalize(adj_mx)

    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


def adj_mx_from_edges_binary(num_pts, edges):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


def h_mx_from_edges_binary(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


def get_sketch_setting():
    return [[0, 1], [1, 2], [2, 3], [3, 4],
                [0, 5], [5, 6], [6, 7], [7, 8],
                [0, 9], [9, 10], [10, 11], [11, 12],
                [0, 13], [13, 14], [14,  15], [15, 16],
                [0, 17], [17, 18], [18, 19], [19, 20]]


def get_hierarchy_sketch():
    return [
               [0, 1], [1, 2], [2, 3], [3, 4],
               [0, 5], [5, 6], [6, 7], [7, 8],
               [0, 9], [9, 10], [10, 11], [11, 12],
               [0, 13], [13, 14], [14, 15], [15, 16],
               [0, 17], [17, 18], [18, 19], [19, 20]
           ], \
           [
                [0, 1], [1, 2],
                [0, 3], [3, 4],
                [0, 5], [5, 6],
                [0, 7], [7, 8],
                [0, 9], [9, 10]
           ], \
           [
               [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]
           ], \
           [[0, 0]]


def get_hierarchy_mapping():
        return [
            [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]],\
            [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], \
            [[0, 1, 2, 3, 4, 5]],
        ]