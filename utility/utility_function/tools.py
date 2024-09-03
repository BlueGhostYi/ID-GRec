import os
import torch
import random
import numpy as np
import scipy.sparse as sp


def set_seed(seed):

    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def read_configuration(filename, model):
    if not os.path.exists(filename):
        print("\tThe path does not have a configuration file for " + model + ".")
        raise IOError
    else:
        with open(filename, "r") as f:
            config = dict()
            line = f.readline()
            while line is not None and line != "":
                try:
                    name, value = line.strip().split("=")
                    config[name.strip()] = value.strip()
                except ValueError:
                    print("\tConfiguration file format error.")
                line = f.readline()
        return config


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('Inputs to shuffle must have the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def mini_batch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 1024)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def create_adj_mat(inter_graph, aug_type, ssl_rate):
    graph_shape = inter_graph.get_shape()
    node_number = graph_shape[0] + graph_shape[1]
    user_index, item_index = inter_graph.nonzero()

    if aug_type == 'nd':
        raise NotImplementedError("The method does not implemented.")
    elif aug_type in ['ed', 'rw']:
        edge_number = inter_graph.count_nonzero()

        keep_index = random.sample(range(edge_number), k=int((1 - ssl_rate) * edge_number))
        user_index = np.array(user_index)[keep_index]
        item_index = np.array(item_index)[keep_index]
        ratings = np.ones_like(user_index, dtype=np.float32)
        new_graph = sp.csr_matrix((ratings, (user_index, item_index + graph_shape[0])), shape=(node_number, node_number))

    adjacency_matrix = new_graph + new_graph.T

    row_sum = np.array(adjacency_matrix.sum(axis=1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    degree_matrix = sp.diags(d_inv)

    norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()

    return norm_adjacency


def convert_sp_mat_to_sp_tensor(sp_mat):
    """
        coo.row: x in user-item graph
        coo.col: y in user-item graph
        coo.data: [value(x,y)]
    """
    coo = sp_mat.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()

    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    value = torch.FloatTensor(coo.data)
    # from a sparse matrix to a sparse float tensor
    sp_tensor = torch.sparse.FloatTensor(index, value, torch.Size(coo.shape))
    return sp_tensor
