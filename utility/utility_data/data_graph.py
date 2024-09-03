import numpy as np
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')


def sparse_adjacency_matrix_with_self(data):
    try:
        norm_adjacency = sp.load_npz(data.path + '/pre_A_with_self.npz')
        print("\t Adjacency matrix loading completed.")
    except:
        adjacency_matrix = sp.dok_matrix((data.num_nodes, data.num_nodes), dtype=np.float32)
        adjacency_matrix = adjacency_matrix.tolil()
        R = data.user_item_net.todok()

        adjacency_matrix[:data.num_users, data.num_users:] = R
        adjacency_matrix[data.num_users:, :data.num_users] = R.T
        adjacency_matrix = adjacency_matrix.todok()
        adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])

        row_sum = np.array(adjacency_matrix.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        degree_matrix = sp.diags(d_inv)

        norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
        sp.save_npz(data.path + '/pre_A_with_self', norm_adjacency)
        print("\t Adjacency matrix constructed.")

    return norm_adjacency


def sparse_adjacency_matrix(data):
    try:
        norm_adjacency = sp.load_npz(data.path + '/pre_A.npz')
        print("\t Adjacency matrix loading completed.")
    except:
        adjacency_matrix = sp.dok_matrix((data.num_nodes, data.num_nodes), dtype=np.float32)
        adjacency_matrix = adjacency_matrix.tolil()
        R = data.user_item_net.todok()

        adjacency_matrix[:data.num_users, data.num_users:] = R
        adjacency_matrix[data.num_users:, :data.num_users] = R.T
        adjacency_matrix = adjacency_matrix.todok()

        row_sum = np.array(adjacency_matrix.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        degree_matrix = sp.diags(d_inv)

        norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
        sp.save_npz(data.path + '/pre_A', norm_adjacency)
        print("\t Adjacency matrix constructed.")

    return norm_adjacency


def sparse_adjacency_matrix_R(data):
    try:
        norm_adjacency = sp.load_npz(data.path + '/pre_R.npz')
        print("\t Adjacency matrix loading completed.")
    except:
        adjacency_matrix = data.user_item_net

        row_sum = np.array(adjacency_matrix.sum(axis=1))
        row_d_inv = np.power(row_sum, -0.5).flatten()
        row_d_inv[np.isinf(row_d_inv)] = 0.
        row_degree_matrix = sp.diags(row_d_inv)

        col_sum = np.array(adjacency_matrix.sum(axis=0))
        col_d_inv = np.power(col_sum, -0.5).flatten()
        col_d_inv[np.isinf(col_d_inv)] = 0.
        col_degree_matrix = sp.diags(col_d_inv)

        norm_adjacency = row_degree_matrix.dot(adjacency_matrix).dot(col_degree_matrix).tocsr()
        sp.save_npz(data.path + '/pre_R', norm_adjacency)
        print("\t Adjacency matrix constructed.")

    return norm_adjacency
