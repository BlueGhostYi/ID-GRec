"""
PyTorch Implementation of LightGCN++
Revisiting LightGCN: Unexpected Inflexibility, Inconsistency, and A Remedy Towards Improved Recommendation
Lee et al. RecSys'24
"""

import torch
from torch import nn
import scipy.sparse as sp
import numpy as np
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer


class LightGCN_pp(nn.Module):
    def __init__(self, config, dataset, device):
        super(LightGCN_pp, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.gamma = float(self.config['gamma'])
        self.alpha = float(self.config['alpha'])
        self.beta = float(self.config['beta'])

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.config['embedding_size']))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.Graph = self.get_sparse_graph()  # sparse matrix
        self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)  # sparse tensor
        self.Graph = self.Graph.coalesce().to(self.device)  # Sort the edge index and remove redundancy

        self.activation = nn.Sigmoid()

    def get_sparse_graph(self):
        try:
            norm_adjacency = sp.load_npz(self.dataset.path + f'/pre_A_{self.alpha}_{self.beta}.npz')
            print("\t Adjacency matrix loading completed.")
        except:
            adjacency_matrix = sp.dok_matrix((self.dataset.num_nodes, self.dataset.num_nodes), dtype=np.float32)
            adjacency_matrix = adjacency_matrix.tolil()
            R = self.dataset.user_item_net.todok()
            adjacency_matrix[:self.dataset.num_users, self.dataset.num_users:] = R
            adjacency_matrix[self.dataset.num_users:, :self.dataset.num_users] = R.T
            adjacency_matrix = adjacency_matrix.todok()

            rowsum_left = np.array(adjacency_matrix.sum(axis=1)) ** -self.alpha
            rowsum_right = np.array(adjacency_matrix.sum(axis=1)) ** -self.beta

            d_inv_left = rowsum_left.flatten()
            d_inv_left[np.isinf(d_inv_left)] = 0.

            d_inv_right = rowsum_right.flatten()
            d_inv_right[np.isinf(d_inv_right)] = 0.

            d_mat_left = sp.diags(d_inv_left)
            d_mat_right = sp.diags(d_inv_right)

            norm_adjacency = d_mat_left.dot(adjacency_matrix)
            norm_adjacency = norm_adjacency.dot(d_mat_right)
            norm_adjacency = norm_adjacency.tocsr()

            sp.save_npz(self.dataset.path + f'/pre_A_{self.alpha}_{self.beta}.npz', norm_adjacency)
            print("\t Adjacency matrix constructed.")

        return norm_adjacency

    def aggregate(self):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # no dropout
        embeddings = [all_embedding]

        for layer in range(int(self.config['GCN_layer'])):
            norm = torch.norm(all_embedding, dim=1) + 1e-12
            all_embedding = all_embedding / norm[:, None]

            all_embedding = torch.sparse.mm(self.Graph, all_embedding)
            embeddings.append(all_embedding)

        zero_embedding = embeddings[0]
        prop_embeddings = torch.mean(torch.stack(embeddings[1:], dim=1), dim=1)

        final_embeddings = (self.gamma * zero_embedding) + ((1 - self.gamma) * prop_embeddings)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def forward(self, user, positive, negative):
        all_user_embeddings, all_item_embeddings = self.aggregate()

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        loss_list = [bpr_loss, reg_loss]

        return loss_list

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings = self.aggregate()
        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = LightGCN_pp(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)