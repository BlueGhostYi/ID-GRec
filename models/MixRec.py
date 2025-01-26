"""
PyTorch Implementation of MixRec
MixRec: Individual and Collective Mixing Empowers Data Augmentation for Recommender Systems  Zhang et al. WWW'25
For more information, please refer to: https://github.com/BlueGhostYi/MixRec
"""

import torch
from torch import nn
import numpy as np
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph


class MixRec(nn.Module):
    def __init__(self, config, dataset, device):
        super(MixRec, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.ssl_lambda = float(self.config['ssl_lambda'])
        self.alpha = float(self.config['alpha'])
        self.beta = float(self.config['beta'])
        self.gamma = float(self.config['gamma'])
        self.temperature = float(self.config['temperature'])

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.config['embedding_size']))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.Graph = utility.utility_data.data_graph.sparse_adjacency_matrix(self.dataset)
        self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)
        self.Graph = self.Graph.coalesce().to(self.device)

        self.activation = nn.Sigmoid()

    def aggregate(self, user_embedding=None, item_embedding=None):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # no dropout
        embeddings = []

        for layer in range(int(self.config['GCN_layer'])):
            all_embedding = torch.sparse.mm(self.Graph, all_embedding)
            embeddings.append(all_embedding)

        final_embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.sum(final_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def mix_aggregate(self, user_beta, item_beta):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # no dropout
        embeddings = []

        for layer in range(int(self.config['GCN_layer'])):
            all_embedding = torch.sparse.mm(self.Graph, all_embedding)

            users_emb, items_emb = torch.split(all_embedding, [self.dataset.num_users, self.dataset.num_items])

            user_index = torch.randperm(users_emb.shape[0]).cuda()
            item_index = torch.randperm(items_emb.shape[0]).cuda()

            users_emb_2 = users_emb[user_index, :]
            items_emb_2 = items_emb[item_index, :]

            users_emb = user_beta * users_emb + (1 - user_beta) * users_emb_2
            items_emb = item_beta * items_emb + (1 - item_beta) * items_emb_2

            all_embedding = torch.cat([users_emb, items_emb])

            embeddings.append(all_embedding)

        final_embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def forward(self, user, positive, negative):
        all_user_embeddings, all_item_embeddings = self.aggregate()

        user_embeddings = all_user_embeddings[user.long()]
        pos_embeddings = all_item_embeddings[positive.long()]

        user_beta = np.random.beta(self.alpha, self.beta)
        item_beta = np.random.beta(self.alpha, self.beta)

        neg_beta = torch.FloatTensor(np.random.dirichlet([self.gamma] * negative.shape[0], 1)).unsqueeze(-1).to(self.device)

        # [B, 1] * [B, dim] = [B, neg, dim]
        mix_user_embeddings = (neg_beta * user_embeddings).sum(dim=1)
        mix_pos_embeddings = (neg_beta * pos_embeddings).sum(dim=1)

        user_index = torch.randperm(user_embeddings.shape[0]).cuda()
        item_index = torch.randperm(pos_embeddings.shape[0]).cuda()

        user_embeddings_2 = user_embeddings[user_index, :]
        pos_embeddings_2 = pos_embeddings[item_index, :]

        cl_user_embeddings = user_beta * user_embeddings + (1 - user_beta) * user_embeddings_2
        cl_item_embeddings = item_beta * pos_embeddings + (1 - item_beta) * pos_embeddings_2

        neg_embeddings = all_item_embeddings[negative.long()]
        neg_embeddings_2 = neg_embeddings[item_index, :]

        mix_neg_embeddings_2 = item_beta * neg_embeddings + (1 - item_beta) * neg_embeddings_2

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_embeddings, pos_embeddings, neg_embeddings)
        bpr_loss_2 = losses.get_InfoNCE_loss_all(user_embeddings, pos_embeddings, mix_neg_embeddings_2, 1.0)

        bpr_loss = item_beta * bpr_loss
        bpr_loss_2 = (1 - item_beta) * bpr_loss_2

        reg_loss = losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        cl_user_embeddings_2 = torch.cat([user_embeddings_2, mix_user_embeddings])
        cl_item_embeddings_2 = torch.cat([pos_embeddings_2, mix_pos_embeddings])

        cl_user_embeddings_3 = torch.cat([user_embeddings, mix_user_embeddings])
        cl_item_embeddings_3 = torch.cat([pos_embeddings, mix_pos_embeddings])

        user_loss = losses.get_InfoNCE_loss_all(user_embeddings, cl_user_embeddings, cl_user_embeddings_2, self.temperature)
        user_loss_2 = losses.get_InfoNCE_loss_all(user_embeddings_2, cl_user_embeddings, cl_user_embeddings_3, self.temperature)
        user_ssl_loss = user_beta * user_loss + (1 - user_beta) * user_loss_2

        item_loss = losses.get_InfoNCE_loss_all(pos_embeddings, cl_item_embeddings, cl_item_embeddings_2, self.temperature)
        item_loss_2 = losses.get_InfoNCE_loss_all(pos_embeddings_2, cl_item_embeddings, cl_item_embeddings_3, self.temperature)
        item_ssl_loss = item_beta * item_loss + (1 - item_beta) * item_loss_2

        ssl_loss = self.ssl_lambda * (user_ssl_loss + item_ssl_loss)

        loss_list = [bpr_loss, bpr_loss_2, reg_loss, ssl_loss]

        return loss_list

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings = self.aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = MixRec(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)
