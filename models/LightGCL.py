"""
Created on June 9, 2023
PyTorch Implementation of LightGCL
LightGCL: SIMPLE YET EFFECTIVE GRAPH CONTRASTIVE LEARNING FOR RECOMMENDATION  Cai et al. ICLR'23
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph


class LightGCL(nn.Module):
    def __init__(self, config, dataset, device):
        super(LightGCL, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.ssl_lambda = float(self.config['ssl_lambda'])
        self.temperature = float(self.config['temperature'])
        self.num_layers = int(self.config['GCN_layer'])
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.config['embedding_size']))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.Graph = utility.utility_data.data_graph.sparse_adjacency_matrix_R(self.dataset)
        self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)
        self.Graph = self.Graph.coalesce().to(self.device)

        svd_u, s, svd_v = torch.svd_lowrank(self.Graph, q=int(self.config['svd_q']))

        u_mul_s = svd_u @ torch.diag(s)
        v_mul_s = svd_v @ torch.diag(s)

        self.u_T = svd_u.T
        self.v_T = svd_v.T
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        del s

        self.act = nn.LeakyReLU(0.5)
        self.activation = nn.Sigmoid()

    def sparse_dropout(self, matrix, dropout):
        indices = matrix.indices()
        values = nn.functional.dropout(matrix.values(), p=dropout)
        size = matrix.size()
        return torch.sparse.FloatTensor(indices, values, size)

    def aggregate(self):
        user_embeddings = [None] * (self.num_layers + 1)
        item_embeddings = [None] * (self.num_layers + 1)
        z_user_embeddings = [None] * (self.num_layers + 1)
        z_item_embeddings = [None] * (self.num_layers + 1)
        g_user_embeddings = [None] * (self.num_layers + 1)
        g_item_embeddings = [None] * (self.num_layers + 1)

        user_embeddings[0] = self.user_embedding.weight
        item_embeddings[0] = self.item_embedding.weight
        g_user_embeddings[0] = self.user_embedding.weight
        g_item_embeddings[0] = self.item_embedding.weight

        for layer in range(1, self.num_layers + 1):
            # graph = self.sparse_dropout(self.Graph, self.dropout)
            graph = self.Graph
            z_user_embeddings[layer] = torch.sparse.mm(graph, item_embeddings[layer - 1])
            z_item_embeddings[layer] = torch.sparse.mm(graph.transpose(0, 1), user_embeddings[layer - 1])

            ut_user_embedding = self.u_T @ user_embeddings[layer - 1]
            vt_item_embedding = self.v_T @ item_embeddings[layer - 1]

            g_user_embeddings[layer] = self.u_mul_s @ vt_item_embedding
            g_item_embeddings[layer] = self.v_mul_s @ ut_user_embedding

            user_embeddings[layer] = z_user_embeddings[layer]
            item_embeddings[layer] = z_item_embeddings[layer]

        final_user_embeddings = torch.stack(user_embeddings, dim=1)
        final_item_embeddings = torch.stack(item_embeddings, dim=1)
        final_user_embeddings = torch.sum(final_user_embeddings, dim=1)
        final_item_embeddings = torch.sum(final_item_embeddings, dim=1)

        final_g_user_embeddings = torch.stack(g_user_embeddings, dim=1)
        final_g_item_embeddings = torch.stack(g_item_embeddings, dim=1)
        final_g_user_embeddings = torch.sum(final_g_user_embeddings, dim=1)
        final_g_item_embeddings = torch.sum(final_g_item_embeddings, dim=1)

        return final_user_embeddings, final_item_embeddings, final_g_user_embeddings, final_g_item_embeddings

    def forward(self, user, positive, negative):
        all_user_embeddings, all_item_embeddings, all_g_user_embeddings, all_g_item_embeddings = self.aggregate()

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        neg_score = torch.log(torch.exp(all_g_user_embeddings[user] @ all_user_embeddings.T / self.temperature).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(all_g_item_embeddings[positive] @ all_item_embeddings.T / self.temperature).sum(1) + 1e-8).mean()

        pos_score = (torch.clamp((all_user_embeddings[user] * all_g_user_embeddings[user]).sum(1) / self.temperature, -5.0, 5.0)).mean()
        pos_score += (torch.clamp((all_item_embeddings[positive] * all_g_item_embeddings[positive]).sum(1) / self.temperature, -5.0, 5.0)).mean()

        ssl_loss = self.ssl_lambda * (-pos_score + neg_score)

        loss_list = [bpr_loss, reg_loss, ssl_loss]

        return loss_list

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings, _, _ = self.aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger
        self.model = LightGCL(config, dataset, device)

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)