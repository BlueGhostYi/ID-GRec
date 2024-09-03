"""
PyTorch Implementation of XSimGCL
XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation  Yu et al. arXiv'22
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph


class XSimGCL(nn.Module):
    def __init__(self, config, dataset, device):
        super(XSimGCL, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.ssl_lambda = float(self.config['ssl_lambda'])
        self.epsilon = float(self.config['epsilon'])
        self.temperature = float(self.config['temperature'])
        self.cl_layer = int(self.config['cl_layer'])
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

    def aggregate(self, perturbed=False):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # no dropout
        # Initial embedding is not included in the official implementation of XSimGCL
        embeddings = []

        all_embedding_cl = all_embedding

        for layer in range(int(self.config['GCN_layer'])):
            all_embedding = torch.sparse.mm(self.Graph, all_embedding)
            if perturbed:
                noise = torch.rand_like(all_embedding).to(self.device)
                all_embedding += torch.sign(all_embedding) * torch.nn.functional.normalize(noise, dim=-1) * self.epsilon

            embeddings.append(all_embedding)
            if layer == self.cl_layer - 1:
                all_embedding_cl = all_embedding

        final_embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])
        users_emb_cl, items_emb_cl = torch.split(all_embedding_cl, [self.dataset.num_users, self.dataset.num_items])
        if perturbed:
            return users_emb, items_emb, users_emb_cl, items_emb_cl
        return users_emb, items_emb

    def forward(self, user, positive, negative):
        all_user_embeddings, all_item_embeddings, users_emb_cl, items_emb_cl = self.aggregate(perturbed=True)

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        user_index = torch.unique(user)
        item_index = torch.unique(positive)

        user_ssl_loss = losses.get_InfoNCE_loss(users_emb_cl[user_index], all_user_embeddings[user_index], self.temperature)
        item_ssl_loss = losses.get_InfoNCE_loss(items_emb_cl[item_index], all_item_embeddings[item_index], self.temperature)

        ssl_loss = self.ssl_lambda * (user_ssl_loss + item_ssl_loss)

        loss_list = [bpr_loss, reg_loss, ssl_loss]

        return loss_list

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings = self.aggregate(perturbed=False)

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = XSimGCL(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)