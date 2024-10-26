"""
PyTorch Implementation of EGCF
Simplify to the Limit! Embedding-less Graph Collaborative Filtering for Recommender Systems Zhang et al. TOIS'24
For more information, please refer to: https://github.com/BlueGhostYi/EGCF
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph


class EGCF(nn.Module):
    def __init__(self, config, dataset, device):
        super(EGCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.ssl_lambda = float(self.config['ssl_lambda'])
        self.temperature = float(self.config['temperature'])
        self.aggregate_mode = self.config['mode']

        self.user_embedding = None

        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        # no pretrain
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.user_Graph = utility.utility_data.data_graph.sparse_adjacency_matrix_R(self.dataset)
        self.user_Graph = tools.convert_sp_mat_to_sp_tensor(self.user_Graph)
        self.user_Graph = self.user_Graph.coalesce().to(self.device)

        if self.aggregate_mode == 'parallel':
            self.Graph = utility.utility_data.data_graph.sparse_adjacency_matrix(self.dataset)
            self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)
            self.Graph = self.Graph.coalesce().to(self.device)

        self.activation_layer = nn.Tanh()
        self.activation = nn.Sigmoid()

    def alternating_aggregate(self):
        item_embedding = self.item_embedding.weight

        all_user_embeddings = []
        all_item_embeddings = []

        for layer in range(int(self.config['GCN_layer'])):
            user_embedding = self.activation_layer(torch.sparse.mm(self.user_Graph, item_embedding))
            item_embedding = self.activation_layer(torch.sparse.mm(self.user_Graph.transpose(0, 1), user_embedding))

            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)

        final_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        final_user_embeddings = torch.sum(final_user_embeddings, dim=1)

        final_item_embeddings = torch.stack(all_item_embeddings, dim=1)
        final_item_embeddings = torch.sum(final_item_embeddings, dim=1)

        return final_user_embeddings, final_item_embeddings

    def parallel_aggregate(self):
        item_embedding = self.item_embedding.weight
        user_embedding = self.activation_layer(torch.sparse.mm(self.user_Graph, item_embedding))

        all_embedding = torch.cat([user_embedding, item_embedding])

        all_embeddings = []

        for layer in range(int(self.config['GCN_layer'])):
            all_embedding = self.activation_layer(torch.sparse.mm(self.Graph, all_embedding))
            all_embeddings.append(all_embedding)

        final_all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.sum(final_all_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def forward(self, user, positive, negative):
        if self.aggregate_mode == 'parallel':
            all_user_embeddings, all_item_embeddings = self.parallel_aggregate()
        else:
            all_user_embeddings, all_item_embeddings = self.alternating_aggregate()

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = losses.get_reg_loss(ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        ssl_user_loss = losses.get_InfoNCE_loss(user_embedding, user_embedding, self.temperature)
        ssl_pos_loss = losses.get_InfoNCE_loss(pos_embedding, pos_embedding,  self.temperature)
        ssl_inter_loss = losses.get_InfoNCE_loss(user_embedding, pos_embedding, self.temperature)

        ssl_loss = self.ssl_lambda * (ssl_user_loss + ssl_pos_loss + ssl_inter_loss)

        loss_list = [bpr_loss, reg_loss, ssl_loss]

        return loss_list

    def get_rating_for_test(self, user):
        if self.aggregate_mode == 'parallel':
            all_user_embeddings, all_item_embeddings = self.parallel_aggregate()
        else:
            all_user_embeddings, all_item_embeddings = self.alternating_aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = EGCF(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)
