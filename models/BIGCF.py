"""
PyTorch Implementation of BIGCF
Exploring the Individuality and Collectivity of Intents behind Interactions for Graph Collaborative Filtering  Zhang et al. SIGIR'24
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph


class BIGCF(nn.Module):
    def __init__(self, config, dataset, device):
        super(BIGCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.ssl_lambda = float(self.config['ssl_lambda'])
        self.ssl_temperature = float(self.config['ssl_temperature'])
        self.int_temperature = float(self.config['int_temperature'])
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.config['embedding_size']))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        self.user_intent = torch.nn.Embedding(num_embeddings=int(self.config['embedding_size']),
                                              embedding_dim=int(self.config['intent_size']))
        self.item_intent = torch.nn.Embedding(num_embeddings=int(self.config['embedding_size']),
                                              embedding_dim=int(self.config['intent_size']))

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1.)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1.)
        nn.init.xavier_uniform_(self.user_intent.weight, gain=1.)
        nn.init.xavier_uniform_(self.item_intent.weight, gain=1.)

        self.Graph = utility.utility_data.data_graph.sparse_adjacency_matrix(self.dataset)
        self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)
        self.Graph = self.Graph.coalesce().to(self.device)

        self.activation = nn.Sigmoid()

    def aggregate(self):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        embeddings = []

        for layer in range(int(self.config['GCN_layer'])):

            all_embedding = torch.sparse.mm(self.Graph, all_embedding)
            embeddings.append(all_embedding)

        gnn_embeddings = torch.stack(embeddings, dim=1)
        gnn_embeddings = torch.sum(gnn_embeddings, dim=1, keepdim=False)

        user_embedding, item_embedding = torch.split(gnn_embeddings, [self.dataset.num_users, self.dataset.num_items], 0)

        user_intent = torch.softmax(user_embedding @ self.user_intent.weight, dim=1) @ self.user_intent.weight.T  # [B, dim]
        item_intent = torch.softmax(item_embedding @ self.item_intent.weight, dim=1) @ self.item_intent.weight.T  # [B, dim]

        intent_embeddings = torch.concat([user_intent, item_intent], dim=0)

        noise = torch.randn_like(gnn_embeddings)

        final_embeddings = gnn_embeddings + intent_embeddings * noise

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])
        users_intent_emb, items_intent_emb = torch.split(intent_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb, users_intent_emb, items_intent_emb

    def forward(self, user, positive, negative):
        user_embeddings, item_embeddings, intent_user_embeddings, intent_item_embeddings = self.aggregate()

        user_embedding = user_embeddings[user.long()]
        pos_embedding = item_embeddings[positive.long()]
        neg_embedding = item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb, self.user_intent.weight, self.item_intent.weight)
        reg_loss = self.reg_lambda * reg_loss

        ssl_user_loss = losses.get_InfoNCE_loss(user_embedding, user_embedding, self.ssl_temperature)

        ssl_pos_loss = losses.get_InfoNCE_loss(pos_embedding, pos_embedding, self.ssl_temperature)

        ssl_inter_loss = losses.get_InfoNCE_loss(user_embedding, pos_embedding, self.ssl_temperature)

        ssl_user_loss_2 = losses.get_InfoNCE_loss(intent_user_embeddings[user.long()], intent_user_embeddings[user.long()], self.ssl_temperature)

        ssl_pos_loss_2 = losses.get_InfoNCE_loss(intent_item_embeddings[positive.long()], intent_item_embeddings[positive.long()], self.ssl_temperature)

        ssl_loss = self.ssl_lambda * (ssl_user_loss + ssl_pos_loss + ssl_inter_loss + ssl_user_loss_2 + ssl_pos_loss_2)

        loss_list = [bpr_loss, reg_loss, ssl_loss]

        return loss_list

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings, _, _ = self.aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = BIGCF(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)