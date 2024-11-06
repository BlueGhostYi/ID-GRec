"""
PyTorch Implementation of HCCF
Hypergraph Contrastive Collaborative Filtering  Xia et al. SIGIR'22
The implementation mainly refers to https://github.com/akaxlh/HCCF/blob/main/torchVersion/Model.py
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph
import torch.nn.functional as F


class HCCF(nn.Module):
    def __init__(self, config, dataset, device):
        super(HCCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.ssl_lambda = float(self.config['ssl_lambda'])
        self.keeprate = float(self.config['keeprate'])
        self.temperature = float(self.config['temperature'])
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.config['embedding_size']))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        self.user_hyper_emb = torch.nn.Embedding(num_embeddings=int(self.config['embedding_size']),
                                                 embedding_dim=int(self.config['hyper_size']))
        self.item_hyper_emb = torch.nn.Embedding(num_embeddings=int(self.config['embedding_size']),
                                                 embedding_dim=int(self.config['hyper_size']))

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1.)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1.)
        nn.init.xavier_uniform_(self.user_hyper_emb.weight, gain=1.)
        nn.init.xavier_uniform_(self.item_hyper_emb.weight, gain=1.)

        self.Graph = utility.utility_data.data_graph.sparse_adjacency_matrix(self.dataset)
        self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)
        self.Graph = self.Graph.coalesce().to(self.device)


        self.activation = nn.Sigmoid()

    def aggregate(self):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        embeddings = [all_embedding]
        gnn_embeddings = []
        hyper_embeddings = []
        uu_hyper = self.user_embedding.weight @ self.user_hyper_emb.weight
        ii_hyper = self.item_embedding.weight @ self.item_hyper_emb.weight

        for layer in range(int(self.config['GCN_layer'])):
            all_embedding = torch.sparse.mm(self.Graph, embeddings[-1])
            hyper_user_embeddings = self.HGNN(F.dropout(uu_hyper, p=1-self.keeprate), embeddings[-1][:self.dataset.num_users])
            hyper_item_embeddings = self.HGNN(F.dropout(ii_hyper, p=1-self.keeprate), embeddings[-1][self.dataset.num_users:])

            gnn_embeddings.append(all_embedding)
            hyper_embeddings.append(torch.concat([hyper_user_embeddings, hyper_item_embeddings], dim=0))
            embeddings.append(all_embedding + hyper_embeddings[-1])

        final_embeddings = sum(embeddings)

        return final_embeddings, gnn_embeddings, hyper_embeddings

    def HGNN(self, adj, embedding):
        latent = adj.T @ embedding
        return adj @ latent

    def dropedge(self, adj, keeprate):
        if keeprate == 1.0:
            return adj
        values = adj._values()
        indices = adj._indices()
        edge_num = values.size()
        mask = ((torch.rand(edge_num) + keeprate).floor()).type(torch.bool)
        new_values = values[mask] / keeprate
        new_indices = indices[:, mask]
        return torch.sparse.FloatTensor(new_indices, new_values, adj.shape)

    def forward(self, user, positive, negative):
        final_embeddings, gnn_embeddings, hyper_embeddings = self.aggregate()
        user_embeddings, item_embeddings = final_embeddings[:self.dataset.num_users], final_embeddings[self.dataset.num_users:]

        user_embedding = user_embeddings[user.long()]
        pos_embedding = item_embeddings[positive.long()]
        neg_embedding = item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb,
                                               self.user_hyper_emb.weight, self.item_hyper_emb.weight)
        reg_loss = self.reg_lambda * reg_loss

        ssl_loss = 0.

        for layer in range(int(self.config['GCN_layer'])):
            embedding_1 = gnn_embeddings[layer].detach()
            embedding_2 = hyper_embeddings[layer]
            ssl_loss += losses.get_InfoNCE_loss(embedding_1[:self.dataset.num_users][user.long()],
                                                        embedding_2[:self.dataset.num_users][user.long()], self.temperature)
            ssl_loss += losses.get_InfoNCE_loss(embedding_1[self.dataset.num_users:][positive.long()],
                                                        embedding_2[self.dataset.num_users:][positive.long()], self.temperature)

        ssl_loss = self.ssl_lambda * ssl_loss

        loss_list = [bpr_loss, reg_loss, ssl_loss]

        return loss_list

    def get_rating_for_test(self, user):
        final_embeddings, gnn_embeddings, hyper_embeddings = self.aggregate()
        all_user_embeddings, all_item_embeddings = final_embeddings[:self.dataset.num_users], final_embeddings[self.dataset.num_users:]
        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = HCCF(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)
