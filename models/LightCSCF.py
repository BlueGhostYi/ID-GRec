"""
PyTorch Implementation of LightCSCF
Revisiting Contrastive Learning in Collaborative Filtering via Parallel Graph Filters Kai et al. AAAI'26
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph


class LightCSCF(nn.Module):
    def __init__(self, config, dataset, device):
        super(LightCSCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.temperature = float(self.config['temperature'])
        self.lambda_gamma = float(self.config['lambda_gamma'])
        self.lambda_reg = float(self.config['lambda_reg'])
        self.lambda_margin = float(self.config['lambda_margin'])

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

    def aggregate(self):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # no dropout
        embeddings = [all_embedding]

        for layer in range(int(self.config['GCN_layer'])):
            all_embedding = torch.sparse.mm(self.Graph, all_embedding)
            embeddings.append(all_embedding)

        final_embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def forward(self, user, positive, negative):
        if self.config['encoder'] == 'MF':
            all_user_embeddings, all_item_embeddings = self.user_embedding.weight, self.item_embedding.weight
            user_embedding = all_user_embeddings[user.long()]
            pos_embedding = all_item_embeddings[positive.long()]
            neg_embedding = all_item_embeddings[negative.long()]

            ego_user_emb = self.user_embedding(user)
            ego_pos_emb = self.item_embedding(positive)
            ego_neg_emb = self.item_embedding(negative)

            bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)
            reg_loss = self.lambda_reg * losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
            na_loss = self.lambda_gamma * self.LightCSCF_loss(user_embedding, pos_embedding,
                                                                         self.temperature)

            loss_list = [bpr_loss, reg_loss, na_loss]
        else:
            all_user_embeddings, all_item_embeddings = self.aggregate()

            user_embedding = all_user_embeddings[user.long()]
            pos_embedding = all_item_embeddings[positive.long()]
            neg_embedding = all_item_embeddings[negative.long()]

            ego_user_emb = self.user_embedding(user)
            ego_pos_emb = self.item_embedding(positive)
            ego_neg_emb = self.item_embedding(negative)

            reg_loss = self.lambda_reg * losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
            na_loss = self.lambda_gamma * self.LightCSCF_loss(user_embedding, pos_embedding, self.temperature)

            loss_list = [reg_loss, na_loss]

        return loss_list

    def LightCSCF_loss(self, embedding1, embedding2, temperature):
        embedding1 = torch.nn.functional.normalize(embedding1, dim=1)
        embedding2 = torch.nn.functional.normalize(embedding2, dim=1)
        sim = (embedding1 * embedding2).sum(dim=-1)
        pos_score = torch.exp(sim / temperature) + torch.exp(torch.relu(sim - self.lambda_margin) / temperature)
        total_score = torch.matmul(embedding1, embedding2.transpose(0, 1)) + torch.matmul(embedding1,
                                                                                          embedding1.transpose(0, 1))
        total_score = torch.exp(total_score / temperature) + torch.exp(
            torch.relu(total_score - self.lambda_margin) / temperature)
        total_score = total_score.sum(dim=1)
        loss = -torch.log(pos_score / total_score + 10e-6)
        return torch.mean(loss)

    def get_rating_for_test(self, user):
        if self.config['encoder'] == 'MF':
            all_user_embeddings, all_item_embeddings = self.user_embedding.weight, self.item_embedding.weight
        else:
            all_user_embeddings, all_item_embeddings = self.aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = LightCSCF(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)