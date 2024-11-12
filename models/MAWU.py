"""
PyTorch Implementation of MAWU
Toward a Better Understanding of Loss Functions for Collaborative Filtering  Park et al. CIKM'23
According to the description in the paper, MF or LightGCN can be selected as the encoder for training.
"""

import torch
import numpy as np
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph


class MAWU(nn.Module):
    def __init__(self, config, dataset, device):
        super(MAWU, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.gamma_user = float(self.config['gamma_user'])
        self.gamma_item = float(self.config['gamma_item'])
        self.reg_lambda = float(self.config['reg_lambda'])

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.config['embedding_size']))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        self.user_margin = torch.nn.Embedding(self.dataset.num_users, 1)
        self.item_margin = torch.nn.Embedding(self.dataset.num_items, 1)

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        if self.config['encoder'] == 'LightGCN':
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

    def margin_alignment(self, embedding_1, embedding_2, margin):
        embedding_1 = torch.nn.functional.normalize(embedding_1, dim=-1)
        embedding_2 = torch.nn.functional.normalize(embedding_2, dim=-1)

        cos_similarity = torch.sum(embedding_1 * embedding_2, dim=-1)
        angle_ui = torch.arccos(torch.clamp(cos_similarity, -1 + 1e-7, 1 - 1e-7))

        angle_ui_plus_margin = angle_ui + (1 - torch.sigmoid(margin))
        angle_ui_plus_margin = torch.clamp(angle_ui_plus_margin, 0., np.pi)

        cos_similarity_margin = torch.cos(angle_ui_plus_margin)
        return - cos_similarity_margin.mean()

    def dot_uniformity(self, embeddings, t=2):
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        cos_similarity = torch.nn.functional.cosine_similarity(embeddings[:, :, None], embeddings.t()[None, :, :])
        cos_similarity = torch.tril(cos_similarity, diagonal=-1)
        cos_similarity = 2 - 2 * cos_similarity
        return cos_similarity.mul(-t).exp().mean().log()

    # Negative samples are not used in DirectAU
    def forward(self, user, positive, negative):
        if self.config['encoder'] == 'LightGCN':
            all_user_embeddings, all_item_embeddings = self.aggregate()
            user_embedding = all_user_embeddings[user.long()]
            item_embedding = all_item_embeddings[positive.long()]
        elif self.config['encoder'] == 'MF':
            user_embedding = self.user_embedding(user)
            item_embedding = self.item_embedding(positive)

        user_margin = self.user_margin(user)
        item_margin = self.item_margin(positive)

        margin = user_margin + item_margin

        align_loss = self.margin_alignment(user_embedding, item_embedding, margin)
        uniform_loss = self.gamma_user * self.dot_uniformity(user_embedding) + \
                       self.gamma_item * self.dot_uniformity(item_embedding)

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)

        reg_loss = losses.get_reg_loss(ego_user_emb, ego_pos_emb)
        reg_loss = self.reg_lambda * reg_loss

        loss_list = [align_loss, uniform_loss, reg_loss]
        return loss_list

    def get_rating_for_test(self, user):
        if self.config['encoder'] == 'LightGCN':
            all_user_embeddings, all_item_embeddings = self.aggregate()
            user_embeddings = all_user_embeddings[user.long()]
        elif self.config['encoder'] == 'MF':
            user_embeddings = self.user_embedding(user)
            all_item_embeddings = self.item_embedding.weight

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = MAWU(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)
