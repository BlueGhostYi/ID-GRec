"""
Created on May 5, 2021
PyTorch Implementation of MF-BPR
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_train.trainer as trainer


class MFBPR(nn.Module):
    def __init__(self, config, dataset, device):
        super(MFBPR, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.config['embedding_size']))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.activation = nn.Sigmoid()

    def forward(self, user, positive, negative):
        user_embedding = self.user_embedding(user)

        pos_embedding = self.item_embedding(positive)
        neg_embedding = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = losses.get_reg_loss(user_embedding, pos_embedding, neg_embedding)
        reg_loss = self.reg_lambda * reg_loss

        loss_list = [bpr_loss, reg_loss]

        return loss_list

    def get_rating_for_test(self, user):
        user_embeddings = self.user_embedding(user)
        item_embeddings = self.item_embedding.weight

        rating = self.activation(torch.matmul(user_embeddings, item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = MFBPR(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)
