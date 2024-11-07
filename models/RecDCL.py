"""
PyTorch Implementation of RecDCL
RecDCL: Dual Contrastive Learning for Recommendation Zhang et al. WWW'24
The version was contributed by Yu Zhang and integrated by Yi Zhang.
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph
import torch.nn.functional as F


class RecDCL(nn.Module):
    def __init__(self, config, dataset, device):
        super(RecDCL, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.embedding_size = int(self.config['embedding_size'])

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.config['embedding_size']))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.activation = nn.Sigmoid()

        self.a = float(self.config['a'])                        # a in Eq. 6, set 1.0 by default
        self.polyc = float(self.config['polyc'])                # c in Eq. 6, set 1e-7 by default
        self.degree = float(self.config['degree'])              # e in Eq. 6, set 4.0 by default

        self.poly_coeff = float(self.config['poly_coeff'])      # alpha in Eq. 10, range of [0.2, 0.5, 1, 2, 5, 10]
        self.bt_coeff = float(self.config['bt_coeff'])          # gamma in Eq. 5, range of [0.005, 0.01, 0.05, 0.1]
        self.all_bt_coeff = float(self.config['all_bt_coeff'])  # set 1.0 by default
        self.mom_coeff = float(self.config['mom_coeff'])        # beta in Eq. 10, range of [1, 5, 10, 20]
        self.momentum = float(self.config['momentum'])          # tau in Eq. 8, range of [0.1, 0.3, 0.5, 0.7, 0.9]

        self.adj_mat = utility.utility_data.data_graph.sparse_adjacency_matrix(self.dataset)
        self.adj_mat = tools.convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

        self.bn = nn.BatchNorm1d(int(self.config['embedding_size']), affine=False)

        layers = []
        embs = str(self.embedding_size) + '-' + str(self.embedding_size) + '-' + str(self.embedding_size)
        sizes = [self.embedding_size] + list(map(int, embs.split('-')))
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

        self.projector = nn.Sequential(*layers)
        self.predictor = nn.Linear(self.embedding_size, self.embedding_size)

        self.u_target_his = torch.randn((self.dataset.num_users, self.embedding_size), requires_grad=False).to(
            self.device)
        self.i_target_his = torch.randn((self.dataset.num_items, self.embedding_size), requires_grad=False).to(
            self.device)

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def bt(self, x, y):
        user_e = self.projector(x)
        item_e = self.projector(y)
        c = self.bn(user_e).T @ self.bn(item_e)
        c.div_(user_e.size()[0])
        # sum the cross-correlation matrix
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().div(self.embedding_size)
        off_diag = self.off_diagonal(c).pow_(2).sum().div(self.embedding_size)
        bt = on_diag + self.bt_coeff * off_diag
        return bt

    def loss_fn(self, p, z):  # cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def poly_feature(self, x):
        user_e = self.projector(x)
        xx = self.bn(user_e).T @ self.bn(user_e)
        poly = (self.a * xx + self.polyc) ** self.degree
        return poly.mean().log()

    def aggregate(self):
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [embeddings]

        for layer in range(int(self.config['GCN_layer'])):
            embeddings = torch.sparse.mm(self.adj_mat, embeddings)
            all_embeddings.append(embeddings)
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        user_emb, item_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return user_emb, item_emb

    def forward(self, user, positive, negative):
        all_user_gcn_embed, all_item_gcn_embed = self.aggregate()

        user_gcn_embed = all_user_gcn_embed[user.long()]
        positive_gcn_embed = all_item_gcn_embed[positive.long()]

        with torch.no_grad():
            u_target, i_target = self.u_target_his.clone()[user.long(), :], self.i_target_his.clone()[positive.long(), :]
            u_target.detach()
            i_target.detach()

            u_target = u_target * self.momentum + user_gcn_embed.data * (1. - self.momentum)
            i_target = i_target * self.momentum + positive_gcn_embed.data * (1. - self.momentum)

            self.u_target_his[user.long(), :] = user_gcn_embed.clone()
            self.i_target_his[positive.long(), :] = positive_gcn_embed.clone()

        user_gcn_embed_n, positive_gcn_embed_n = F.normalize(user_gcn_embed, dim=-1), F.normalize(positive_gcn_embed, dim=-1)
        user_gcn_embed, positive_gcn_embed = self.predictor(user_gcn_embed), self.predictor(positive_gcn_embed)

        if self.all_bt_coeff == 0:
            bt_loss = 0.0
        else:
            bt_loss = self.bt(user_gcn_embed_n, positive_gcn_embed_n)

        if self.poly_coeff == 0:
            poly_loss = 0.0
        else:
            poly_loss = self.poly_feature(user_gcn_embed_n) / 2 + self.poly_feature(positive_gcn_embed_n) / 2

        if self.mom_coeff == 0:
            mom_loss = 0.0
        else:
            mom_loss = self.loss_fn(user_gcn_embed, i_target) / 2 + self.loss_fn(positive_gcn_embed, u_target) / 2

        loss_list = [self.all_bt_coeff * bt_loss, poly_loss * self.poly_coeff, mom_loss * self.mom_coeff]

        return loss_list

    def get_rating_for_test(self, user):
        all_user_gcn_embed, all_item_gcn_embed = self.aggregate()

        user_gcn_embed = all_user_gcn_embed[user.long()]

        rating = self.activation(torch.matmul(user_gcn_embed, all_item_gcn_embed.t()))

        return rating

class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = RecDCL(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)