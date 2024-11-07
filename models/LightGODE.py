"""
PyTorch Implementation of LightGODE
Do We Really Need Graph Convolution During Training? Light Post-Training Graph-ODE for Efficient Recommendation Zhang et al. CIKM'24
Additional requirements: torchdiffeq
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph
from torchdiffeq import odeint


class LightGODE(nn.Module):
    def __init__(self, config, dataset, device):
        super(LightGODE, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.gamma = float(self.config['gamma'])
        self.reg_lambda = float(self.config['reg_lambda'])
        self.train_strategy = self.config['train_strategy']
        self.train_stage = 'pretrain'
        if self.train_strategy == 'MF':
            self.use_mf = True
        elif self.train_strategy == 'GODE':
            self.use_mf = False
        else:
            self.use_mf = None

        self.t = float(self.config['t'])
        self.restore_user = None
        self.restore_item = None

        self.Graph = utility.utility_data.data_graph.sparse_adjacency_matrix(self.dataset)
        self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)
        self.Graph = self.Graph.coalesce().to(self.device)

        self.encoder = ODEEncoder(self.use_mf, self.dataset.num_users, self.dataset.num_items,
                                  int(config['embedding_size']), self.Graph, t=torch.tensor([0, self.t]))

        self.activation = nn.Sigmoid()

    def get_batch_embeddings(self, user, item):
        if self.train_strategy == 'MF_init' and self.train_stage == 'pretrain':
            self.encoder.update(self.use_mf == self.training)

        user_embedding, item_embedding = self.encoder(user, item)

        return user_embedding, item_embedding

    def forward(self, user, positive, negative):
        if self.restore_user is not None or self.restore_item is not None:
            self.restore_user, self.restore_item = None, None

        user_embeddings, item_embeddings = self.get_batch_embeddings(user.long(), positive.long())

        align_loss = losses.get_align_loss(user_embeddings, item_embeddings)
        uniform_loss = self.gamma * (losses.get_uniform_loss(user_embeddings) +
                                     losses.get_uniform_loss(item_embeddings)) / 2

        ego_user_emb = self.encoder.user_embedding(user)
        ego_pos_emb = self.encoder.item_embedding(positive)

        reg_loss = losses.get_reg_loss(ego_user_emb, ego_pos_emb)
        reg_loss = self.reg_lambda * reg_loss

        loss_list = [align_loss, uniform_loss, reg_loss]
        return loss_list

    def get_rating_for_test(self, user):
        if self.restore_user is None or self.restore_item is None:
            if self.train_strategy == 'MF_init':
                self.encoder.update(use_mf=self.training)
            self.restore_user, self.restore_item = self.encoder.get_all_embeddings()

        user_embeddings = self.restore_user[user.long()]
        all_item_embeddings = self.restore_item

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class ODEEncoder(nn.Module):
    def __init__(self, use_mf, num_users, num_items, emb_size, adj, t=torch.tensor(([0,1])), solver='euler', use_w=False):
        super(ODEEncoder, self).__init__()
        self.use_mf = use_mf
        self.use_w = use_w
        self.num_users = num_users
        self.num_items = num_items
        self.adj = adj
        self.t = t
        self.odefuc = ODEFunc(self.num_users, self.num_items, self.adj, k_hops=1)
        self.solver = solver

        self.user_embedding = torch.nn.Embedding(num_users, emb_size)
        self.item_embedding = torch.nn.Embedding(num_items, emb_size)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

    def update(self, use_mf):
        self.use_mf = use_mf

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_all_embeddings(self):
        all_embeddings = self.get_ego_embeddings()
        if not self.use_mf:
            t = self.t.type_as(all_embeddings)
            self.odefuc.update_e(all_embeddings)
            z1 = odeint(self.odefuc, all_embeddings, t, method=self.solver)[1]
            all_embeddings = z1

        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items])
        return user_embeddings, item_embeddings

    def forward(self, user, item):
        user_all_embeddings, item_all_embeddings = self.get_all_embeddings()
        u_embedding = user_all_embeddings[user]
        i_embedding = item_all_embeddings[item]
        return u_embedding, i_embedding


class ODEFunc(nn.Module):
    def __init__(self, num_users, num_items, adj, k_hops=1):
        super(ODEFunc, self).__init__()
        self.adj = adj
        self.num_users = num_users
        self.num_items = num_items

    def update_e(self, emb):
        self.e = emb

    def forward(self, t, x):
        ax = torch.spmm(self.adj, x)
        f = ax + self.e
        return f


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = LightGODE(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)