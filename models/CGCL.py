"""
PyTorch Implementation of CGCL
Candidate–aware Graph Contrastive Learning for Recommendation  He et al. SIGIR'23
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph


class CGCL(nn.Module):
    def __init__(self, config, dataset, device):
        super(CGCL, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.ssl_lambda_alpha = float(self.config['ssl_lambda_alpha'])
        self.ssl_lambda_beta = float(self.config['ssl_lambda_beta'])
        self.ssl_lambda_gamma = float(self.config['ssl_lambda_gamma'])
        self.alpha = float(self.config['alpha'])
        self.beta = float(self.config['beta'])
        self.gamma = float(self.config['gamma'])
        self.temperature = float(self.config['temperature'])

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

        return users_emb, items_emb, embeddings

    def forward(self, user, positive, negative):
        all_user_embeddings, all_item_embeddings, embeddings_list = self.aggregate()

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        # ssl_loss
        center_embedding = embeddings_list[0]
        candidate_embedding = embeddings_list[1]
        context_embedding = embeddings_list[2]

        layer_ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, user.long(), positive.long())

        candidate_ssl_loss = self.ssl_candidate_loss(candidate_embedding, center_embedding, user.long(), positive.long())

        struct_ssl_loss = self.ssl_struct_loss(context_embedding, candidate_embedding, user.long(), positive.long())

        loss_list = [bpr_loss, reg_loss, layer_ssl_loss, candidate_ssl_loss, struct_ssl_loss]

        return loss_list

    def ssl_layer_loss(self, current_emb, pre_emb, user, item):
        layer_user_emb, layer_item_emb = torch.split(current_emb, [self.dataset.num_users, self.dataset.num_items])
        pre_user_emb_all, pre_item_emb_all = torch.split(pre_emb, [self.dataset.num_users, self.dataset.num_items])

        # user side
        pre_user_emb = pre_user_emb_all[user]
        cur_user_emb = layer_item_emb[item]

        cur_user_emb = torch.nn.functional.normalize(cur_user_emb)
        pre_user_emb = torch.nn.functional.normalize(pre_user_emb)
        pre_user_emb_all = torch.nn.functional.normalize(pre_user_emb_all)

        pos_user_score = torch.mul(cur_user_emb, pre_user_emb).sum(dim=1)
        ttl_user_score = torch.matmul(cur_user_emb, pre_user_emb_all.transpose(0, 1))

        pos_user_score = torch.exp(pos_user_score / self.temperature)
        ttl_user_score = torch.exp(ttl_user_score / self.temperature).sum(dim=1)

        # item side
        pre_item_emb = pre_item_emb_all[item]
        cur_item_emb = layer_user_emb[user]

        cur_item_emb = torch.nn.functional.normalize(cur_item_emb)
        pre_item_emb = torch.nn.functional.normalize(pre_item_emb)
        pre_item_emb_all = torch.nn.functional.normalize(pre_item_emb_all)

        pos_item_score = torch.mul(cur_item_emb, pre_item_emb).sum(dim=1)
        ttl_item_score = torch.matmul(cur_item_emb, pre_item_emb_all.transpose(0, 1))

        pos_item_score = torch.exp(pos_item_score / self.temperature)
        ttl_item_score = torch.exp(ttl_item_score / self.temperature).sum(dim=1)

        user_loss = - torch.log(pos_user_score / ttl_user_score + 10e-8).sum()
        item_loss = - torch.log(pos_item_score / ttl_item_score + 10e-8).sum()
        ssl_loss = self.ssl_lambda_alpha * (self.alpha * user_loss + (1 - self.alpha) * item_loss)
        return ssl_loss

    def ssl_candidate_loss(self, current_emb, pre_emb, user, item):
        layer_user_emb, layer_item_emb = torch.split(current_emb, [self.dataset.num_users, self.dataset.num_items])
        pre_user_emb_all, pre_item_emb_all = torch.split(pre_emb, [self.dataset.num_users, self.dataset.num_items])

        # user side
        pre_user_emb = pre_user_emb_all[user]
        cur_user_emb = layer_item_emb[item]

        cur_user_emb = torch.nn.functional.normalize(cur_user_emb)
        pre_user_emb = torch.nn.functional.normalize(pre_user_emb)
        pre_user_emb_all = torch.nn.functional.normalize(pre_user_emb_all)

        pos_user_score = torch.mul(cur_user_emb, pre_user_emb).sum(dim=1)
        ttl_user_score = torch.matmul(cur_user_emb, pre_user_emb_all.transpose(0, 1))

        pos_user_score = torch.exp(pos_user_score / self.temperature)
        ttl_user_score = torch.exp(ttl_user_score / self.temperature).sum(dim=1)

        # item side
        pre_item_emb = pre_item_emb_all[item]
        cur_item_emb = layer_user_emb[user]

        cur_item_emb = torch.nn.functional.normalize(cur_item_emb)
        pre_item_emb = torch.nn.functional.normalize(pre_item_emb)
        pre_item_emb_all = torch.nn.functional.normalize(pre_item_emb_all)

        pos_item_score = torch.mul(cur_item_emb, pre_item_emb).sum(dim=1)
        ttl_item_score = torch.matmul(cur_item_emb, pre_item_emb_all.transpose(0, 1))

        pos_item_score = torch.exp(pos_item_score / self.temperature)
        ttl_item_score = torch.exp(ttl_item_score / self.temperature).sum(dim=1)

        user_loss = - torch.log(pos_user_score / ttl_user_score + 10e-8).sum()
        item_loss = - torch.log(pos_item_score / ttl_item_score + 10e-8).sum()

        ssl_loss = self.ssl_lambda_beta * (self.beta * user_loss + (1 - self.beta) * item_loss)
        return ssl_loss

    def ssl_struct_loss(self, neighbor_emb, center_emb, user, item):
        neighbor_user, neighbor_item = torch.split(neighbor_emb, [self.dataset.num_users, self.dataset.num_items])
        center_user, center_item = torch.split(center_emb, [self.dataset.num_users, self.dataset.num_items])

        anchor_user_emb = center_user[user]
        neigh_items_emb = neighbor_item[item]

        neigh_items_emb = torch.nn.functional.normalize(neigh_items_emb)
        anchor_user_emb = torch.nn.functional.normalize(anchor_user_emb)
        center_user = torch.nn.functional.normalize(center_user)

        pos_user_score = torch.mul(neigh_items_emb, anchor_user_emb).sum(dim=1)
        ttl_user_score = torch.matmul(neigh_items_emb, center_user.transpose(0, 1))

        pos_user_score = torch.exp(pos_user_score / self.temperature)
        ttl_user_score = torch.exp(ttl_user_score / self.temperature).sum(dim=1)

        neigh_users_emb = neighbor_user[user]
        anchor_item_emb = center_item[item]

        neigh_users_emb = torch.nn.functional.normalize(neigh_users_emb)
        anchor_item_emb = torch.nn.functional.normalize(anchor_item_emb)
        center_item = torch.nn.functional.normalize(center_item)

        pos_item_score = torch.mul(neigh_users_emb, anchor_item_emb).sum(dim=1)
        ttl_item_score = torch.matmul(neigh_users_emb, center_item.transpose(0, 1))

        pos_item_score = torch.exp(pos_item_score / self.temperature)
        ttl_item_score = torch.exp(ttl_item_score / self.temperature).sum(dim=1)

        user_loss = - torch.log(pos_user_score / ttl_user_score + 10e-8).sum()
        item_loss = - torch.log(pos_item_score / ttl_item_score + 10e-8).sum()

        ssl_loss = self.ssl_lambda_gamma * (self.gamma * user_loss + (1 - self.gamma) * item_loss)
        return ssl_loss

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings, _ = self.aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = CGCL(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)