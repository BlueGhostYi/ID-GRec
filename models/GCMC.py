"""
Created on June 4, 2023
PyTorch Implementation of GCMC
Neural Graph Collaborative Filtering  Wang et al. SIGIR'19
The implementation mainly refers to https://github.com/xiangwang1223/neural_graph_collaborative_filtering.
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph


class GCMC(nn.Module):
    def __init__(self, config, dataset, device):
        super(GCMC, self).__init__()
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

        initializer = nn.init.xavier_uniform_
        self.weight_dict = nn.ParameterDict()
        layers = [int(self.config['embedding_size'])] + eval(self.config['layer_size'])

        for layer in range(int(self.config['GCN_layer'])):
            self.weight_dict.update(
                {'W_gcn_%d' % layer: nn.Parameter(initializer(torch.empty(layers[layer], layers[layer + 1])))})
            self.weight_dict.update(
                {'b_gcn_%d' % layer: nn.Parameter(initializer(torch.empty(1, layers[layer + 1])))})
            self.weight_dict.update(
                {'W_mlp_%d' % layer: nn.Parameter(initializer(torch.empty(layers[layer], layers[layer + 1])))})
            self.weight_dict.update(
                {'b_mlp_%d' % layer: nn.Parameter(initializer(torch.empty(1, layers[layer + 1])))})

        if eval(config['mess_dropout']):
            self.mess_dropout = eval(config['mess_drop_prob'])

        self.Graph = utility.utility_data.data_graph.sparse_adjacency_matrix(self.dataset)
        self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)
        self.Graph = self.Graph.coalesce().to(self.device)

        self.activation = nn.Sigmoid()
        self.activation_layer = nn.Tanh()

    def node_dropout(self, graph, keep_prob):
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        random_index = torch.rand(len(values)) + (1 - keep_prob)
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/(1 - keep_prob)
        new_graph = torch.sparse.FloatTensor(index.t(), values, size)
        return new_graph

    def aggregate(self):
        # [user + item, emb_dim]
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        all_embeddings = [ego_embeddings]

        # aggregate
        for layer in range(int(self.config['GCN_layer'])):
            # # [node, emb_dim]
            side_embeddings = torch.sparse.mm(self.Graph, ego_embeddings)

            gcn_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gcn_%d' % layer]) \
                             + self.weight_dict['b_gcn_%d' % layer]

            gcn_embeddings = nn.LeakyReLU(negative_slope=0.2)(gcn_embeddings)

            mlp_embeddings = torch.matmul(gcn_embeddings, self.weight_dict['W_mlp_%d' % layer]) \
                             + self.weight_dict['b_mlp_%d' % layer]

            # message dropout
            ego_embeddings = nn.Dropout(self.mess_dropout[layer])(mlp_embeddings)

            norm_embeddings = nn.functional.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings.append(norm_embeddings)

        final_embeddings = torch.cat(all_embeddings, dim=1)  # [node, layer+1, emb_dim]

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])
        return users_emb, items_emb

    def forward(self, user, positive, negative):
        all_user_embeddings, all_item_embeddings = self.aggregate()

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]


        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        loss_list = [bpr_loss, reg_loss]

        return loss_list

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings = self.aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = GCMC(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)
