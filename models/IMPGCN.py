"""
PyTorch Implementation of IMP-GCN
Interest-aware Message-Passing GCN for Recommendation  Liu et al. WWW'21
The implementation mainly refers to https://github.com/PTMZ/IMP_GCN
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph


class IMPGCN(nn.Module):
    def __init__(self, config, dataset, device):
        super(IMPGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.num_groups = int(self.config['group'])
        self.num_layers = int(self.config['GCN_layer'])

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.config['embedding_size']))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.fc = torch.nn.Linear(int(self.config['embedding_size']), int(self.config['embedding_size']))
        self.fc_group = torch.nn.Linear(int(self.config['embedding_size']), self.num_groups)
        self.leaky = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=0.4)

        self.Graph = utility.utility_data.data_graph.sparse_adjacency_matrix(self.dataset)
        self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)  # sparse tensor
        self.Graph = self.Graph.coalesce().to(self.device)  # Sort the edge index and remove redundancy

        self.activation = nn.Sigmoid()

    def get_subgraph(self, graph, group, dim):
        index = graph._indices()
        value = graph._values()
        dv = group[index[dim, :]]
        return torch.sparse.FloatTensor(index, value * dv, graph.size())

    def aggregate(self):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        ego_embedding = all_embedding
        side_embedding = torch.sparse.mm(self.Graph, all_embedding)
        temp_embedding = self.dropout(self.leaky(self.fc(ego_embedding + side_embedding)))  # Eq. 13 in paper
        group_scores = self.dropout(self.fc_group(temp_embedding))

        a_top, a_top_idx = torch.topk(group_scores, k=1, sorted=False)
        one_hot_embedding = torch.eq(group_scores, a_top).float()

        user_group_embedding, item_group_embedding = torch.split(one_hot_embedding, [self.dataset.num_users, self.dataset.num_items])
        item_group_embedding = torch.ones(item_group_embedding.shape).to(self.device)
        group_embedding = torch.cat([user_group_embedding, item_group_embedding]).t()  # [num_group, num_nodes]

        subgraph_list = []
        for g in range(self.num_groups):
            temp_graph = self.get_subgraph(self.Graph, group_embedding[g], 1)
            temp_graph = self.get_subgraph(temp_graph, group_embedding[g], 0)
            subgraph_list.append(temp_graph)

        all_group_embeddings = [[None for _ in range(self.num_groups)] for _ in range(self.num_layers)]

        for group in range(self.num_groups):
            all_group_embeddings[0][group] = all_embedding

        for layer in range(1, self.num_layers):
            for group in range(self.num_groups):
                all_group_embeddings[layer][group] = torch.sparse.mm(subgraph_list[group], all_group_embeddings[layer-1][group])

        all_embedding_list = [torch.sum(torch.stack(emb), 0) for emb in all_group_embeddings]

        final_embeddings = torch.stack(all_embedding_list, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

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
        self.model = IMPGCN(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device,self.logger)
