"""
PyTorch Implementation of SCCF
Unifying Graph Convolution and Contrastive Learning in Collaborative Filtering  Wu te al. KDD'24
The version was contributed by Yu Zhang and integrated by Yi Zhang.
"""
import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_train.trainer as trainer
import utility.utility_data.data_graph
import torch.nn.functional as F


class SCCF(nn.Module):
    def __init__(self, config, dataset, device):
        super(SCCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])

        self.temperature = float(self.config['temperature'])               # t = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]
        self.encoder = self.config['encoder']

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users,
                                                 embedding_dim=int(self.config['embedding_size']))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items,
                                                 embedding_dim=int(self.config['embedding_size']))

        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.Graph = utility.utility_data.data_graph.sparse_adjacency_matrix(self.dataset)
        self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)
        self.Graph = self.Graph.coalesce().to(self.device)

        self.activation = nn.Sigmoid()

    def aggregate(self):
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [embeddings]

        for layer in range(int(self.config['GCN_layer'])):
            embeddings = torch.sparse.mm(self.Graph, embeddings)
            all_embeddings.append(embeddings)
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        user_emb, item_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return user_emb, item_emb

    def forward(self, user, positive, negative):
        if self.encoder == "MF":
            all_user_gcn_embed, all_item_gcn_embed = self.user_embedding.weight, self.item_embedding.weight
        else:
            all_user_gcn_embed, all_item_gcn_embed = self.aggregate()

        u_idx, u_inv_idx, u_counts = torch.unique(user, return_counts=True, return_inverse=True)
        i_idx, i_inv_idx, i_counts = torch.unique(positive, return_counts=True, return_inverse=True)
        u_counts, i_counts = u_counts.reshape(-1, 1).float(), i_counts.reshape(-1, 1).float()

        user_gcn_embed = all_user_gcn_embed[user.long()]
        positive_gcn_embed = all_item_gcn_embed[positive.long()]
        user_gcn_embed, positive_gcn_embed = F.normalize(user_gcn_embed, dim=-1),  F.normalize(positive_gcn_embed, dim=-1)
        ip = (user_gcn_embed * positive_gcn_embed).sum(dim=1)
        up_score = (ip / self.temperature).exp() + (ip ** 2 / self.temperature).exp()

        up = up_score.log().mean()

        user_gcn_embed = all_user_gcn_embed[u_idx.long()]
        positive_gcn_embed = all_item_gcn_embed[i_idx.long()]
        user_gcn_embed, positive_gcn_embed = F.normalize(user_gcn_embed, dim=-1),  F.normalize(positive_gcn_embed, dim=-1)
        sim_mat = user_gcn_embed @ positive_gcn_embed.T
        score = (sim_mat / self.temperature).exp() + (sim_mat ** 2 / self.temperature).exp()

        down = (score * (u_counts @ i_counts.T)).mean().log()

        loss_list = [-up, down]
        return loss_list

    def get_rating_for_test(self, user):
        if self.encoder == "MF":
            rating = self.activation(torch.matmul(self.user_embedding(user), self.item_embedding.weight.t()))
        else:
            all_user_gcn_embed, all_item_gcn_embed = self.aggregate()
            user_gcn_embed = all_user_gcn_embed[user.long()]
            rating = self.activation(torch.matmul(user_gcn_embed, all_item_gcn_embed.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = SCCF(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # This function provides a universal training and testing process
    # that can be customized according to actual situations.
    def train(self):
        trainer.universal_trainer(self.model, self.args, self.config, self.dataset, self.device, self.logger)