"""
PyTorch Implementation of NCL
Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning  Lin et al. WWW'22
Additional requirements: faiss
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_data.data_graph
import utility.utility_train.batch_test as batch_test
from time import time
import faiss
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class NCL(nn.Module):
    def __init__(self, config, dataset, device):
        super(NCL, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config['reg_lambda'])
        self.ssl_lambda = float(self.config['ssl_lambda'])
        self.proto_lambda = float(self.config['proto_lambda'])
        self.k = int(self.config['k'])
        self.alpha = float(self.config['alpha'])
        self.temperature = float(self.config['temperature'])
        self.cl_layer = int(self.config['cl_layer'])

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

    def kmeans(self, embeddings):
        kmeans = faiss.Kmeans(d=int(self.config['embedding_size']), k=self.k, gpu=True)
        kmeans.train(embeddings)
        cluster_centroids = kmeans.centroids
        _, I = kmeans.index.search(embeddings, 1)
        centroids = torch.Tensor(cluster_centroids).to(self.device)  # [K, dim]
        node2cluster = torch.LongTensor(I).squeeze().to(self.device)  # [embeddings.shape[0]]

        return centroids, node2cluster

    def E_step(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()

        self.user_centroids, self.user_2cluster = self.kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.kmeans(item_embeddings)

    def forward(self, user, positive, negative, epoch):
        all_user_embeddings, all_item_embeddings, embedding_list = self.aggregate()

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        bpr_loss = losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        reg_loss = losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        init_embeddings = embedding_list[0]
        layer_embeddings = embedding_list[self.cl_layer * 2]

        init_user_embeddings, init_item_embeddings = torch.split(init_embeddings, [self.dataset.num_users, self.dataset.num_items])
        layer_user_embeddings, layer_item_embeddings = torch.split(layer_embeddings, [self.dataset.num_users, self.dataset.num_items])

        user_ssl_loss = self.ssl_layer_loss(layer_user_embeddings[user.long()], init_user_embeddings[user.long()], init_user_embeddings)
        item_ssl_loss = self.ssl_layer_loss(layer_item_embeddings[positive.long()], init_item_embeddings[positive.long()], init_item_embeddings)

        ssl_loss = self.ssl_lambda * (user_ssl_loss + self.alpha * item_ssl_loss)

        if epoch < 20:
            loss_list = [bpr_loss, reg_loss, ssl_loss]
            return loss_list

        user_2cluster = self.user_2cluster[user.long()]
        item_2cluster = self.item_2cluster[positive.long()]

        user_2centroids = self.user_centroids[user_2cluster]
        item_2centroids = self.item_centroids[item_2cluster]

        user_proto_loss = losses.get_InfoNCE_loss(init_user_embeddings[user.long()], user_2centroids, self.temperature)
        item_proto_loss = losses.get_InfoNCE_loss(init_item_embeddings[positive.long()], item_2centroids, self.temperature)

        proto_loss = self.proto_lambda * (user_proto_loss + item_proto_loss) * int(self.config['batch_size'])

        loss_list = [bpr_loss, reg_loss, ssl_loss, proto_loss]

        return loss_list

    def ssl_layer_loss(self, embedding_1, embedding_2, embedding_all):
        embedding_1 = torch.nn.functional.normalize(embedding_1)
        embedding_2 = torch.nn.functional.normalize(embedding_2)
        embedding_all = torch.nn.functional.normalize(embedding_all)

        pos_score = (embedding_1 * embedding_2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)

        ttl_score = torch.matmul(embedding_1, embedding_all.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)

        cl_loss = - torch.log(pos_score / ttl_score + 10e-8)

        return torch.sum(cl_loss)

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings, _ = self.aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = NCL(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # Customized training and testing process for NCL
    def train(self):
        self.model.to(self.device)

        Optim = torch.optim.Adam(self.model.parameters(), lr=float(self.config['learn_rate']))

        best_results = dict()
        best_results['count'] = 0
        best_results['epoch'] = 0
        best_results['recall'] = [0. for _ in eval(self.config['top_K'])]
        best_results['ndcg'] = [0. for _ in eval(self.config['top_K'])]

        for epoch in range(int(self.config['training_epochs'])):
            start_time = time()

            if epoch >= 20:
                self.model.E_step()

            self.model.train()

            sample_data = self.dataset.sample_data_to_train_all()
            users = torch.Tensor(sample_data[:, 0]).long()
            pos_items = torch.Tensor(sample_data[:, 1]).long()
            neg_items = torch.Tensor(sample_data[:, 2]).long()

            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            users, pos_items, neg_items = tools.shuffle(users, pos_items, neg_items)
            num_batch = len(users) // int(self.config['batch_size']) + 1

            total_loss_list = []

            for batch_i, (batch_users, batch_positive, batch_negative) in \
                    enumerate(
                        tools.mini_batch(users, pos_items, neg_items, batch_size=int(self.config['batch_size']))):
                loss_list = self.model(batch_users, batch_positive, batch_negative, epoch)

                if batch_i == 0:
                    assert len(loss_list) >= 1
                    total_loss_list = [0.] * len(loss_list)

                total_loss = 0.
                for i in range(len(loss_list)):
                    loss = loss_list[i]
                    total_loss += loss
                    total_loss_list[i] += loss.item()

                Optim.zero_grad()
                total_loss.backward()
                Optim.step()

            end_time = time()
            loss_strs = str(round(sum(total_loss_list) / num_batch, 6)) \
                        + " = " + " + ".join([str(round(i / num_batch, 6)) for i in total_loss_list])

            print("\t Epoch: %4d| train time: %.3f | train_loss: %s" % (epoch + 1, end_time - start_time, loss_strs))
            self.logger.info(
                "Epoch: %4d | Training time: %.3f | training loss: %s" % (epoch + 1, end_time - start_time, loss_strs))

            if epoch % int(self.config['interval']) == 0:
                result, best_results = batch_test.general_test(self.dataset, self.model, self.device, self.config, epoch, best_results)
                self.logger.info(
                    "Epoch: %4d | Test recall: %s | Test NDCG: %s" % (epoch + 1, result['recall'], result['ndcg']))

        print("\t Model training process completed.")

        self.logger.info('Model training process completed.')
        result, best_results = batch_test.general_test(self.dataset, self.model, self.device, self.config, int(self.config['training_epochs']),
                                                       best_results)
        self.logger.info("Best epoch: %4d | Best recall: %s | Best NDCG: %s" % (
        best_results['epoch'], best_results['recall'], best_results['ndcg']))
