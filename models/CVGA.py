"""
PyTorch Implementation of CVGA
Revisiting Graph-based Recommender Systems from the Perspective of Variational Auto-Encoder  Zhang et al. TOIS'23
"""

import torch
from torch import nn
import utility.utility_function.losses as losses
import utility.utility_function.tools as tools
import utility.utility_data.data_graph
import utility.utility_train.batch_test as batch_test
from time import time
from tqdm import tqdm
import numpy as np


class CVGA(nn.Module):
    def __init__(self, config, dataset, device):
        super(CVGA, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.p_dims = [int(config['embedding_size']), self.dataset.num_items]
        self.q_dims = [self.dataset.num_items, int(config['embedding_size'])]
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]

        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])

        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.drop = nn.Dropout(float(config['dropout']))

        self.Graph = utility.utility_data.data_graph.sparse_adjacency_matrix_R(self.dataset)
        self.Graph = tools.convert_sp_mat_to_sp_tensor(self.Graph)
        self.Graph = self.Graph.coalesce().to(self.device)

        self.activation = nn.Sigmoid()

    def encode(self):
        for i, layer in enumerate(self.q_layers):
            if i == 0:
                h = layer(self.Graph)
            else:
                h = layer(torch.sparse.mm(self.Graph.t(), h))
            h = self.drop(h)

            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
                return mu, logvar

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std) + mu
        return z

    def forward(self, user, x):
        mu, logvar = self.encode()

        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z[user.long()])

        recon_loss, KL_loss = losses.get_ELBO_loss(recon_x, x, mu[user.long()], logvar[user.long()], 1.0)
        loss_list = [recon_loss, KL_loss]

        return loss_list

    def get_rating_for_test(self, user):
        mu, logvar = self.encode()
        z = self.reparameterize(mu, logvar)
        rating = self.decode(z[user.long()])

        return rating


class Trainer():
    def __init__(self, args, config, dataset, device, logger):
        self.model = CVGA(config, dataset, device)
        self.args = args
        self.device = device
        self.config = config
        self.dataset = dataset
        self.logger = logger

    # Customized training and testing process for CVGA
    def train(self):
        self.CVGA_trainer()

    def CVGA_trainer(self):
        self.model.to(self.device)

        Optim = torch.optim.Adam(self.model.parameters(), lr=float(self.config['learn_rate']))

        user_list = list(range(self.dataset.num_users))
        np.random.shuffle(user_list)
        train_data = self.dataset.user_item_net

        best_results = dict()
        best_results['count'] = 0
        best_results['epoch'] = 0
        best_results['recall'] = [0. for _ in eval(self.config['top_K'])]
        best_results['ndcg'] = [0. for _ in eval(self.config['top_K'])]
        best_results['stop'] = 0

        for epoch in range(int(self.config['training_epochs'])):
            print('-' * 100)
            start_time = time()

            self.model.train()

            num_batch = self.dataset.num_users // int(self.config['batch_size']) + 1

            total_loss_list = []

            for batch_id, start_id in tqdm(enumerate(range(0, self.dataset.num_users, int(self.config['batch_size']))),
                                           desc='Training epoch ' + str(epoch + 1), total=int(num_batch)):
                end_id = min(start_id + int(self.config['batch_size']),self.dataset.num_users)
                batch_data = train_data[user_list[start_id:end_id]]

                users = torch.Tensor(user_list[start_id:end_id]).long()
                users = users.to(self.device)
                data = torch.FloatTensor(batch_data.toarray()).to(self.device)

                loss_list = self.model(users, data)

                if batch_id == 0:
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
                self.logger.info("Epoch: %4d | Test recall: %s | Test NDCG: %s" % (epoch + 1, result['recall'], result['ndcg']))
                if best_results['stop'] > 0:
                    break

        print("\t Model training process completed.")

        self.logger.info('Model training process completed.')
        self.logger.info("Best epoch: %4d | Best recall: %s | Best NDCG: %s" % (best_results['epoch'], best_results['recall'], best_results['ndcg']))