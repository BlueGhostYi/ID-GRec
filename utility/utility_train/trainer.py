import torch
import utility.utility_train.batch_test as batch_test
import utility.utility_function.tools as tools
from time import time
from tqdm import tqdm


def universal_trainer(model, args, config, dataset, device, logger):
    model.to(device)

    Optim = torch.optim.Adam(model.parameters(), lr=float(config['learn_rate']))

    best_results = dict()
    best_results['count'] = 0
    best_results['epoch'] = 0
    best_results['recall'] = [0. for _ in eval(config['top_K'])]
    best_results['ndcg'] = [0. for _ in eval(config['top_K'])]
    best_results['stop'] = 0

    for epoch in range(int(config['training_epochs'])):
        print('-' * 100)
        start_time = time()

        model.train()

        sample_data = dataset.sample_data_to_train_all()
        users = torch.Tensor(sample_data[:, 0]).long()
        pos_items = torch.Tensor(sample_data[:, 1]).long()
        neg_items = torch.Tensor(sample_data[:, 2]).long()

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        users, pos_items, neg_items = tools.shuffle(users, pos_items, neg_items)
        num_batch = len(users) // int(config['batch_size']) + 1

        total_loss_list = []

        for batch_i, (batch_users, batch_positive, batch_negative) in \
                tqdm(enumerate(tools.mini_batch(users, pos_items, neg_items, batch_size=int(config['batch_size']))), desc='Training epoch ' + str(epoch + 1), total=int(num_batch)):
            loss_list = model(batch_users, batch_positive, batch_negative)

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

        print("Training time: %.3f | training loss: %s" % (end_time - start_time, loss_strs))
        logger.info("Epoch: %4d | Training time: %.3f | training loss: %s" % (epoch + 1, end_time - start_time, loss_strs))

        if epoch % int(config['interval']) == 0:
            result, best_results = batch_test.general_test(dataset, model, device, config, epoch, best_results)
            logger.info("Epoch: %4d | Test recall: %s | Test NDCG: %s" % (epoch + 1, result['recall'], result['ndcg']))
            if best_results['stop'] > 0:
                break

    print("Model training process completed.")
    logger.info('Model training process completed.')
    logger.info("Best epoch: %4d | Best recall: %s | Best NDCG: %s" % (best_results['epoch'], best_results['recall'], best_results['ndcg']))
