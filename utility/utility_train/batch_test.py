import torch
import numpy as np
from utility.utility_data.data_loader import Data
from utility.utility_function.tools import mini_batch
import utility.utility_function.metrics as metrics


def general_test(dataset, model, device, config, epoch, best_results):
    if int(config["sparsity_test"]) == 0:
        result = Test(dataset, model, device, config)
        if result['recall'][0] > best_results['recall'][0]:
            best_results['count'] = 0
            best_results['epoch'] = epoch + 1
            best_results['recall'] = result['recall']
            best_results['ndcg'] = result['ndcg']
        else:
            best_results['count'] += 1
            if best_results['count'] >= int(config['early_stopping']):
                print("Early stop......")
                print("Best epoch:   ", best_results['epoch'], " Best recall:", best_results['recall'], "Best NDCG:", best_results['ndcg'])
                best_results['stop'] = 99999
                return result, best_results

        print("Current epoch:", epoch + 1, " Test recall:", result['recall'], "Test NDCG:", result['ndcg'])
        print("Best epoch:   ", best_results['epoch'], " Best recall:", best_results['recall'], "Best NDCG:", best_results['ndcg'])

    else:
        result = sparsity_test(dataset, model, device, config)
        print("\t level_1: recall:", result[0]['recall'], ',ndcg:', result[0]['ndcg'])
        print("\t level_2: recall:", result[1]['recall'], ',ndcg:', result[1]['ndcg'])
        print("\t level_3: recall:", result[2]['recall'], ',ndcg:', result[2]['ndcg'])
        print("\t level_4: recall:", result[3]['recall'], ',ndcg:', result[3]['ndcg'])

    return result, best_results


def Test(dataset: Data, model, device, config):
    model = model.eval()

    topK = eval(config['top_K'])

    model_results = {'precision': np.zeros(len(topK)),
                     'recall': np.zeros(len(topK)),
                     'hit': np.zeros(len(topK)),
                     'ndcg': np.zeros(len(topK))}

    with torch.no_grad():
        users = list(dataset.test_dict.keys())  # get user list to test
        users_list, rating_list, ground_true_list = [], [], []
        num_batch = len(users) // int(config['test_batch_size']) + 1

        for batch_users in mini_batch(users, batch_size=int(config['test_batch_size'])):
            exclude_users, exclude_items = [], []
            all_positive = dataset.get_user_pos_items(batch_users)
            ground_true = [dataset.test_dict[u] for u in batch_users]

            batch_users_device = torch.Tensor(batch_users).long().to(device)

            rating = model.get_rating_for_test(batch_users_device)

            # Positive items are excluded from the recommended list
            for i, items in enumerate(all_positive):
                exclude_users.extend([i] * len(items))
                exclude_items.extend(items)
            rating[exclude_users, exclude_items] = -1

            # get the top-K recommended list for all users
            _, rating_k = torch.topk(rating, k=max(topK))

            rating = rating.cpu()
            del rating

            users_list.append(batch_users)
            rating_list.append(rating_k.cpu())
            ground_true_list.append(ground_true)

        assert num_batch == len(users_list)
        enum_list = zip(rating_list, ground_true_list)

        results = []
        for single_list in enum_list:
            results.append(test_one_batch(single_list, topK))

        for result in results:
            model_results['recall'] += result['recall']
            model_results['precision'] += result['precision']
            model_results['ndcg'] += result['ndcg']

        model_results['recall'] /= float(len(users))
        model_results['precision'] /= float(len(users))
        model_results['ndcg'] /= float(len(users))

        return model_results


def test_one_batch(X, topK):
    recommender_items = X[0].numpy()
    ground_true_items = X[1]
    r = metrics.get_label(ground_true_items, recommender_items)
    precision, recall, ndcg = [], [], []

    for k_size in topK:
        recall.append(metrics.recall_at_k(r, k_size, ground_true_items))
        precision.append(metrics.precision_at_k(r, k_size, ground_true_items))
        ndcg.append(metrics.ndcg_at_k(r, k_size, ground_true_items))

    return {'recall': np.array(recall), 'precision': np.array(precision), 'ndcg': np.array(ndcg)}


def sparsity_test(dataset: Data, model, device, config):
    sparsity_results = []
    model = model.eval()
    # top-20, 40, ..., 100
    topK = eval(config['top_K'])

    with torch.no_grad():
        for users in dataset.split_test_dict:
            model_results = {
                'precision': np.zeros(len(topK)),
                'recall': np.zeros(len(topK)),
                'hit': np.zeros(len(topK)),
                'ndcg': np.zeros(len(topK))
            }
            users_list, rating_list, ground_true_list = [], [], []
            num_batch = len(users) // int(config['test_batch_size']) + 1

            for batch_users in mini_batch(users, batch_size=int(config['test_batch_size'])):
                exclude_users, exclude_items = [], []
                all_positive = dataset.get_user_pos_items(batch_users)
                ground_true = [dataset.test_dict[u] for u in batch_users]

                batch_users_device = torch.Tensor(batch_users).long().to(device)

                rating = model.get_rating_for_test(batch_users_device)

                # Positive items are excluded from the recommended list
                for i, items in enumerate(all_positive):
                    exclude_users.extend([i] * len(items))
                    exclude_items.extend(items)

                rating[exclude_users, exclude_items] = -1

                # get the top-K recommended list for all users
                _, rating_k = torch.topk(rating, k=max(topK))

                rating = rating.cpu()
                del rating

                users_list.append(batch_users)
                rating_list.append(rating_k.cpu())
                ground_true_list.append(ground_true)

            assert num_batch == len(users_list)
            enum_list = zip(rating_list, ground_true_list)

            results = []
            for single_list in enum_list:
                results.append(test_one_batch(single_list, topK))

            for result in results:
                model_results['recall'] += result['recall']
                model_results['precision'] += result['precision']
                model_results['ndcg'] += result['ndcg']

            model_results['recall'] /= float(len(users))
            model_results['precision'] /= float(len(users))
            model_results['ndcg'] /= float(len(users))
            sparsity_results.append(model_results)

    return sparsity_results