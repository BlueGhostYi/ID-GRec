import numpy as np


def ndcg_at_k(r, k, test_data):
    """
        Normalized discounted cumulative gain
    """
    assert len(r) == len(test_data)

    prediction_data = r[:, :k]
    test_matrix = np.zeros((len(prediction_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1

    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(prediction_data * (1. / np.log2(np.arange(2, k + 2))), axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def recall_at_k(r, k, test_data):
    right_prediction = r[:, :k].sum(1)
    recall_num = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_prediction / recall_num)
    return recall


def precision_at_k(r, k, test_data):
    right_prediction = r[:, :k].sum(1)
    precision_num = k
    precision = np.sum(right_prediction) / precision_num
    return precision


def F1(pre, rec):
    F1 = []
    for i in range(len(pre)):
        if pre[i] + rec[i] > 0:
            F1.append((2.0 * pre[i] * rec[i]) / (pre[i] + rec[i]))
        else:
            F1.append(0.)
    return F1


def get_label(true_data, pred_data):
    r = []
    for i in range(len(true_data)):
        ground_true = true_data[i]
        pred_top_k = pred_data[i]
        pred = list(map(lambda x: x in ground_true, pred_top_k))
        pred = np.array(pred).astype("float")
        r.append(pred)

    return np.array(r).astype("float")
