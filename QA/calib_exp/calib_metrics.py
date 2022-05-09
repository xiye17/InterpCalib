import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

def auc_score(x, y):
    fpr, tpr, _ = roc_curve(y, x)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def _f1auc_score(x, y):
    x = np.ravel(x)
    y = np.ravel(y)
    desc_score_indices = np.argsort(x, kind="mergesort")[::-1]
    x = x[desc_score_indices]
    y = y[desc_score_indices]

    distinct_value_indices = np.where(np.diff(x, append=0))[0]
    # threshold_idxs = np.r_[distinct_value_indices, y.size - 1]

    # accumulate the true positives with decreasing threshold
    threshold_values = x[distinct_value_indices]

    # for t in threshold_values:
    #     results = np.array([np.mean(y[x < t])  for t in T])
    threshold_f1 = np.array([np.mean(y[:(t + 1)]) for t in distinct_value_indices])
    # print(threshold_values)
    print(threshold_f1)
    return auc(threshold_values, threshold_f1)

def f1auc_score(score, f1):
    score = np.ravel(score)
    f1 = np.ravel(f1)
    sorted_idx = np.argsort(-score)
    score = score[sorted_idx]
    f1 = f1[sorted_idx]
    num_test = f1.size
    segment = min(1000, score.size - 1)
    T = np.arange(segment) + 1
    T = T/segment
    results = np.array([np.mean(f1[:int(num_test * t)])  for t in T])
    # print(results)
    return np.mean(results)