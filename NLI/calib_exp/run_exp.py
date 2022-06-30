import os
import sys
from collections import OrderedDict
sys.path.append('.')
import argparse
import numpy as np
import random
from tqdm import tqdm
from common.indexed_feature import IndexedFeature, FeatureVocab
from common.utils import read_json, dump_json, load_bin, dump_to_bin
from itertools import chain
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnli')
    parser.add_argument('--method', type=str, default='lime')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--do_baseline', default=False, action='store_true')
    parser.add_argument('--do_maxprob', default=False, action='store_true')
    parser.add_argument('--do_bow', default=False, action='store_true')
    parser.add_argument('--do_unnorm', default=False, action='store_true')
    parser.add_argument('--do_tok', default=False, action='store_true')
    parser.add_argument('--rm_baseline', default=False, action='store_true')    
    parser.add_argument('--n_run', type=int, default=20)
    parser.add_argument('--train_size', type=int, default=500)
    parser.add_argument('--model', type=str, default='rf')
    parser.add_argument('--show_imp', default=False, action='store_true')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--force_dev_size', type=int, default=0)
    parser.add_argument('--arg_n_tree', type=int, default=100)
    parser.add_argument('--arg_max_depth', type=int, default=None)
    args = parser.parse_args()
    if args.split is None:
        args.split = 'subhans-dev' if args.dataset == 'mnli' else 'dev'
    return args

def f1_prob_curve(f1, score):
    sorted_idx = np.argsort(-score)
    score = score[sorted_idx]
    f1 = f1[sorted_idx]
    num_test = f1.size
    # T = [0.1, 0.2, 0.3, 0.4, 0.5]
    # T = [00.5, 0.6, 0.7, 0.8, 0.9]
    T = [0.25, 0.5, 0.75]
    results = np.array([np.mean(f1[:int(num_test * t)])  for t in T])
    return results

def auc_score(x, y):
    fpr, tpr, _ = roc_curve(y, x)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def train_max_accuracy(x, y):
    x = x.flatten()
    best_acc = 0
    best_v = 0
    for v in x:
        p = x > v
        ac = np.sum(p == y) / y.size
        if ac > best_acc:
            best_acc = ac
            best_v= v
    return best_acc, best_v

def test_max_accuracy(x, y, v):
    x = x.flatten()
    p = x > v
    ac = np.sum(p == y) / y.size
    return ac, v

def feat_to_list(indexed_feat, vocab):
    val_feat = [.0] * len(vocab)
    for f, v in indexed_feat.data.items():
        val_feat[vocab[f]] = v
    return val_feat

def make_np_dataset(indexed_data):
    vocab = FeatureVocab()
    for k, v in indexed_data.items():
        feat = v['feature']
        for f in feat.data:
            vocab.add(f)
    print('Total Num of Feature', len(vocab))
    y = np.array([v['label'] for v in indexed_data.values()])
    x = np.array([feat_to_list(v['feature'], vocab) for v in indexed_data.values()])
    
    for index, name in vocab.id_to_feat.items():
        if 'LENGTH' in name:
            x[:, index] = x[:, index] / np.max(x[:, index])
    return x, y, vocab

def interp_calibrator_model(cls, vocab):
    if isinstance(cls, LogisticRegression):
        feat_imp = cls.coef_.flatten()
    elif isinstance(cls, RandomForestClassifier):
        feat_imp = cls.feature_importances_
    elif isinstance(cls, GradientBoostingClassifier):
        feat_imp = cls.feature_importances_
    else:
        print('UnInterpretable model')
        return
    
    imp_idx = np.argsort(-np.abs(feat_imp))
    for i in imp_idx[:100]:
        print(vocab.get_word(i), feat_imp[i])

def proc_input_data(args, data):
    new_data = OrderedDict()
    for qas_id, ex in data.items():
        new_feat = IndexedFeature()
        for f, val in ex['feature'].data.items():
            if args.do_unnorm and 'NORMED' in f:
                continue
            if not args.do_unnorm and 'UNNORM' in f:
                continue
            if 'TOK_IN' in f:
                continue
            new_feat.add(f, val)
        new_data[qas_id] = {'label': ex['label'], 'feature': new_feat}
    data = new_data
    if args.do_baseline:
        new_data = OrderedDict()
        for qas_id, ex in data.items():
            new_feat = IndexedFeature()
            for f, val in ex['feature'].data.items():
                if not f.startswith('BASELINE'):
                    continue
                new_feat.add(f, val)
            new_data[qas_id] = {'label': ex['label'], 'feature': new_feat}
        data = new_data
    if args.do_maxprob:
        new_data = OrderedDict()
        for qas_id, ex in data.items():
            new_feat = IndexedFeature()
            prev_feat = ex['feature']
            entailment_prob = prev_feat['BASELINE_ENTAILMENT_PROB']
            maxprob = entailment_prob if entailment_prob > 0.5 else (1 - entailment_prob)
            new_feat.add('MAXPROB', maxprob)
            new_data[qas_id] = {'label': ex['label'], 'feature': new_feat}
        data = new_data
    if args.do_bow:
        new_data = OrderedDict()
        for qas_id, ex in data.items():
            new_feat = IndexedFeature()
            for f, val in ex['feature'].data.items():
                if not ('BASELINE' in f or 'BOW' in f):
                    continue
                new_feat.add(f, val)
            new_data[qas_id] = {'label': ex['label'], 'feature': new_feat}
        data = new_data
    if args.do_tok:
        new_data = OrderedDict()
        for qas_id, ex in data.items():
            new_feat = IndexedFeature()
            for f, val in ex['feature'].data.items():
                if 'LINK' in f:
                    continue
                new_feat.add(f, val)
            new_data[qas_id] = {'label': ex['label'], 'feature': new_feat}
        data = new_data
    return data


def get_feature_importances(cls):
    if isinstance(cls, LogisticRegression):
        feat_imp = cls.coef_.flatten()
    elif isinstance(cls, RandomForestClassifier):
        feat_imp = cls.feature_importances_
    elif isinstance(cls, GradientBoostingClassifier):
        feat_imp = cls.feature_importances_
    else:
        return None
    return feat_imp

# def train_test_split(X, Y, permuted_idx=None, ratio=0.75):
#     num_data = X.shape[0]
#     perm_idx = np.random.permutation(num_data) if permuted_idx is None else permuted_idx
#     num_train = int(num_data * ratio)
#     train_idx = perm_idx[:num_train]
#     dev_idx = perm_idx[num_train:]
#     train_x, train_y = X[train_idx, :], Y[train_idx]
#     dev_x, dev_y = X[dev_idx, :], Y[dev_idx]

#     return train_x, train_y, dev_x, dev_y

def apply_train_test_split(X, Y, train_test_split, force_dev_size=0):
    train_idx, dev_idx = train_test_split

    # for hyper par search
    if force_dev_size:
        train_size = len(train_idx)
        dev_idx = train_idx[(train_size-force_dev_size):]
        train_idx = train_idx[:(train_size-force_dev_size)]

    train_x, train_y = X[train_idx, :], Y[train_idx]
    dev_x, dev_y = X[dev_idx, :], Y[dev_idx]

    return train_x, train_y, dev_x, dev_y

def one_pass_exp(args, X, Y, vocab, train_test_split):    

    # train_x, train_y, dev_x, dev_y = train_test_split(X, Y, permuted_idx=permuted_idx, ratio=args.train_ratio)
    train_x, train_y, dev_x, dev_y = apply_train_test_split(X, Y, train_test_split, force_dev_size=args.force_dev_size)
    # print(train_x.shape, dev_x.shape)
    majority_acc = max(np.sum(dev_y == 0), np.sum(dev_y == 1)) / dev_y.size
    
    if args.do_maxprob:
        train_acc, best_threshould = train_max_accuracy(train_x, train_y)
        dev_acc, _= test_max_accuracy(dev_x, dev_y, best_threshould)
        dev_auc = auc_score(dev_x, dev_y)
        return majority_acc, train_acc, dev_acc, dev_auc, f1_prob_curve(dev_y, dev_x.flatten()), None
    
    if args.model == 'lr':
        clf = LogisticRegression(C=10000.0, max_iter=2000).fit(train_x, train_y)
    elif args.model == 'rf':
        clf = RandomForestClassifier(n_estimators=args.arg_n_tree, max_depth=args.arg_max_depth).fit(train_x, train_y)
    elif args.model == 'svm':
        clf = SVC(C=10.0).fit(train_x, train_y)
    elif args.model == 'gdbt':
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=10).fit(train_x, train_y)
    else:
        raise RuntimeError('Model not supported')
    # clf = DecisionTreeClassifier(max_depth=3).fit(train_x, train_y)
    # clf = GradientBoostingClassifier(random_state=args.seed, n_estimators=100, max_depth=3).fit(train_x, train_y)    
    train_pred = clf.predict(train_x)
    dev_pred = clf.predict(dev_x)
    # print('Label Distribution', np.sum(dev_y == 0)/dev_y.size, np.sum(dev_y == 1)/dev_y.size)
    # print('Train ACC', np.sum(train_pred == train_y)/train_pred.size,
    #     'Dev Acc', np.sum(dev_pred == dev_y) / dev_pred.size)
    # interp_calibrator_model(clf, vocab)
    train_acc = np.sum(train_pred == train_y)/train_pred.size
    dev_acc = np.sum(dev_pred == dev_y) / dev_pred.size

    dev_score = clf.predict_proba(dev_x)[:,1]
    f1_curve = f1_prob_curve(dev_y, dev_score)
    dev_auc = auc_score(dev_score, dev_y)
    return majority_acc, train_acc, dev_acc, dev_auc, f1_curve, get_feature_importances(clf)

def gen_predefined_train_test_splits(baseids, n_run, ratio):
    num_data = len(baseids)
    
    baseid_indexer = OrderedDict()
    for i, b in enumerate(baseids):
        if b in baseid_indexer:
            baseid_indexer[b].append(i)
        else:
            baseid_indexer[b] = [i]
    num_base = len(baseid_indexer)
    print('Number of Exs', num_data, 'Number of Unique Base', num_base)
    
    predefined_permutations = [np.random.permutation(num_base) for _ in range(n_run)]
    baseid_groups = list(baseid_indexer.values())
    num_train = ratio
    
    splits = []
    for perm in predefined_permutations:
        flat_idx = list(chain(*[baseid_groups[i] for i in perm]))
        train_index = flat_idx[:num_train]
        dev_index = flat_idx[num_train:]
        splits.append((train_index, dev_index))
    print('Train Size', len(splits[0][0]), 'Dev Size', len(splits[0][1]))
    return splits


def gen_fixed_train_test_splits(baseids, n_run, ratio):    
    num_data = len(baseids)
    x = list(range(num_data))
    random.seed(123)
    random.shuffle(x)
    splits = [(x[:500], x[2000:2500])]
    return splits

def quantify_colum(x, k=2, method='val'):
    if method == 'val':
        interval = np.arange(k) / k
        vals = (np.max(x) - np.min(x)) * interval + np.min(x)
    if method == 'percent':
        q = 100 * np.arange(k) / k
        vals = np.percentile(x, q)
    # print(np.min(x), np.max(x))
    # print(vals)
    # exit()
    new_x = np.zeros_like(x)

    for i, v in enumerate(vals):
        new_x[x >= v] = i / k
    return new_x

def quantify_dataset(X, vocab):
    # print(X.shape)
    for i in range(X.shape[1]):
        fname = vocab.get_word(i)
        if 'PROB' in fname:
            continue
        if 'BOW' in fname:
            continue
        # print(fname)
        X[:,i] = quantify_colum(X[:,i], k=5, method='val')
    return X, vocab

def main():
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_bin('calib_exp/data/{}_{}_{}_calib_data.bin'.format(args.dataset, args.split, args.method))
    baseids = [data[x]['baseid'] for x in data]
    data = proc_input_data(args, data)
    X, Y, vocab = make_np_dataset(data)
    # X, vocab = quantify_dataset(X, vocab)
    print(X.shape, Y.shape)
    print(np.sum(Y == 0)/ Y.size, np.sum(Y == 1)/ Y.size)

    # num_data = X.shape[0]
    # predefined_permutations = [np.random.permutation(num_data) for _ in range(args.n_run)]
    predefined_splits = gen_predefined_train_test_splits(baseids, args.n_run, args.train_size)
    agg_results = []
    for train_test_split in tqdm(predefined_splits, total=len(predefined_splits), desc='Runing Random Exp'):
        agg_results.append(one_pass_exp(args, X, Y, vocab, train_test_split))
    
    agg_base_acc = np.array([x[0] for x in agg_results])
    agg_train_acc = np.array([x[1] for x in agg_results])
    agg_dev_acc = np.array([x[2] for x in agg_results])
    agg_auc = np.array([x[3] for x in agg_results])
    agg_f1_curve = np.array([x[4] for x in agg_results]).mean(axis=0)

    print('AVG MAJORITY ACC {:.3f}'.format(agg_base_acc.mean()))
    print('AVG TRAIN ACC {:.3f}'.format(agg_train_acc.mean()))
    print('AVG DEV ACC: {:.3f} +/- {:.3f}, AUC: {:.3f}, MAX: {:.3f}, MIN: {:.3f}'.format(agg_dev_acc.mean(), agg_dev_acc.std(), agg_auc.mean(), agg_dev_acc.max(), agg_dev_acc.min()))
    
    # print numbers for copy paste
    exp_numbers = [agg_base_acc.mean(), agg_dev_acc.mean(), agg_auc.mean()]
    print(','.join(['{:.3f}'.format(x) for x in exp_numbers]))
    exp_numbers = exp_numbers[1:]
    print(','.join(['{:.1f}'.format(a * 100) for a in exp_numbers]))

    if agg_results[0][5] is not None and args.show_imp:
        agg_feat_imp = np.array([x[5] for x in agg_results])
        # print(agg_feat_imp.shape)
        agg_feat_imp = np.mean(agg_feat_imp, axis=0)
        imp_idx = np.argsort(-np.abs(agg_feat_imp))
        for rank, i in enumerate(imp_idx[:20]):
            print(rank, vocab.get_word(i), agg_feat_imp[i])
        
if __name__ == "__main__":
    main()