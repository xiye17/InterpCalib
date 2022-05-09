import os
import sys
sys.path.append('.')
from os.path import join
from common.utils import read_json, dump_json, load_bin, dump_to_bin
from collections import OrderedDict
from types import SimpleNamespace
from transformers import AutoTokenizer
import torch
from calib_exp.run_tagger import load_cached_dataset
import argparse
from common.index_feature import IndexedFeature, FeatureVocab
import numpy as np
from tqdm import tqdm
import string
import re
from collections import Counter

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--input_norm', type=str, default='all',
        choices=['none', 'all', 'counted'])
    parser.add_argument('--method', type=str, default='lime')
    parser.add_argument('--include_neg', default=False, action='store_true')
    parser.add_argument('--no_punct', dest='include_punct', default=True, action='store_false')
    parser.add_argument('--split', type=str, default=None)
    args = parser.parse_args()
    if args.split == None:
        args.split = 'addsent-dev' if args.dataset == 'squad' else 'dev'
    return args
    

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def load_interp_info(file_dict, qas_id):
    return torch.load(file_dict[qas_id])

def build_file_dict(args):
    # prefix = 'squad_sample-addsent_roberta-base'
    prefix = '{}_{}_roberta-base'.format(args.dataset, args.split)
    fnames = os.listdir(join('interpretations', args.method, prefix))
    qa_ids = [x.split('-',1)[1].split('.')[0] for x in fnames]
    fullnames = [join('interpretations', args.method, prefix, x) for x in fnames]
    return dict(zip(qa_ids, fullnames))

def merge_attention_by_segments(attention, segments):
    new_val = []
    for a, b in segments:
        new_val.append(np.sum(attention[a:b, :], axis=0))
    attention = np.stack(new_val, axis=0)
    new_val = []
    for a, b in segments:
        new_val.append(np.sum(attention[:, a:b], axis=1))
    attention = np.stack(new_val, axis=1)
    return attention

def aggregate_link_attribution(args, interp, tags):
    
    attribution_val = interp['attribution'].numpy()
    # attribution_val[attribution_val < 0 ]  = 0
    aggregated_attribution = np.sum(attribution_val, axis=0)
    aggregated_attribution = np.sum(aggregated_attribution, axis=0)

    if not args.include_neg:
        aggregated_attribution[aggregated_attribution < 0] = 0
    aggregated_attribution = merge_attention_by_segments(aggregated_attribution,  tags['segments'])
    aggregated_attribution = aggregated_attribution / np.sum(np.sum(aggregated_attribution))
    map_weight = aggregated_attribution

    diag_attribution = np.diag(aggregated_attribution)
    gather_weight = np.sum(aggregated_attribution, axis=1)
    dispatch_weight = np.sum(aggregated_attribution, axis=0)
    agg_weight = (gather_weight + dispatch_weight)

    return agg_weight, map_weight

def merge_attribution_by_segments(attention, segments):
    new_val = []
    for a, b in segments:
        new_val.append(np.sum(attention[a:b], axis=0))
    attention = np.stack(new_val, axis=0)
    return attention

def aggregate_token_attribution(args, interp, tags, polarity):
    attribution_val = interp['attribution'].numpy().copy()    
    if polarity == 'POS':
        attribution_val[attribution_val < 0] = 0
    elif polarity == 'NEG':
        attribution_val[attribution_val > 0] = 0
    elif polarity == 'NEU':
        pass
    else:
        raise RuntimeError('Invalid polarity')
        
    attribution_val = merge_attribution_by_segments(attribution_val, tags['segments'])
    assert attribution_val.shape[0] == len(tags['segments'])
    attribution_val = attribution_val / np.sum(attribution_val)
    return attribution_val

def normalize_token_attr(args, feat, attributions, norm_method=None):
    if norm_method is None:
        norm_method = args.input_norm
    if norm_method == 'none':
        return feat
    if norm_method == 'all':
        sum_v = np.sum(attributions)
        for k in feat.data:
            feat.data[k] = feat.data[k] / sum_v if sum_v != 0 else 0
        return feat
    if norm_method == 'counted':
        sum_v = sum(feat.data.values())
        for k in feat.data:
            feat.data[k] = feat.data[k] / sum_v if sum_v != 0 else 0
        return feat
    raise RuntimeError(norm_method)
        
def extract_token_attr_feature_in_question(args, words, tags, attributions):
    context_start = words.index('</s>')
    tags = tags[1:context_start]
    attributions = attributions[1:context_start]

    feat = IndexedFeature()
    unnorm = IndexedFeature()
    sum_v = 0
    for i, (token, pos, tag) in enumerate(tags):
        v = attributions[i]
        if pos == 'PUNCT' and not args.include_punct:
            continue
        feat.add('NORMED_TOK_Q_' + tag, v)
        unnorm.add('UNNORM_TOK_Q_' + tag, v)
        sum_v += v
    
    feat = normalize_token_attr(args, feat, attributions)
    feat.add_set(unnorm)
    feat.add('SUM_TOK_Q', sum_v)
    return feat

def extract_token_attr_feature_in_context(args, words, tags, attributions):
    context_start = words.index('</s>')
    tags = tags[context_start + 2: -1]
    attributions = attributions[context_start + 2: -1]

    feat = IndexedFeature()
    unnorm = IndexedFeature()
    sum_v = 0
    for i, (token, pos, tag) in enumerate(tags):
        v = attributions[i]
        if pos == 'PUNCT' and not args.include_punct:
            continue
        feat.add('NORMED_TOK_C_' + tag, v)
        feat.add('UNNORM_TOK_C_' + tag, v)
        sum_v += v

    feat = normalize_token_attr(args, feat, attributions)
    feat.add_set(unnorm)
    feat.add('SUM_TOK_C', sum_v)
    return feat

def extract_token_attr_feature_in_input(args, words, tags, attributions, ans_range):
    feat = IndexedFeature()
    context_start = words.index('</s>')
    normed_a_feat = IndexedFeature()
    unnormed_a_feat = IndexedFeature()
    for i, (token, pos, tag) in enumerate(tags):
        v = attributions[i]
        if pos == 'PUNCT' and not args.include_punct:
            continue
        feat.add('TOK_IN_' + tag, v)
        
        if i >= ans_range[0] and i <= ans_range[1]:
            unnormed_a_feat.add('UNNORM_TOK_A_' + tag, v)
            normed_a_feat.add('NORMED_TOK_A_' + tag, v)
    
    feat.add_set(normalize_token_attr(args, normed_a_feat, [], 'counted'))
    feat.add_set(unnormed_a_feat)
    return feat

def source_of_token(idx, tok, pos, tag, context_start, ans_range):
    if tok in ['<s>', '</s>']:
        return 'S'
    if idx >= 1 and idx < context_start:
        return 'Q'
    if idx >= ans_range[0] and idx <= ans_range[1]:
        # print(tok)
        return 'A'        
    return 'C'

def ranked_pair(a, b):
    return (a, b) if a < b else (b, a)

def extract_link_attr_feature(args, words, tags, attributions, ans_range):
    feat = IndexedFeature()
    context_start = words.index('</s>')
    for i, (i_token, i_pos, i_tag) in enumerate(tags):
        i_src = source_of_token(i, i_token, i_pos, i_tag, context_start, ans_range)
        for j, (j_token, j_pos, j_tag) in enumerate(tags):
            if (i_pos == 'PUNCT' or j_pos == 'PUNCT') and not args.include_punct:
                continue
            val = attributions[i][j]
            j_src = source_of_token(j, j_token, j_pos, j_tag, context_start, ans_range)
            a_src, b_src = ranked_pair(i_src, j_src)
            a_tag, b_tag = ranked_pair(i_tag, j_tag)

            feat.add('LINK_{}_{}_{}_{}'.format(a_src, b_src, a_tag, b_tag), val)
            feat.add('LINK_{}_{}'.format(a_tag, b_tag), val)
            feat.add('LINK_AGG_{}_{}'.format(i_src, j_src), val)
    return feat

def extract_token_attr_stats_in_input(args, words, tags, attributions, part):
    feat = IndexedFeature()
    context_start = words.index('</s>')
    if part == 'Q':
        tags = tags[1:context_start]
        attributions = attributions[1:context_start]
    if part == 'C':
        tags = tags[context_start + 2: -1]
        attributions = attributions[context_start + 2: -1]

    feat.add('STAT_MEAN_' + part, attributions.mean())
    feat.add('STAT_STD_' + part, attributions.std())
    return feat

def extract_bow_feature(args, words, tags, ans_range):
    feat = IndexedFeature()
    context_start = words.index('</s>')
    for i, (i_token, i_pos, i_tag) in enumerate(tags):
        i_src = source_of_token(i, i_token, i_pos, i_tag, context_start, ans_range)
        if i_src == 'Q' or i_src == 'A' or i_src == 'C':
            # print('BOW_{}_{}'.format(i_src, i_tag))
            feat.add('BOW_{}_{}'.format(i_src, i_tag))
            feat.add('BOW_IN_{}'.format(i_tag))
    return feat

def lematize_pos_tag(x):
    tok, pos, tag = x
    if tag == 'NNS':
        tag = 'NN'
    if tag == 'NNPS':
        tag = 'NNP'
    if tag.startswith('JJ'):
        tag = 'JJ'
    if tag.startswith('RB'):
        tag = 'RB'
    if tag.startswith('W'):
        tag = 'W'
    if tag.startswith('PRP'):
        tag = 'PRP'
    if tag.startswith('VB'):
        tag = 'VB'
    if pos == 'PUNCT':
        tag = 'PUNCT'
    return tok, pos, tag

def extract_baseline_feature(args, interp, preds):
    feat = IndexedFeature()
    # base feature
    for rank, p in enumerate(preds[:5]):
        feat.add(f'BASELINE_PROB_{rank}', p['probability'])
    feat.add('BASELINE_CONTEXT_LENGTH', len(interp['example'].context_text))
    feat.add('BASELINE_PRED_ANS_LENGTH', len(preds[0]['text']))

    top_pred = preds[0]['text']
    first_distinct_prob = 0
    for i, p in enumerate(preds[1:]):
        overlapping = f1_score(p['text'], top_pred)
        if overlapping > 0:
            continue
        first_distinct_prob = p['probability']
        # print(i+1, top_pred, p['text'], first_distinct_prob)
    feat.add('FIRST_DISTINCT_PROB', first_distinct_prob)

    return feat

def orig_eval_of_squad(prediction, ex):
    ground_truths = list(map(lambda x: x['text'], ex.answers)) if ex.answers else [ex.answer_text]
    cur_exact_match = metric_max_over_ground_truths(exact_match_score,
                                                    prediction, ground_truths)
    cur_f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    return cur_exact_match, cur_f1

def extract_feature_for_instance(args, interp, tags, preds):
    # label
    pred_text = preds[0]['text']
    ex = interp['example']
    # gt_text = answer_text if ex.answer_text else ex.answers[0]['text']
    exact_match, f1 = orig_eval_of_squad(pred_text, ex)
    calib_label = 1 if exact_match > 0 else 0
    
    named_feat = IndexedFeature()
    named_feat.add_set(extract_baseline_feature(args, interp, preds))

    prelim_result = interp['prelim_result']
    segments = tags['segments']
    transformed_start_idx = 0
    while segments[transformed_start_idx + 1][0] <= prelim_result['start_index']:
        transformed_start_idx += 1
    transformed_end_idx = 0
    while segments[transformed_end_idx][1] < (prelim_result['end_index'] + 1):
        transformed_end_idx += 1
    ans_range = (transformed_start_idx, transformed_end_idx)
    # syntactic feature    
    words, tags_for_tok = tags['words'], tags['tags']
    tags_for_tok = [lematize_pos_tag(x) for x in tags_for_tok]
    named_feat.add_set(extract_bow_feature(args, words, tags_for_tok, ans_range))
    named_feat.add_set(extract_polarity_feature(args, interp, tags, words, tags_for_tok, ans_range, 'POS'))
    # named_feat.add_set(extract_polarity_feature(args, interp, tags, words, tags_for_tok, ans_range, 'NEG'))
    return {'feature': named_feat, 'label': calib_label, 'f1_score': f1}

def extract_polarity_feature(args, interp, tags, words, tags_for_tok, ans_index, polarity, include_basic=True, include_stats=False):
    named_feat = IndexedFeature()
    if args.method in ['tokig', 'lime', 'shap']:
        token_attribution = aggregate_token_attribution(args, interp, tags, polarity)
        link_attribution = None
    elif args.method in ['probe']:
        token_attribution, link_attribution = aggregate_link_attribution(args, interp, tags, polarity)
    assert token_attribution.size == len(words)
    if link_attribution is not None:
        assert link_attribution.shape == (len(words), len(words))
    if include_basic:
        named_feat.add_set(extract_token_attr_feature_in_question(args, words, tags_for_tok, token_attribution))
        named_feat.add_set(extract_token_attr_feature_in_context(args, words, tags_for_tok, token_attribution))
        named_feat.add_set(extract_token_attr_feature_in_input(args, words, tags_for_tok, token_attribution, ans_index))
        if link_attribution is not None:
            named_feat.add_set(extract_link_attr_feature(args, words, tags_for_tok, link_attribution, ans_index))
    if include_stats:
        named_feat.add_set(extract_token_attr_stats_in_input(args, words, tags_for_tok, token_attribution, 'Q'))
        named_feat.add_set(extract_token_attr_stats_in_input(args, words, tags_for_tok, token_attribution, 'C'))
        named_feat.add_set(extract_token_attr_stats_in_input(args, words, tags_for_tok, token_attribution, 'IN'))
    named_feat.add_prefix(polarity + '_')
    return named_feat

def label_sanity_check(data):
    all_ex = 0
    all_f1 = 0
    for k, v in data.items():
        all_ex = all_ex + v['label']
        all_f1 = all_f1 + v['f1_score']
    all_ex = all_ex / len(data)
    all_f1 = all_f1 / len(data)
    print('CHECKING EX: {:.3f}, F1; {:.3f}'.format(all_ex, all_f1))

def main():
    args = _parse_args()
    tokenizer = AutoTokenizer.from_pretrained('roberta-base',do_lower_case=False,cache_dir='hf_cache')
    tagger_info = load_bin('misc/{}_{}_tag_info.bin'.format(args.dataset, args.split))
    interp_dict = build_file_dict(args)
    preds_info = read_json('misc/{}_{}_predictions.json'.format(args.split, args.dataset))
    
    proced_instances = OrderedDict()
    for qas_id in tqdm(tagger_info, total=len(tagger_info),desc='transforming'):
        tags = tagger_info[qas_id]
        preds = preds_info[qas_id]
        if qas_id not in interp_dict:
            continue
        interp = load_interp_info(interp_dict, qas_id)
        proced_instances[qas_id] = extract_feature_for_instance(args, interp, tags, preds)
    dump_to_bin(proced_instances, 'calib_exp/data/{}_{}_{}_calib_data.bin'.format(args.dataset, args.split, args.method))
    # label_sanity_check(proced_instances)


if __name__ == "__main__":
    main()
    