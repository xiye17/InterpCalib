import os
import sys
sys.path.append('.')
from os.path import join
from common.utils import read_json, dump_json, load_bin, dump_to_bin
from collections import OrderedDict
from types import SimpleNamespace
from transformers import AutoTokenizer
import torch
from calib_exp.calib_utils import load_cached_dataset
import argparse
from common.indexed_feature import IndexedFeature, FeatureVocab
import numpy as np
from tqdm import tqdm
import string
import re
from collections import Counter

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnli')
    parser.add_argument('--method', type=str, default='atattr')
    parser.add_argument('--input_norm', type=str, default='all',
        choices=['none', 'all', 'counted'])
    parser.add_argument('--include_neg', default=False, action='store_true')
    parser.add_argument('--no_puct', dest='include_punct', default=True, action='store_false')
    parser.add_argument('--split', type=str, default=None)
    args = parser.parse_args()
    if args.split is None:
        args.split = 'subhans-dev' if args.dataset == 'mnli' else 'dev'
    return args

def load_interp_info(file_dict, qas_id):
    return torch.load(file_dict[qas_id])

def build_file_dict(args):
    # prefix = 'squad_sample-addsent_roberta-base'
    prefix = '{}_{}_roberta-base'.format(args.dataset, args.split)
    fnames = os.listdir(join('interpretations', args.method, prefix))
    ids = [ args.split + x.split('-',1)[0] for x in fnames]
    fullnames = [join('interpretations', args.method, prefix, x) for x in fnames]
    return dict(zip(ids, fullnames))

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

def aggregate_link_attribution(args, interp, tags, polarity):
    
    attribution_val = interp['attribution'].numpy().copy()
    # attribution_val[attribution_val < 0 ]  = 0
    aggregated_attribution = np.sum(attribution_val, axis=0)
    aggregated_attribution = np.sum(aggregated_attribution, axis=0)

    if polarity == 'POS':
        aggregated_attribution[aggregated_attribution < 0] = 0
    elif polarity == 'NEG':
        aggregated_attribution[aggregated_attribution > 0] = 0
    elif polarity == 'NEU':
        pass
    else:
        raise RuntimeError('Invalid polarity')
    aggregated_attribution = merge_attention_by_segments(aggregated_attribution,  tags['segments'])
    aggregated_attribution = aggregated_attribution / np.sum(np.sum(aggregated_attribution))
    map_weight = aggregated_attribution

    diag_attribution = np.diag(aggregated_attribution)
    gather_weight = np.sum(aggregated_attribution, axis=1)
    dispatch_weight = np.sum(aggregated_attribution, axis=0)
    agg_weight = (gather_weight + dispatch_weight)

    return agg_weight, map_weight

def aggregate_arch_attribution(args, interp, tags, polarity):
    importance = interp['importances']
    attribution_val = np.zeros((len(tags['words']), len(tags['words'])))
    for (i, j, imp, report) in importance:
        attribution_val[i, j] = imp
    # print(attribution_val)
    attribution_val = attribution_val + np.transpose(attribution_val)
    
    # attribution_val = interp['attribution'].numpy().copy()
    # attribution_val[attribution_val < 0 ]  = 0
    aggregated_attribution = attribution_val

    if polarity == 'POS':
        aggregated_attribution[aggregated_attribution < 0] = 0
    elif polarity == 'NEG':
        aggregated_attribution[aggregated_attribution > 0] = 0
    elif polarity == 'NEU':
        pass
    else:
        raise RuntimeError('Invalid polarity')
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
    if polarity != 'NEU':
        attribution_val = attribution_val / (np.sum(attribution_val) + 1e-8)
    return attribution_val


def normalize_token_attr(args, feat, attributions, norm_method=None):
    if norm_method is None:
        norm_method = args.input_norm
    if norm_method == 'none':
        return feat
    if norm_method == 'all':
        sum_v = np.sum(attributions)
        if sum_v == 0.0:
            return IndexedFeature()
        for k in feat.data:
            feat.data[k] = feat.data[k] / sum_v
        return feat
    if norm_method == 'counted':
        sum_v = sum(feat.data.values())
        for k in feat.data:
            feat.data[k] = feat.data[k] / sum_v
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
        feat.add('NORMED_TOK_P_' + tag, v)
        unnorm.add('UNNORM_TOK_P_' + tag, v)
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
        feat.add('NORMED_TOK_H_' + tag, v)
        feat.add('UNNORM_TOK_H_' + tag, v)
        sum_v += v

    feat = normalize_token_attr(args, feat, attributions)
    feat.add_set(unnorm)
    feat.add('SUM_TOK_C', sum_v)
    return feat

def extract_token_attr_feature_in_input(args, words, tags, attributions):
    feat = IndexedFeature()
    context_start = words.index('</s>')
    for i, (token, pos, tag) in enumerate(tags):
        v = attributions[i]
        if pos == 'PUNCT' and not args.include_punct:
            continue
        feat.add('TOK_IN_' + tag, v)

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

def source_of_token(idx, tok, pos, tag, context_start):
    if tok in ['<s>', '</s>']:
        return 'S'
    if idx >= 1 and idx < context_start:
        return 'P'
    return 'H'

def ranked_pair(a, b):
    return (a, b) if a < b else (b, a)

def extract_link_attr_feature(args, words, tags, attributions):
    feat = IndexedFeature()
    context_start = words.index('</s>')
    for i, (i_token, i_pos, i_tag) in enumerate(tags):
        i_src = source_of_token(i, i_token, i_pos, i_tag, context_start)
        for j, (j_token, j_pos, j_tag) in enumerate(tags):
            if (i_pos == 'PUNCT' or j_pos == 'PUNCT') and not args.include_punct:
                continue
            val = attributions[i][j]
            j_src = source_of_token(j, j_token, j_pos, j_tag, context_start)
            a_src, b_src = ranked_pair(i_src, j_src)
            a_tag, b_tag = ranked_pair(i_tag, j_tag)

            feat.add('LINK_{}_{}_{}_{}'.format(a_src, b_src, a_tag, b_tag), val)
            feat.add('LINK_{}_{}'.format(a_tag, b_tag), val)
            feat.add('LINK_AGG_{}_{}'.format(i_src, j_src), val)
    return feat

def extract_bow_feature(args, words, tags):
    feat = IndexedFeature()
    context_start = words.index('</s>')
    for i, (i_token, i_pos, i_tag) in enumerate(tags):
        i_src = source_of_token(i, i_token, i_pos, i_tag, context_start)
        if i_src == 'H' or i_src == 'P':
            # print('BOW_{}_{}'.format(i_src, i_tag))
            feat.add('BOW_{}_{}'.format(i_src, i_tag))
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

def augment_pos_tag_with_common(words, tags_for_tok):
    context_start = words.index('</s>')
    premise = words[1:context_start]
    hypothesis = words[context_start + 2:-1]
    # premise = [lemmatizer.lemmatize(x.lower()) for x in premise]
    # hypothesis = [lemmatizer.lemmatize(x.lower()) for x in hypothesis]
    common_toks = set(premise) & set(hypothesis)
    # print(tags_for_tok)
    new_tags = [(tag[0], tag[1], 'COM-' + tag[2]) if (word in common_toks) else tag for (word, tag) in zip(words, tags_for_tok)]
    return new_tags

def extract_baseline_feature(args, interp, preds):
    feat = IndexedFeature()
    # base feature
    feat.add('BASELINE_ENTAILMENT_PROB', preds['entailment'])
    feat.add('BASELINE_CONTRADICTION_PROB', preds['contradiction'])
    return feat

def orig_eval_of_squad(prediction, ex):
    ground_truths = list(map(lambda x: x['text'], ex.answers)) if ex.answers else [ex.answer_text]
    cur_exact_match = metric_max_over_ground_truths(exact_match_score,
                                                    prediction, ground_truths)
    cur_f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    return cur_exact_match, cur_f1

# hans is special
def get_split_specific_calib_label(preds, feature):
    if feature.genre == 'hans':
        entailment_prob = preds['entailment']
        pred_label = 1 if entailment_prob > 0.5 else 0
        calib_label = 1 if pred_label == feature.label else 0   
    else:
        pred_cat = sorted([(k, preds[k]) for k in preds], key=lambda x: x[1], reverse=True)[0][0]
        calib_label = 1 if pred_cat == feature.gold else 0
        # print(feature.gold, pred_cat, calib_label)
    return calib_label

def extract_feature_for_instance(args, interp, tags, preds):
    # label
    feature = interp['example']
    # gt_text = answer_text if ex.answer_text else ex.answers[0]['text']
    named_feat = IndexedFeature()
    named_feat.add_set(extract_baseline_feature(args, interp, preds))
 
    calib_label = get_split_specific_calib_label(preds, feature)
    words, tags_for_tok = tags['words'], tags['tags']
    tags_for_tok = [lematize_pos_tag(x) for x in tags_for_tok]
    tags_for_tok = augment_pos_tag_with_common(words, tags_for_tok)
    named_feat.add_set(extract_bow_feature(args, words, tags_for_tok))
    named_feat.add_set(extract_polarity_feature(args, interp, tags, words, tags_for_tok, 'POS', include_stats=False))
    named_feat.add_set(extract_polarity_feature(args, interp, tags, words, tags_for_tok, 'NEG', include_stats=False))
    # named_feat.add_set(extract_polarity_feature(args, interp, tags, words, tags_for_tok, 'NEU', include_basic=True, include_stats=False))
    
    return {'feature': named_feat, 'label': calib_label, 'baseid': feature.pair_id}

def extract_polarity_feature(args, interp, tags, words, tags_for_tok, polarity, include_basic=True, include_stats=False):
    named_feat = IndexedFeature()
    # syntactic feature 
    if args.method in ['tokig', 'lime', 'shap']:
        token_attribution = aggregate_token_attribution(args, interp, tags, polarity)
        link_attribution = None
    elif args.method in ['latattr', 'atattr']:
        token_attribution, link_attribution = aggregate_link_attribution(args, interp, tags, polarity)
    elif args.method in ['arch']:
        token_attribution, link_attribution = aggregate_arch_attribution(args, interp, tags, polarity)
    assert token_attribution.size == len(words)
    if link_attribution is not None:
        assert link_attribution.shape == (len(words), len(words))
    if include_basic:
        named_feat.add_set(extract_token_attr_feature_in_question(args, words, tags_for_tok, token_attribution))
        named_feat.add_set(extract_token_attr_feature_in_context(args, words, tags_for_tok, token_attribution))
        named_feat.add_set(extract_token_attr_feature_in_input(args, words, tags_for_tok, token_attribution))
        if link_attribution is not None:
            named_feat.add_set(extract_link_attr_feature(args, words, tags_for_tok, link_attribution))
    if include_stats:
        named_feat.add_set(extract_token_attr_stats_in_input(args, words, tags_for_tok, token_attribution, 'Q'))
        named_feat.add_set(extract_token_attr_stats_in_input(args, words, tags_for_tok, token_attribution, 'C'))
        named_feat.add_set(extract_token_attr_stats_in_input(args, words, tags_for_tok, token_attribution, 'IN'))

    named_feat.add_prefix(polarity + '_')
    return named_feat
    

def label_sanity_check(data):
    all_ex = 0
    for k, v in data.items():
        all_ex = all_ex + v['label']
    all_ex = all_ex / len(data)
    print('CHECKING EX: {:.3f}'.format(all_ex))

def main():
    args = _parse_args()
    tokenizer = AutoTokenizer.from_pretrained('roberta-base',do_lower_case=False,cache_dir='hf_cache')
    tagger_info = load_bin('misc/{}_{}_tag_info.bin'.format(args.dataset, args.split))
    interp_dict = build_file_dict(args)
    preds_info = read_json('misc/{}_{}_predictions.json'.format(args.dataset, args.split))
    
    proced_instances = OrderedDict()
    for id in tqdm(tagger_info, total=len(tagger_info),desc='transforming'):
        tags = tagger_info[id]
        preds = preds_info[id]
        if id not in interp_dict:            
            continue
        interp = load_interp_info(interp_dict, id)
        proced_instances[id] = extract_feature_for_instance(args, interp, tags, preds)
    dump_to_bin(proced_instances, 'calib_exp/data/{}_{}_{}_calib_data.bin'.format(args.dataset, args.split, args.method))
    label_sanity_check(proced_instances)


if __name__ == "__main__":
    main()