import os
import sys
sys.path.append('.')
from os.path import join
from common.utils import read_json, dump_json, load_bin, dump_to_bin
from collections import OrderedDict
from types import SimpleNamespace
from transformers import AutoTokenizer
import torch
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.tagging
import spacy
import string
from spacy.tokens import Doc
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--split', type=str, default=None)
    args = parser.parse_args()
    if args.split is None:
        args.split = 'addsent-dev' if args.dataset == 'squad' else 'dev'
    return args


def _merge_roberta_tokens_into_words(tokenizer, feature):
    tokens = feature.tokens

    decoded_each_tok = [
        bytearray([tokenizer.byte_decoder[c] for c in t]).decode("utf-8", errors=tokenizer.errors) for t in tokens
    ]

    token_to_orig_map = feature.token_to_orig_map

    end_points = []
    context_start = tokens.index(tokenizer.eos_token)
    force_break = False
    for i, t in enumerate(decoded_each_tok):
        # special token
        if t in tokenizer.all_special_tokens:
            end_points.append(i)
            force_break = True
            continue

        if t in string.punctuation:
            end_points.append(i)
            force_break = True
            continue

        # no alphanum
        if not any([x.isalnum() for x in t.lstrip()]):
            end_points.append(i)
            force_break = True
            continue

        if t.lstrip == "'s":
            end_points.append(i)
            force_break = True
            continue

        if force_break:
            end_points.append(i)
            force_break = False
            continue

        # if in question segment
        if i <= context_start:
            if t[0] == ' ':
                decoded_each_tok[i] = t[:]
                end_points.append(i)
        else:
            if token_to_orig_map[i] != token_to_orig_map[i - 1]:
                end_points.append(i)
    end_points.append(len(decoded_each_tok))

    # if in context segment
    segments = []
    for i in range(1, len(end_points)):
        if end_points[i - 1] == end_points[i]:
            continue
        segments.append((end_points[i - 1], end_points[i]))
    
    merged_tokens = []
    for s0, s1 in segments:
        merged_tokens.append(''.join(decoded_each_tok[s0:s1]))
    return merged_tokens, segments


def load_cached_dataset(dataset, split):
    cache_file = './cached/{}_{}_squad_roberta-base_512'.format(split, dataset)
    features_and_dataset = torch.load(cache_file)
    features, dataset, examples = (
        features_and_dataset["features"],
        features_and_dataset["dataset"],
        features_and_dataset["examples"],
    )
    return features, examples

def assign_pos_tags(hf_tokens, nlp):
    words = [x.lstrip() for x in hf_tokens]
    spaces = [ False if i == len(hf_tokens) - 1 else hf_tokens[i + 1][0] == ' ' for i in range(len(hf_tokens))]    
    valid_idx = [i for i, w in enumerate(words) if len(w)]

    words = [words[i] for i in valid_idx]
    spaces = [spaces[i] for i in valid_idx]
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    proced_tokens = nlp.tagger(doc)

    tag_info = [('','NULL', 'NULL')] * len(hf_tokens)
    for i, proc_tok in zip(valid_idx, proced_tokens):
        tag_info[i] = (proc_tok.text, proc_tok.pos_, proc_tok.tag_)
    return tag_info

def process_instance(tokenizer, nlp, feat, example):
    print(feat.qas_id)
    words, segments = _merge_roberta_tokens_into_words(tokenizer, feat)    
    question_end = words.index(tokenizer.sep_token)

    context_start = words.index(tokenizer.eos_token)
    question_tokens = words[1:context_start]
    conatext_tokens = words[context_start + 2: -1]
    question_tag_info = assign_pos_tags(question_tokens, nlp)
    context_tag_info = assign_pos_tags(conatext_tokens, nlp)
    tag_info = [('<s>', 'SOS', 'SOS')] + question_tag_info + [('<\s>', 'EOS', 'EOS'), ('<\s>', 'EOS', 'EOS')] + context_tag_info +  [('<\s>', 'EOS', 'EOS')]
    assert len(tag_info) == len(words)
    # print(tag_info)
    instance_info = {'words': words, 'segments': segments, 'tags': tag_info}
    return instance_info

def main():
    args = _parse_args()
    tokenizer = AutoTokenizer.from_pretrained('roberta-base',do_lower_case=False,cache_dir='hf_cache')
    features, examples = load_cached_dataset(args.dataset, args.split)
    nlp = spacy.load("en_core_web_sm") 
    
    proced_instances = OrderedDict()
    for feat, ex in zip(features, examples):
        proced_instances[feat.qas_id] = process_instance(tokenizer, nlp, feat, ex)
    dump_to_bin(proced_instances, 'misc/{}_{}_tag_info.bin'.format(args.dataset, args.split))

def demo_spacy():
    nlp = spacy.load("en_core_web_sm") 
    result = nlp.tagger(nlp.tokenizer('Tokenize the "Super Bowl 50" sentence'))
    for token in result:
        print(token, token.pos_, token.tag_)

if __name__ == "__main__":
    # demo_spacy()
    main()
    