from os.path import join
from collections import OrderedDict
import torch

from torch.nn.functional import softmax
from dataset_utils import MNLI_LABEL_MAPPING
from common.utils import *

def evaluate_and_save_mnli(args, examples, predictions, output_dir=None):
    probs = softmax(predictions, dim=1).tolist()
    preds = torch.argmax(predictions, dim=1).tolist()
    
    acc = 0
    probs_file = OrderedDict()
    hans_format_file = OrderedDict()
    label_str_map = dict([(v,k) for (k, v) in MNLI_LABEL_MAPPING.items()])
    for ex, prob, pred in zip(examples, probs, preds):        
        #handling hans
        if ex.genre == 'hans':
            pred = MNLI_LABEL_MAPPING['entailment'] if prob[MNLI_LABEL_MAPPING['entailment']] >= 0.5 else MNLI_LABEL_MAPPING['neutral']
            # print(pred, ex.label)
        acc += 1 if ex.label == pred else 0
        probs_file[ex.id] = dict([(label_str_map[i], p) for (i, p) in enumerate(prob)])
        hans_label = label_str_map[pred]
        if hans_label != 'entailment':
            hans_label = 'non-entailment'
        hans_format_file[ex.pair_id] = hans_label
    
    if output_dir:
        with open(join(output_dir, 'hansform_pred.txt'), 'w') as f:
            lines = ['pairID,gold_label\n'] + ['{},{}\n'.format(k,v) for (k,v) in hans_format_file.items()]
            f.writelines(lines)
        dump_json(probs_file, join(output_dir, 'predictions.json'))
    # sanity_check(probs_file, read_json(join(output_dir, 'predictions.json')))
    acc = acc/len(examples)
    
    return {'acc': acc, 'size': len(examples)}