import pickle
import json



def dump_to_bin(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load_bin(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def add_common_args(parser):
    parser.add_argument('--do_mini', action='store_true', default=False, help='Test with mini dataset for debugging')    
    parser.add_argument('--model_name', type=str, default='roberta-large', help='Model choosen')
    parser.add_argument('--cache_dir', default='hf_cache', type=str, help='custom cache dir')
    return parser

def read_json(fname):
    with open(fname, encoding='utf-8') as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)