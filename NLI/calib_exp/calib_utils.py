import torch

def load_cached_dataset(dataset, split='dev'):
    cache_file = './cached/{}_{}_mnli_roberta-base_128'.format(split, dataset)
    return torch.load(cache_file)
