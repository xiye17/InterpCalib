import json
from torch.utils.data import Dataset
import torch


MNLI_LABEL_MAPPING = {
    'neutral': 0,
    'entailment': 1,
    'contradiction': 2,
}

MNLI_LABELS = ['neutral', 'entailment', 'contradiction']

class NLIFeature:
    def __init__(self, premise, hypothesis, gold, genre, pair_id, id, idx, input_ids, token_type_ids, label):
        self.premise = premise
        self.hypothesis = hypothesis
        self.gold = gold
        self.genre = genre
        self.pair_id = pair_id
        self.id = id
        self.idx = idx
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.label = label

class ListDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i]

naive_collate_fn = lambda x: x

def mnli_collate_fn(tokenizer, data):
    batch_size = len(data)
    
    batched_input = {
        'input_ids': [x.input_ids for x in data],
        'token_type_ids': [x.token_type_ids for x in data]
    }
    
    encoded = tokenizer.pad(batched_input, return_tensors='pt')
    labels = torch.LongTensor([x.label for x in data])

    return encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids'], labels

def get_nli_examples(filename, tokenizer, split, max_seq_len):
    
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
        lines = [x.rstrip() for x in lines]
    
    examples = [json.loads(x) for x in lines]
    
    features = []
    for i, ex in enumerate(examples):
        premise = ex['sentence1']
        hypothesis = ex['sentence2']
        gold = ex['gold_label']
        genre = ex.get('genre', 'hans')
        pair_id = ex['pairID']
        id = f'{split}{i}'
        encoded = tokenizer(premise, hypothesis, truncation=True, max_length=max_seq_len, return_token_type_ids=True)
        input_ids = encoded['input_ids']
        token_type_ids = encoded['token_type_ids']

        # quick handling hans set
        if gold == 'non-entailment':
            gold = 'neutral'

        if gold not in MNLI_LABEL_MAPPING:
            continue
        label = MNLI_LABEL_MAPPING[gold]

        feat = NLIFeature(premise, hypothesis, gold, genre, pair_id, id, i, input_ids, token_type_ids, label)
        # print(premise, hypothesis, gold, genre, pair_id, id)
        # print(input_ids, token_type_ids, label)
        features.append(feat)
    # print(unlabeled)
    return features



DATASET_COLLATE_FN_MAPPING = {
    'mnli': mnli_collate_fn,
}

DATASET_LABEL_STATS_MAPPING = {
    'mnli': MNLI_LABEL_MAPPING,
}


if __name__=='__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    get_nli_examples('outputs/mismatched-dev_mnli.jsonl', tokenizer, 'dev', 128)