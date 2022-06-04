# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import os
import random
import timeit
from math import ceil
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from dataset_utils import ListDataset, get_nli_examples, MNLI_LABELS, DATASET_COLLATE_FN_MAPPING, DATASET_LABEL_STATS_MAPPING, naive_collate_fn
from metrics import evaluate_and_save_mnli
from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, 
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,    
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_roberta import create_position_ids_from_input_ids
from run_nli import load_and_cache_examples
from common.config import *
from common.utils import mkdir_f
from expl_models.latattr_models import LAtAttrRobertaForSequenceClassification
from vis_tools.vis_utils import visualize_token_attributions
from run_tokig import dump_tokig_info, ig_analyze
from expl_models.perturb_models import run_lime_attribution, run_shap_attribution
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def predict_with_mask(active_mask, tokenizer,  model, base_inputs, full_input_ids):
    input_ids = tokenizer.mask_token_id * torch.ones_like(full_input_ids)
    input_ids[0, active_mask == 1]  = full_input_ids[0, active_mask == 1]
    prob = model.probe_forward(**base_inputs, input_ids=input_ids).item()
    return prob

def batch_predict_with_mask(list_mask, tokenizer,  model, base_inputs, batch_size, full_input_ids):
    # inputs = {
    #     "input_ids": batch[0],
    #     "attention_mask": batch[1],
    #     "token_type_ids": batch[2],        
    #     "pred_indexes": batch_pred_indexes,
    #     "final_logits": batch_logits,
    # }
    
    num_instances = len(list_mask)
    num_batch = int(ceil(num_instances / batch_size))
    batched_inputs = {
        'attention_mask': base_inputs['attention_mask'].expand(batch_size, -1),
        'pred_indexes': base_inputs['pred_indexes'].expand(batch_size),
        'final_logits': base_inputs['final_logits'].expand(batch_size, -1),
        'position_ids': base_inputs['position_ids'].expand(batch_size, -1),
        'return_kl': False,
    }
    if 'token_type_ids' in base_inputs:
        batched_inputs['token_type_ids'] = base_inputs['token_type_ids'].expand(batch_size, -1)
    
    all_scores = []
    for batch_idx in range(num_batch):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        
        # last batch
        last_batch = batch_end > num_instances
        if last_batch:
            batch_end = num_instances
            last_size = num_instances - batch_start
            batched_inputs = {
                'attention_mask': base_inputs['attention_mask'].expand(last_size, -1),
                'pred_indexes': base_inputs['pred_indexes'].expand(last_size),
                'final_logits': base_inputs['final_logits'].expand(last_size, -1),
                'position_ids': base_inputs['position_ids'].expand(last_size, -1),
                'return_kl': False,
            }
            if 'token_type_ids' in base_inputs:
                batched_inputs['token_type_ids'] = base_inputs['token_type_ids'].expand(last_size, -1)

        batched_input_ids = []
        for i in range(batch_start, batch_end):
            active_mask = list_mask[i]
            input_ids = tokenizer.mask_token_id * torch.ones_like(full_input_ids)
            input_ids[0, active_mask == 1]  = full_input_ids[0, active_mask == 1]
            batched_input_ids.append(input_ids)
        batched_input_ids = torch.cat(batched_input_ids)
        prob = model.probe_forward(**batched_inputs, input_ids=batched_input_ids)
        # print(prob.shape)
        all_scores.append(prob)
    all_scores = torch.cat(all_scores).cpu().numpy()
    return all_scores

def fit_locality(args, tokenizer, model, inputs, feature):    
    inputs['return_kl'] = False

    full_input_ids = inputs.pop('input_ids')
    doc_size = full_input_ids.size(1)
    full_positioin_ids = create_position_ids_from_input_ids(full_input_ids, tokenizer.pad_token_id).to(full_input_ids.device)

    # fix position id
    inputs['position_ids'] = full_positioin_ids
    # fix cls ? maybe    
    # score_fn = partial(predict_with_mask, tokenizer=tokenizer, model=model, base_inputs=inputs, full_input_ids=full_input_ids)
    score_fn = partial(batch_predict_with_mask, tokenizer=tokenizer, model=model, base_inputs=inputs, full_input_ids=full_input_ids, batch_size=args.eval_perturb_size)
    if args.variant == 'lime':
        np_attribution = run_lime_attribution(args, doc_size, score_fn).reshape((1,-1))
    elif args.variant == 'shap':
        np_attribution = run_shap_attribution(args, doc_size, score_fn).reshape((1,-1))
    else:
        raise RuntimeError('Variant must be shap or lime')
    return torch.from_numpy(np_attribution)

def predict_and_fit_locality(args, batch, model, tokenizer, batch_examples):
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)
    assert batch[0].size(0) == 1    
    # run predictions
    with torch.no_grad():
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }

        if args.model_type in ["roberta", "distilbert", "camembert", "bart"]:
            del inputs["token_type_ids"]
        outputs = model.forward(**inputs)

    batch_logits = outputs[0]
    # run attributions
    batch_pred_indexes = torch.argmax(batch_logits, dim=1)
    
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],        
        "pred_indexes": batch_pred_indexes,
        "final_logits": batch_logits,
    }
    if args.model_type in ["roberta", "distilbert", "camembert", "bart"]:
        del inputs["token_type_ids"]
    
    batch_attributions = fit_locality(args, tokenizer, model, inputs, batch_examples[0]) 
    return batch_logits, batch_attributions

def perturb_interp(args, model, tokenizer, prefix=""):
    if not os.path.exists(args.interp_dir):
        os.makedirs(args.interp_dir)
    model.requires_grad_(False)
    dataset, examples = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    # dataset.examples = dataset.examples

    specific_collate_fn = partial(DATASET_COLLATE_FN_MAPPING[args.dataset], tokenizer)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1, collate_fn=naive_collate_fn)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    all_predictions = []
    start_time = timeit.default_timer()

    for batch_examples in tqdm(eval_dataloader, desc="Evaluating", disable=args.disable_tqdm):
        model.eval()
        batch = specific_collate_fn(batch_examples)
        batch_predictions, batch_attributions = predict_and_fit_locality(
            args,
            batch,
            model,
            tokenizer,
            batch_examples
        )
        dump_tokig_info(args, batch_examples, batch_predictions, batch_attributions)
        all_predictions.append(batch_predictions)
        
    all_predictions = torch.cat(all_predictions)
    evalTime = timeit.default_timer() - start_time
    results = evaluate_and_save_mnli(args, examples[:len(all_predictions)], all_predictions, args.output_dir)
    return results

def main():
    parser = argparse.ArgumentParser()
    register_args(parser)
    
    parser.add_argument("--eval_perturb_size", default=20, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--variant",default=None,type=str,help='shap or lime')
    parser.add_argument("--do_vis", action="store_true", help="Whether to run vis on the dev set.")
    parser.add_argument("--interp_dir",default=None,type=str,required=True,help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--visual_dir",default=None,type=str,help="The output visualization dir.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=len(DATASET_LABEL_STATS_MAPPING[args.dataset]),
        # finetuning_task=data_args.task_name,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)
   
    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_vis:
        ig_analyze(args, tokenizer)
    else:
        checkpoint = args.model_name_or_path
        model = LAtAttrRobertaForSequenceClassification.from_pretrained(checkpoint)  # , force_download=True)
        model.to(args.device)

        # Evaluate
        result = perturb_interp(args, model, tokenizer, prefix="")
        logger.info("Results: {}".format(result))

if __name__ == "__main__":
    main()
