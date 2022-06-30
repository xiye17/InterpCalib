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
from run_nli import load_and_cache_examples
from common.config import register_args
from common.utils import mkdir_f
from expl_models.tokig_models import TokIGRobertaForSequenceClassification
from vis_tools.vis_utils import visualize_token_attributions

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def remove_padding(batch, feature):    
    new_batch = tuple(x[:,:len(feature.input_ids)] for x in batch[:3]) + (batch[3],)
    return new_batch

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def ig_analyze(args, tokenizer):
    filenames = os.listdir(args.interp_dir)
    filenames.sort(key=lambda x: int(x.split('-')[0]))
    # print(len(filenames))
    mkdir_f(args.visual_dir)
    for fname in tqdm(filenames, desc='Visualizing'):
        interp_info = torch.load(os.path.join(args.interp_dir, fname))
        # datset_stats.append(stats_of_ig_interpretation(tokenizer, interp_info))
        visualize_token_attributions(args, tokenizer, interp_info)

def dump_tokig_info(args, examples, predictions, attributions):
    
    # attentions, attributions
    # N_Layer * B * N_HEAD * L * L
    attributions = attributions.detach().cpu().requires_grad_(False)
    predictions = torch.softmax(predictions, dim=1)
    predictions = predictions.detach().cpu().requires_grad_(False)

    for example, prediction, attribution in zip(
        examples,
        predictions,
        torch.unbind(attributions)
    ):
        actual_len = len(example.input_ids)
        attribution = attribution[:actual_len].clone()
        filename = os.path.join(args.interp_dir, f'{example.idx}-{example.pair_id}.bin')
        torch.save({'example': example, 'probs': prediction, 'prediction': MNLI_LABELS[torch.argmax(prediction).item()],
            'attribution': attribution}, filename)

def predict_and_tokig_attribute(args, batch, model, tokenizer, batch_examples):
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)
    # run predictions
    with torch.no_grad():
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "output_attentions": True,
        }

        if args.model_type in ["roberta", "distilbert", "camembert", "bart"]:
            del inputs["token_type_ids"]
        outputs = model.forward(**inputs)

    batch_logits, batch_attentions = outputs
    # run attributions
    batch_pred_indexes = torch.argmax(batch_logits, dim=1)
    batch_attentions = torch.stack(batch_attentions)
    
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],        
        "pred_indexes": batch_pred_indexes,
        "final_logits": batch_logits,
    }
    if args.model_type in ["roberta", "distilbert", "camembert", "bart"]:
        del inputs["token_type_ids"]

    # for data parallel 
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],        
        "pred_indexes": batch_pred_indexes,
        "final_logits": batch_logits,
        "num_steps": args.ig_steps,
        "do_attribute": True,
    }
    if args.model_type in ["roberta", "distilbert", "camembert", "bart"]:
        del inputs["token_type_ids"]
    
    batch_attributions = model.forward(**inputs)    
    return batch_logits, batch_attentions, batch_attributions

def tokig_interp(args, model, tokenizer, prefix=""):
    if not os.path.exists(args.interp_dir):
        os.makedirs(args.interp_dir)
    model.requires_grad_(False)
    dataset, examples = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    # dataset.examples = dataset.examples

    specific_collate_fn = partial(DATASET_COLLATE_FN_MAPPING[args.dataset], tokenizer)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=naive_collate_fn)

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
        batch_predictions, batch_attentions, batch_attributions = predict_and_tokig_attribute(
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

    parser.add_argument("--ig_steps", type=int, default=300, help="steps for running integrated gradient")
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
        model = TokIGRobertaForSequenceClassification.from_pretrained(checkpoint)  # , force_download=True)
        model.to(args.device)

        # Evaluate
        result = tokig_interp(args, model, tokenizer, prefix="")
        logger.info("Results: {}".format(result))

if __name__ == "__main__":
    main()
