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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pickle
import random
import torch
import numpy as np
from itertools import cycle
import datetime

from model import Model
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer, AutoTokenizer, RobertaPreLayerNormConfig, RobertaPreLayerNormForMaskedLM, RobertaPreLayerNormModel
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist  
from torch.distributed.fsdp import ( 
    FullyShardedDataParallel as FSDP, 
    MixedPrecision, 
    FullStateDictConfig, 
    StateDictType, 
) 
from torch.utils.data.distributed import DistributedSampler 

from preprocess import preprocess
from dataset import TextDataset

logger = logging.getLogger(__name__)        
        

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def merge_dataset(dataset_list, dataset_size, task_ratio):
    """merge multiple datasets with different languages, and sample a batch according a distribution"""
    iterators = [iter(d) for d in dataset_list]

    n = len(dataset_list)
    ids = list(range(n))
    probs = [float(item) / sum(dataset_size) for item in dataset_size]
    probs = [item**task_ratio for item in probs]
    probs = [float(item) / sum(probs) for item in probs]

    while True:
        i = random.choices(ids, probs)[0]
        try:
            item = next(iterators[i])
        except StopIteration:
            iterators[i] = iter(dataset_list[i])
            item = next(iterators[i])
        yield i, item
        
def train(args, train_datasets, eval_dataset, model, tokenizer):
    """ Train the model """
    # set tensorboard
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(log_dir=args.tensorboard_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_samplers = [RandomSampler(train_dataset) for train_dataset in train_datasets]
    
    train_dataloaders = [DataLoader(train_dataset, 
                                    sampler = train_sampler, 
                                    batch_size = args.train_batch_size, 
                                    drop_last = True, 
                                    num_workers = 4) 
                         for train_dataset,train_sampler in zip(train_datasets,train_samplers)]
    
    model.to(args.device)
    
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()     
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_steps,
                                                num_training_steps = args.max_steps)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))   
        
    if args.local_rank == 0:
        torch.distributed.barrier()    
    

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", sum([len(train_dataset) for train_dataset in train_datasets])* (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)) 
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)


    global_step = args.start_step
    losses, mlm_losses, step = [], [], 0

    model.zero_grad()
    set_seed(args)  

    for _,batch in merge_dataset(train_dataloaders,[len(x) for x in train_datasets],0.7):
        model.train()
        step+=1
        
        # forward
        mlm_ids = [x.to(args.device) for x in batch]        
        loss, mlm_loss = model(mlm_ids[0])
        
        # store loss
        losses.append(loss.item())
        if mlm_loss != 0:
            mlm_losses.append(mlm_loss)                
        if args.n_gpu > 1:
            loss = loss.mean() 
        
        # backward
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
            
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # update model
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  
            global_step += 1
            
            if args.local_rank in [-1, 0] and global_step % 5 == 0:
                writer.add_scalar("op_lr", optimizer.state_dict()['param_groups'][0]['lr'], global_step)
                writer.add_scalar("loss", round(np.mean(losses),3), global_step)
            
            if global_step % 500 == 0:
                logger.info("steps: %s avg: %s mlm: %s ", global_step, 
                            round(np.mean(losses),3),
                            round(np.mean(mlm_losses),3),
                           )
                losses, mlm_losses = [], []

            
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True) 
                with FSDP.state_dict_type( 
                    model, StateDictType.FULL_STATE_DICT, save_policy 
                ): 
                    # logger.info("***processing parameter***")
                    model_best_state = model.state_dict()

                # evaluate model and save model
                if args.local_rank in [-1, 0]:  
                    checkpoint_prefix = 'checkpoint'
                    results = evaluate(args, model, tokenizer,eval_dataset)
                    for key, value in results.items():
                        logger.info("  %s = %s", key, round(value,6))      
                        
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, 
                                                                                global_step, 
                                                                                round(results['loss'], 6)))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    
                    model_to_save = {} 
                    for key, value in model_best_state.items(): 
                        # logger.info("model key: {}, value: {}".format(key,value))
                        if 'encoder' in key: 
                            new_key = key.split('.', 1)[1] 
                            model_to_save[new_key] = value 
                    logger.info("***start save model***")
                    output_file = os.path.join(output_dir, 'pytorch_model.bin') 
                    torch.save(model_to_save, output_file) 
                    output_config_file = os.path.join(output_dir, 'config.json') 
                    model.config.to_json_file(output_config_file) 

                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)


                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    last_output_file = os.path.join(last_output_dir, 'pytorch_model.bin') 
                    torch.save(model_to_save, last_output_file) 
                    last_output_config_file = os.path.join(last_output_dir, 'config.json') 
                    model.config.to_json_file(last_output_config_file) 

                    tokenizer.save_pretrained(last_output_dir)
                    idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                    with open(idx_file, 'w', encoding='utf-8') as idxf:
                        idxf.write(str(0) + '\n')

                    torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                    step_file = os.path.join(last_output_dir, 'step_file.txt')
                    with open(step_file, 'w', encoding='utf-8') as stepf:
                        stepf.write(str(global_step) + '\n')

        if args.max_steps > 0 and global_step > args.max_steps:
            break
    
    if args.local_rank in [-1, 0]:
        writer.close()


def evaluate(args, model, tokenizer, eval_dataset,prefix=""):
    """ Evaluate the model """
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_dataloader = DataLoader(eval_dataset, 
                                 sampler = SequentialSampler(eval_dataset), 
                                 batch_size = args.eval_batch_size,
                                 num_workers = 4, 
                                 drop_last = True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)


    model.eval()
    losses, mlm_losses = [], []
    for batch in eval_dataloader:
        mlm_ids =[x.to(args.device) for x in batch] 
        with torch.no_grad():      
            loss, mlm_loss = model(mlm_ids[0])
            losses.append(loss.item())
            if mlm_loss != 0:
                mlm_losses.append(mlm_loss)                      

    result = {
        "loss": round(np.mean(losses),4),
        "mlm_loss": round(np.mean(mlm_losses),4),
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_path", default=None, type=str, required=True,
                        help="The input training data path")
    parser.add_argument("--eval_data_path", default=None, type=str,
                        help="The input evaluating data path")    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--tensorboard_dir", default=None, type=str, required=True,
                        help="The output directory where the tensorboard logs will be written.")

    ## Other parameters
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                        "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")  
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    parser.add_argument("--node_index", type=int, default=-1,
                        help="For distributed training: local_rank")    
    parser.add_argument("--gpu_per_node", type=int, default=-1,
                        help="For distributed training: local_rank")     
    parser.add_argument('--lang', type=str)
    parser.add_argument('--pretraining_type', type=str)
    parser.add_argument('--input_type', type=str)
    parser.add_argument('--tokenizer_type', type=str)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--initializer_range', type=float, default=0.02)

    args = parser.parse_args()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    args.local_rank = local_rank
    
    if args.local_rank in [-1, 0]:  
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir)


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        import datetime
        torch.distributed.init_process_group(backend='nccl',timeout=datetime.timedelta(0,1800000))
        args.local_rank+=args.node_index*args.gpu_per_node
        args.n_gpu = 1

    args.device = device

    # Setup logging
    time = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d-%H%M%S')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename='pre-train_{}.log'.format(time),
                        filemode='w',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier() 

    args.start_step = 0
    model_name_or_path = args.model_name_or_path
    # reload the last checkpoint if exist
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        model_name_or_path = checkpoint_last
        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} step".format(checkpoint_last, args.start_step))  
        
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path if args.tokenizer_name is None else args.tokenizer_name) 

    if args.tokenizer_type == 'starcoder':
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path if args.tokenizer_name is None else args.tokenizer_name, use_fast=False)
        tokenizer.add_special_tokens({'pad_token': '<pad>', 'mask_token': '<mask>', 'sep_token': '</s>', 'cls_token': '<s>'})

    if args.pretraining_type == "continue": 
        config = RobertaConfig.from_pretrained(model_name_or_path if args.config_name is None else args.config_name)
        model = RobertaModel.from_pretrained(model_name_or_path,config=config)  
    elif args.pretraining_type == "from_scratch":
        config = RobertaPreLayerNormConfig(
            vocab_size=len(tokenizer.get_vocab()),
            hidden_size=args.hidden_size,
            max_position_embeddings=514,     
            num_attention_heads=args.num_attention_heads,
            num_hidden_layers=args.num_hidden_layers,
            initializer_range=args.initializer_range,
        )
        
        model = RobertaPreLayerNormModel(config=config)  

    model = Model(model,config,tokenizer,args)

    bfSixteen = MixedPrecision( 
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ) 
    fpSixteen = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    model = FSDP(model, mixed_precision=bfSixteen)   
    
    if args.local_rank == 0:
        torch.distributed.barrier()  

    logger.info("Training/evaluation parameters %s", args)
    
    if args.local_rank == -1:
        local_rank = 0
        world_size = 1
    else:
        local_rank = args.local_rank
        world_size = torch.distributed.get_world_size()
    
    # reload and preprocess data
    train_datasets = [] 
    langs = sorted(args.lang.split(','))
    for lang in langs: 
        total_num = 0
        train_code_dataset = []
        file_path , postfix = args.train_data_path.split(',')
        filename = os.path.join(os.path.join(file_path,lang),postfix)
        logger.info("Load from dataset file at %s", filename)
        for idx,js in enumerate(pickle.load(open(filename,'rb'))):
            if len(js['docstring_tokens']) != 0:
                total_num += 1
                if total_num % world_size == local_rank:
                    train_code_dataset.append(preprocess(js,tokenizer,lang, args))
        train_datasets.append(TextDataset(tokenizer, args,train_code_dataset,"train"))
    
    eval_code_dataset = []
    langs = sorted(args.lang.split(','))
    file_path,postfix = args.eval_data_path.split(',')
    for lang in langs:
        filename = os.path.join(os.path.join(file_path,lang),postfix)
        logger.info("Load from dataset file at %s", filename)
        for idx,js in enumerate(pickle.load(open(filename,'rb'))):
            if len(js['docstring_tokens']) != 0:
                eval_code_dataset.append(preprocess(js,tokenizer,lang, args))
    eval_dataset = TextDataset(tokenizer,args,eval_code_dataset,"eval")   
    
    # Training
    train(args, train_datasets, eval_dataset, model, tokenizer)



if __name__ == "__main__":
    main()