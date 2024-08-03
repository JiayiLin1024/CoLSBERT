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

import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer, AutoTokenizer)

import torch.distributed as dist  
from torch.distributed.fsdp import ( 
    FullyShardedDataParallel as FSDP, 
    MixedPrecision, 
    FullStateDictConfig, 
    StateDictType, 
) 
from torch.utils.data.distributed import DistributedSampler 
from utils import setup_distributed

# remove docstring
import re
from io import StringIO
import tokenize

# os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6'
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url

import re
from io import StringIO
import  tokenize

def remove_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remain comments:
            if token_type == tokenize.COMMENT:
                out += token_string
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)

def convert_examples_to_features(js,tokenizer,args):
    #code
    if args.input_type == 'join_tokens':
        code=' '.join(js['code_tokens'])
    elif args.input_type == 'source_code':
        code=remove_docstrings(js['code'], args.language)
    
    # code=' '.join(js['code_tokens'])
    code_tokens=tokenizer.tokenize(code)[:args.code_length-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length
    
    nl=' '.join(js['docstring_tokens'])
    nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        if args.local_rank not in [-1, 0] :
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        self.examples = []
        data=[]

        # Load data features from cache or dataset file
        args.data_dir = '/'.join(file_path.split('/')[:-1])
        cached_features_file = os.path.join(args.data_dir, 'cached_span-{}_{}_{}_{}'.format(
            args.ttype,
            args.input_type,
            'for',
            'finetune'))
        
        try:
            logger.info(f"Loading features from cached file {cached_features_file}")
            # load finetune dataset from cache
            self.examples = torch.load(cached_features_file)
        
        except:
            logger.info("Creating features from dataset file at %s", args.data_dir)

            if os.path.isfile(file_path) is False:
                raise ValueError(f"Input file path {file_path} not found")
        
            with open(file_path) as f:
                for line in f:
                    line=line.strip()
                    js=json.loads(line)
                    data.append(js)

            for js in data:
                self.examples.append(convert_examples_to_features(js,tokenizer,args))
            
            # save to cache
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.examples, cached_features_file)
            
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))  

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache                           
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def aggregate(code_vec, nl_vec):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # We gather tensors from all gpus to get more negatives to contrast with.
    gathered_code_vec = [
        torch.zeros_like(code_vec) for _ in range(world_size)
    ]
    gathered_nl_vec = [
        torch.zeros_like(nl_vec) for _ in range(world_size)
    ]
    dist.all_gather(gathered_code_vec, code_vec)
    dist.all_gather(gathered_nl_vec, nl_vec)

    all_code_vec = torch.cat(
        [code_vec]
        + gathered_code_vec[:rank]
        + gathered_code_vec[rank + 1 :]
    )
    all_nl_vec = torch.cat(
        [nl_vec]
        + gathered_nl_vec[:rank]
        + gathered_nl_vec[rank + 1 :]
    )

    return all_code_vec, all_nl_vec


def train(args, model, tokenizer, local_rank):
    """ Train the model """
    #get training dataset
    args.ttype = 'train'
    train_dataset=TextDataset(tokenizer, args, args.train_data_file)
    # ddp
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,best_mrr=0,0,0 
    for idx in range(args.num_train_epochs): 
        # ddp
        train_dataloader.sampler.set_epoch(idx)
        for step,batch in enumerate(train_dataloader):
            # get inputs
            code_inputs = batch[0].to(local_rank)    
            nl_inputs = batch[1].to(local_rank)
            # get code and nl vectors
            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)
            
            # calculate scores and loss
            code_vec, nl_vec = aggregate(code_vec, nl_vec)
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)

            scores = scores / args.temp

            loss_fct = CrossEntropyLoss().to(local_rank)
            loss = loss_fct(scores, torch.arange(code_vec.size(0), device=scores.device))
            
            # report loss
            tr_loss += loss.item()
            tr_num+=1
            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss=0
                tr_num=0
            
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        # evaluate    
        results = evaluate(args, model, tokenizer,args.eval_data_file, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
        
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True) 
        with FSDP.state_dict_type( 
            model, StateDictType.FULL_STATE_DICT, save_policy 
        ): 
            model_best_state = model.state_dict() 
        
        # save best model on local_rank=0
        if results['eval_mrr']>best_mrr and local_rank in [-1, 0]:
            best_mrr=results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)         
                 
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_best_state, output_dir) 
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer,file_name,eval_when_training=False):
    if 'valid' in file_name:
        args.ttype = 'valid'
    elif 'test' in file_name:
        args.ttype = 'test'
    query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)
    
    args.ttype = 'codebase'
    code_dataset = TextDataset(tokenizer, args, args.codebase_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)    

    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model.eval()
    code_vecs=[] 
    nl_vecs=[]
    for batch in query_dataloader:  
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().to(torch.float32).numpy())   

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        with torch.no_grad():
            code_vec= model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().to(torch.float32).numpy())    
    model.train()    
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)

    scores=np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls=[]
    code_urls=[]
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)
        
    ranks=[]
    for url, sort_id in zip(nl_urls,sort_ids):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if code_urls[idx]==url:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    
    result = {
        "eval_mrr":float(np.mean(ranks))
    }

    return result



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # language
    parser.add_argument('--language', default='python', type=str, 
                        help='language')
    # temperature
    parser.add_argument("--temp", default=0.05, type=float,
                        help="The temperature hyperparameter in cosine similarity")
    
    parser.add_argument("--port", default=None, type=str,
                        help="master node port.")
    
    parser.add_argument('--input_type', type=str)
    

    # print arguments
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    args.local_rank = local_rank

    if args.local_rank in [-1, 0]:  
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # ddp
    setup_distributed(args)
    
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    logger.info("device: %s, n_gpu: %s", args.device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path)  
      
    model=Model(model)
    logger.info("Training/evaluation parameters %s", args)
    # fsdp
    model = model.to(local_rank)
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
    
    # Training
    if args.do_train:
        train(args, model, tokenizer, local_rank)

    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        model = RobertaModel.from_pretrained(args.model_name_or_path)    
        model= Model(model)
        model = model.to(local_rank)
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device),strict=False)      
        model = FSDP(model, mixed_precision=bfSixteen)
        # model.to(args.device)
        result=evaluate(args, model, tokenizer,args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        model = RobertaModel.from_pretrained(args.model_name_or_path)    
        model= Model(model)
        model = model.to(local_rank)
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device),strict=False)      
        # model.to(args.device)
        model = FSDP(model, mixed_precision=bfSixteen)
        result=evaluate(args, model, tokenizer,args.test_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

    return results


if __name__ == "__main__":
    main()