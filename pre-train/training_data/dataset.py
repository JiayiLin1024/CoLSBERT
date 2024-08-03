import random
import torch
from torch.utils.data import Dataset


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, dataset, mode):
        self.args = args
        self.tokenizer = tokenizer
        self.dataset = dataset
        
        if mode != "train":
            random.shuffle(self.dataset)
            self.dataset = self.dataset[:10000]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item): 
        js = self.dataset[item]

        doc_tokens = js["docstring_split_tokenize"].copy()
        
        code_tokens = js["function_tokens_tokenize"].copy()
        
        _truncate_seq_pair(doc_tokens, code_tokens, self.args.block_size-3)
        text_tokens = [self.tokenizer.cls_token] + doc_tokens + [self.tokenizer.sep_token] + code_tokens + [self.tokenizer.sep_token]
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        mlm_ids = text_ids + [self.tokenizer.pad_token_id] * (self.args.block_size-len(text_ids)) 

        
        return (torch.tensor(mlm_ids),)



            

    