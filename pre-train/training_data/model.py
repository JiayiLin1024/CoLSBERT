import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import random

def mask_tokens(inputs,tokenizer,args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15).to(inputs.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(inputs.device), value=0.0)
    padding_mask = labels.lt(119)
    probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(inputs.device) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.register_buffer(
        "bias", torch.tril(torch.ones((args.block_size, args.block_size), dtype=torch.uint8)).view(1, args.block_size, args.block_size)
        )


    def forward(self, mlm_ids): 

        mlm_loss = 0
        masked_source_ids,masked_lm_labels=mask_tokens(mlm_ids,self.tokenizer,self.args)
        
        attention_mask = masked_source_ids.ne(1)[:,:,None]*masked_source_ids.ne(1)[:,None,:]
        encoder_outputs = self.encoder(masked_source_ids,attention_mask=attention_mask).last_hidden_state
        
        encoder_outputs = encoder_outputs.view(-1,encoder_outputs.size(-1))[masked_lm_labels.view(-1).ne(-100)]
        prediction_scores = self.lm_head(encoder_outputs)
        
        masked_lm_labels = masked_lm_labels.view(-1)[masked_lm_labels.view(-1).ne(-100)]

        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(prediction_scores, masked_lm_labels)  
        mlm_loss = lm_loss.item()
               
            
        return lm_loss, mlm_loss
