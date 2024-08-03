import os
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import pickle
from pandas.io.json import json_normalize
from transformers import AutoTokenizer, RobertaTokenizer
from itertools import chain
from transformers import (Trainer,
                          RobertaConfig,
                          TrainingArguments,
                          RobertaForMaskedLM,
                          RobertaPreLayerNormConfig,
                          RobertaPreLayerNormForMaskedLM,
                          DataCollatorForLanguageModeling)

from torch.utils.data import IterableDataset
import torch

# all_shuffle
train_cache_dir = "../dataset/StarCoder_data/parquet_saved_tokenized_all"
valid_cache_dir = "../dataset/CSN/data_cache"

max_seq_length = 512
tokenizer_path = "../tokenizer/trained_from_old_one/BPE"
saved_model = "../saved_models/CoLSBERT"
training_log = "../logs/CoLSBERT"


tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
train_files = []
for root, dirs, files in os.walk(train_cache_dir):
    for file in files:
        file_path = os.path.join(root, file)
        print(file_path)
        train_files.append(file_path)

# Stream from local files: https://huggingface.co/docs/datasets/v2.15.0/en/about_mapstyle_vs_iterable#downloading-and-streaming
# Stream from load_from_disk: https://github.com/huggingface/datasets/issues/5838#issuecomment-1543712801
train_tokenized_datasets = load_dataset("parquet", data_files={'train': train_files}, streaming=True)
# for example in train_tokenized_datasets["train"]: # debug
#     print(example)
#     break

valid_tokenized_datasets = load_from_disk(valid_cache_dir)

config = RobertaPreLayerNormConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=1920,
        max_position_embeddings=514,     
        num_attention_heads=20,
        num_hidden_layers=32,
        type_vocab_size=1,
        # initializer_range=0.0185
)

model = RobertaPreLayerNormForMaskedLM(config=config)

data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        mlm=True,
)

# solve bug "Streaming dataset into Trainer: does not implement __len__, max_steps has to be specified": https://discuss.huggingface.co/t/streaming-dataset-into-trainer-does-not-implement-len-max-steps-has-to-be-specified/32893
training_args = TrainingArguments(
    output_dir=saved_model,
    overwrite_output_dir=True,
    # num_train_epochs=10,
    per_device_train_batch_size=8,
    do_eval=True,
    logging_dir=training_log,
    logging_steps=10,
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    warmup_ratio=0.01,
    # warmup_steps=10_000,
    lr_scheduler_type="linear",
    learning_rate=2e-4,
    adam_epsilon=1e-6,
    max_grad_norm=1.0, 
    max_steps=4000000, # must set max_steps for streaming
    save_steps=10_000,
    bf16=True,
    fsdp=True,
)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_datasets["train"], #.shuffle(42).select(range(256)),
    eval_dataset=valid_tokenized_datasets["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(saved_model)
tokenizer.save_pretrained(saved_model)
