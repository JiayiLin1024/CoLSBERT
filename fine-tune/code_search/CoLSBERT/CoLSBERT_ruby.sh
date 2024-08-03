#!/bin/sh
source activate python39

# Training
lang=ruby
pretrained_model=pre-train/saved_models/CoLSBERT
CUDA_VISIBLE_DEVICES=6,7 python run_CoLSBERT.py \
    --output_dir saved_models/CoLSBERT/$lang \
    --model_name_or_path $pretrained_model  \
    --do_train \
    --train_data_file dataset/CSN/$lang/train.jsonl \
    --eval_data_file dataset/CSN/$lang/valid.jsonl \
    --codebase_file dataset/CSN/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 

# Evaluating
CUDA_VISIBLE_DEVICES=6 python run_CoLSBERT.py \
    --output_dir saved_models/CoLSBERT/$lang \
    --model_name_or_path $pretrained_model  \
    --do_eval \
    --do_test \
    --eval_data_file dataset/CSN/$lang/valid.jsonl \
    --test_data_file dataset/CSN/$lang/test.jsonl \
    --codebase_file dataset/CSN/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456