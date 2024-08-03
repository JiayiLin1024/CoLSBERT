#!/bin/sh
source activate python39

pretrained_model=pre-train/saved_models/CoLSBERT

# Training
CUDA_VISIBLE_DEVICES=6,7 python run_CoLSBERT.py \
    --output_dir saved_models/CoLSBERT \
    --model_name_or_path $pretrained_model \
    --do_train \
    --train_data_file dataset/train.jsonl \
    --eval_data_file dataset/valid.jsonl \
    --test_data_file dataset/test.jsonl \
    --num_train_epochs 2 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 
    
# Evaluating	
CUDA_VISIBLE_DEVICES=6 python run_CoLSBERT.py \
    --output_dir saved_models/CoLSBERT \
    --model_name_or_path $pretrained_model \
    --do_eval \
    --do_test \
    --eval_data_file dataset/valid.jsonl \
    --test_data_file dataset/test.jsonl \
    --num_train_epochs 2 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 