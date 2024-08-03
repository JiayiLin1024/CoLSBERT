#!/bin/bash
#SBATCH -J 1.5B				    
#SBATCH -p batch								
#SBATCH -N 1									
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8															            

echo Start slurm job at `date`

source activate python39

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:22371 --nproc_per_node=8 --nnodes=1 run_RobertaPreLayerNorm.py \
    --output_dir ../saved_models/model_size/1.5B \
    --train_data_path ../dataset/CSN,train.pkl \
    --eval_data_path ../dataset/CSN,valid.pkl \
    --tensorboard_dir ../logs/model_size/1.5B \
    --model_name_or_path roberta-large \
    --tokenizer_name ../tokenizer/trained_from_old_one/BPE \
    --lang ruby,python,go,java,php,javascript \
    --pretraining_type from_scratch \
    --input_type source_code \
    --tokenizer_type roberta \
    --num_attention_heads 20 \
    --num_hidden_layers 32 \
    --hidden_size 1920 \
    --block_size 512 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 100 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --initializer_range 0.02 \
    --node_index 0 \
    --gpu_per_node 8 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 1.0 \
    --max_steps 100000 \
    --warmup_steps 1000 \
    --save_steps 50000 \
    --seed 123456

echo End slurm job at `date`