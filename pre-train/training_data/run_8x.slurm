#!/bin/bash
#SBATCH -J eight_times				    
#SBATCH -p batch								
#SBATCH -N 1									
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4															            

echo Start slurm job at `date`

source activate python39

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:22372 --nproc_per_node=4 --nnodes=1 run.py \
    --output_dir ../saved_models/dataset_size/eight_times \
    --train_data_path ../dataset/StarCoder_data/processed_with_filter/eight_times_CSN,train.pkl \
    --eval_data_path ../dataset/CSN,valid.pkl \
    --tensorboard_dir ../logs/dataset_size/eight_times \
    --model_name_or_path roberta-base \
    --tokenizer_name ../tokenizer/trained_from_old_one/BPE \
    --lang ruby,python,go,java,php,javascript \
    --pretraining_type from_scratch \
    --input_type source_code \
    --tokenizer_type roberta \
    --block_size 512 \
    --per_gpu_train_batch_size 100 \
    --per_gpu_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --node_index 0 \
    --gpu_per_node 4 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 1.0 \
    --max_steps 100000 \
    --warmup_steps 10000 \
    --save_steps 5000 \
    --seed 123456

echo End slurm job at `date`