#!/bin/bash
#SBATCH -J pre-test			    
#SBATCH -p batch								
#SBATCH -N 8									
#SBATCH --cpus-per-task=220
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks=64			
#SBATCH --gres=gpu:8									            

echo Start slurm job at `date`

### init virtual environment if needed
source activate python39

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

### the command to run

srun torchrun --rdzv_id 3456 --rdzv_backend=c10d --rdzv_endpoint=$head_node_ip:22371 --nproc_per_node=8 --nnodes=8 train_1B_streaming.py
    
echo End slurm job at `date`