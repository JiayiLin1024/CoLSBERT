import os
import subprocess
import torch
# ddp
import torch.distributed as dist


def setup_distributed(args):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """

    if "SLURM_JOB_ID" in os.environ:  # slurm
        world_size = int(os.environ['SLURM_NTASKS'])
        # world_size = 4
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        node_list = os.environ["SLURM_JOB_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        port = args.port

        # set environ for ddp
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
    else:   # torch.distributed
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    args.local_rank = local_rank