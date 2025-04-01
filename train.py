"""
torchrun --nproc_per_node 1 train.py
torchrun --nproc_per_node 2 train.py --tp_size 2 --run_name process_group_manager --use_wandb
"""

import os
import wandb
import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
from torch.optim import AdamW
from transformers import AutoConfig

from picotron.model import Llama
import picotron.process_group_manager as pgm
from picotron.process_group_manager import setup_process_group_manager

from picotron.utils import print, set_all_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for LLaMa model")

    # Env args
    parser.add_argument("--omp_num_threads", type=str, default="1")
    parser.add_argument("--tokenizers_parallelism", type=str, default="false")

    # Model args
    parser.add_argument("--model_name", type=str, default="/mnt/data/Models/SmolLM/SmolLM-360M-Instruct/")
    parser.add_argument("--num_hidden_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--num_key_value_heads", type=int, default=4)

    # Training args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=1)

    # Distributed training args
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--dp_size", type=int, default=1, help="Data Parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline Parallel size")
    parser.add_argument("--pp_engine", type=str, default="afab", choices=["1f1b", "afab"])

    # Logging args
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    # Set Environment variables 
    os.environ["OMP_NUM_THREADS"] = args.omp_num_threads
    os.environ["TOKENIZES_PARALLELISM"] = args.tokenizers_parallelism
    os.environ["DEVICE"] = "cude"

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl"
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    dist.init_process_group(rank=global_rank, world_size=world_size, backend=backend, init_method=f"env://", timeout=datetime.timedelta(minutes=2))
    setup_process_group_manager(dp_size=args.dp_size, pp_size=args.pp_size, tp_size=args.tp_size)
    is_wandb_rank = pgm.process_group_manager.tp_rank == 0 and pgm.process_group_manager.pp_rank == 0 and pgm.process_group_manager.dp_rank == 0

    set_all_seed(args.seed)
    if is_wandb_rank and args.use_wandb:
        wandb.init(
            project="picotron_tutorial",
            name=f"{args.run_name}_{pgm.process_group_manager}",
            config={
                "tensor_parallel_size": pgm.process_group_manager.tp_world_size,
                "pipeline_parallel_size": pgm.process_group_manager.pp_world_size,
                "data_parallel_size": pgm.process_group_manager.dp_world_size,
                "model": args.model_name,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
        )

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_hidden_layers = args.num_hidden_layers
    model_config.num_attention_heads = args.num_attention_heads
    model_config.num_key_value_heads = args.num_key_value_heads
    model_config.max_position_embeddings = args.seq_len

    # print(model_config, is_print_rank=(global_rank == 0))

    model = Llama(config=model_config)
    model.to(dtype).to(device)
    model.train()

    dist.barrier()

    # print(model, is_print_rank=(global_rank == 0))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    dist.barrier()

    input_ids = torch.randint(0, model_config.vocab_size, (args.micro_batch_size, args.seq_len), device=device)
    target_ids = torch.randint(0, model_config.vocab_size, (args.micro_batch_size, args.seq_len), device=device)

    optimizer.zero_grad()

    outputs = model(input_ids=input_ids)

    target_ids = target_ids.reshape(-1)
    outputs = outputs.reshape(-1, model_config.vocab_size)

    loss = F.cross_entropy(outputs, target_ids)

    loss.backward()

    optimizer.step()

    # print(f"Loss: {loss.item():.4f}", is_print_rank=(global_rank == 0))
    print(f"[rank {pgm.process_group_manager.global_rank}], Loss: {loss:.4f}")

    if is_wandb_rank and args.use_wandb:
        wandb.log({"loss": loss.item()})

    if is_wandb_rank and args.use_wandb:
        wandb.finish()

    dist.destroy_process_group()