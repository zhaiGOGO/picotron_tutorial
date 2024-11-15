"""
torchrun --nproc_per_node 2 train.py --dp_size 2 --use_wandb
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

from model import Llama
import process_group_manager as pgm
from process_group_manager import setup_process_group_manager
from utils import set_all_seed, print

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Training script for LLaMA model")
    
    # Environment arguments
    parser.add_argument("--omp_num_threads", type=str, default="1")
    parser.add_argument("--tokenizers_parallelism", type=str, default="false")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    
    # Training arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    
    # Distributed training arguments
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--cp_size", type=int, default=1, help="Context Parallel size")
    parser.add_argument("--dp_size", type=int, default=1, help="Data Parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline Parallel size")
    parser.add_argument("--pp_engine", type=str, default="afab", choices=["1f1b", "afab"])

    # Logging arguments
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = args.omp_num_threads
    os.environ["TOKENIZERS_PARALLELISM"] = args.tokenizers_parallelism
    os.environ["DEVICE"] = "cpu"
    
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "gloo"
    device = torch.device("cpu")
    dtype = torch.float32

    dist.init_process_group(rank=global_rank, world_size=world_size, backend=backend, init_method=f"env://", timeout=datetime.timedelta(minutes=2))
    setup_process_group_manager(tp_size=args.tp_size, cp_size=args.cp_size, pp_size=args.pp_size, dp_size=args.dp_size)

    assert world_size == args.tp_size * args.pp_size * args.dp_size * args.cp_size, "world_size must be equal to tp_size * pp_size * dp_size * cp_size"

    is_wandb_rank = pgm.process_group_manager.tp_rank == 0 and pgm.process_group_manager.dp_rank == 0 and pgm.process_group_manager.cp_rank == 0 and pgm.process_group_manager.pp_is_last_stage
    set_all_seed(args.seed)

    if is_wandb_rank and args.use_wandb:
        wandb.init(
            project="picotron",
            name=f"{args.run_name}_{pgm.process_group_manager}",
            config={
                "tensor_parallel_size": pgm.process_group_manager.tp_size,
                "context_parallel_size": pgm.process_group_manager.cp_size,
                "pipeline_parallel_size": pgm.process_group_manager.pp_size,
                "data_parallel_size": pgm.process_group_manager.dp_size,
                "model": args.model_name,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
        )

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_hidden_layers = args.num_hidden_layers
    model_config.num_attention_heads = args.num_attention_heads
    model_config.num_key_value_heads = args.num_key_value_heads
    model_config.max_position_embeddings = args.seq_length

    model = Llama(config=model_config)
    model.to(dtype).to(device)            
    model.train()

    dist.barrier()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    dist.barrier()
    
    # Create dummy data
    input_ids = torch.randint(0, model_config.vocab_size, (args.micro_batch_size, args.seq_length), device=device)
    target_ids = torch.randint(0, model_config.vocab_size, (args.micro_batch_size, args.seq_length), device=device)

    # Training step
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(input_ids=input_ids)
    
    # Compute loss
    target_ids = target_ids.reshape(-1)
    outputs = outputs.view(-1, model_config.vocab_size)
    loss = F.cross_entropy(outputs, target_ids)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()

    print(f"[rank {pgm.process_group_manager.global_rank}], Loss: {loss:.4f}")
       
    if is_wandb_rank and args.use_wandb:
        wandb.log({"loss": loss.item()})

    if is_wandb_rank and args.use_wandb:
        wandb.finish()

    dist.destroy_process_group()