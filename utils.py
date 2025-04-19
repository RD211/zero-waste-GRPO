import argparse
import functools
import logging

from contextlib import contextmanager
from copy import deepcopy

import torch
import deepspeed
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset, Dataset, concatenate_datasets
from configs.cfg import DatasetConfig
from concurrent.futures import ThreadPoolExecutor
    

class GPUFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', rank=0):
        super().__init__(fmt, datefmt, style)
        self.rank = rank
        self.total_mem_mb = 96 * 1024  # 96 GB in MB

    def format(self, record):
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**2  # in MB
            usage_pct = (mem_alloc / self.total_mem_mb) * 100
            gpu_stats = f"GPU{gpu_id}: {mem_alloc:.1f}MB ({usage_pct:.1f}%)"
        else:
            gpu_stats = "GPU: N/A"

        record.gpu = gpu_stats
        record.rank = self.rank
        return super().format(record)

def init_logger(rank):
    logger = logging.getLogger("gpu_logger")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = GPUFormatter(
            fmt="%(asctime)s | rank:%(rank)s | %(name)s | %(levelname)s | %(gpu)s | %(message)s",
            rank=rank
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Prevent logs from propagating to the root logger
    logger.propagate = False

    return logger


PRECISION_STR_TO_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, torch.nn.Embedding)
        )
    return num_params


def get_num_flop_per_token(num_params: int, model_config) -> int:
    l, h, q, t = (
        model_config.n_layers,
        model_config.n_heads,
        model_config.dim // model_config.n_heads,
        model_config.seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token


def build_lr_scheduler(optimizer: torch.optim, warmup_steps: int):

    def linear_warmup_constant(
        warmup_steps: int, current_step: int
    ) -> float:
        """Computes linear warmup followed by linear decay.

        Per LambdaLR requirement, this is accomplished by returning
        a multiplicative factor to adjust the learning rate to
        create the desired schedule.
        """
        if current_step < warmup_steps:
            # linear warmup
            # 0-indexed step, hence + 1 adjustments
            current_step += 1
            curr_adjustment = float(current_step / (warmup_steps + 1))

        else:
            # constant
            curr_adjustment = 1

        return curr_adjustment

    lr_lambda = functools.partial(linear_warmup_constant, warmup_steps)
    return LambdaLR(optimizer, lr_lambda)
    
@torch.no_grad()
def clip_grad_norm_(parameters, grad_max_norm):
  grads = [p.grad for p in parameters if p.grad is not None]
  total_norm = torch.nn.utils.get_total_norm(grads, error_if_nonfinite=True)
  torch.nn.utils.clip_grads_with_norm_(parameters, grad_max_norm, total_norm)
  return total_norm

def load_datasets(cfg: DatasetConfig, seed: int) -> Dataset:
    
    train_datasets = []

    for dataset in cfg.train_datasets:
        dataset = load_dataset(dataset.name_or_path, dataset.subset, split=dataset.split)
        train_datasets.append(dataset)
    
    # We sample based on max_examples and ratios.
    if cfg.max_train_examples is not None:
        ratios = [dataset.ratio for dataset in cfg.train_datasets]
        total_ratio = sum(ratios)
        num_samples = cfg.max_train_examples

    # Sample based on ratios
    if num_samples == -1:
        for i, dataset in enumerate(train_datasets):
            train_datasets[i] = dataset.shuffle(
                seed=seed
            )
    else:
        samples_per_dataset = [int(num_samples * ratio / total_ratio) for ratio in ratios]
        for i, dataset in enumerate(train_datasets):
            train_datasets[i] = dataset.shuffle(
                seed=seed
            ).select(range(samples_per_dataset[i]))
            
    # Concatenate the datasets
    train_datasets = concatenate_datasets(train_datasets)
    train_datasets = train_datasets.shuffle(seed=seed)
    
    return train_datasets


# Fully copied from https://github.com/huggingface/trl/blob/main/trl/models/utils.py#L225
def prepare_deepspeed(model, accelerator):
    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    print(config_kwargs)
    stage = config_kwargs["zero_optimization"]["stage"]

    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes)
            if getattr(model.config, "hidden_sizes", None)
            else getattr(model.config, "hidden_size", None)
        )
        if hidden_size is not None and stage == 3:
            # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
            # @ step 0: expected module 1, but got module 0`
            # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                }
            )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO
    # disabled (stage 0)
    if stage != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model



def gather_incremental_state_dict_to_cpu(accelerator, logger, model, batch_size=60):
    state_dict = {}
    
    def _copy_to_cpu(name_param):
        name, param = name_param
        return name, param.detach().cpu()
    
    
    # List of (name, parameter) pairs
    params = list(model.named_parameters())
    total = len(params)
    logger.info(f"Gathering {total} parameters in batches of {batch_size}")

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = params[start:end]

        # Create a list of just the parameters in this batch.
        batch_params = [p for _, p in batch]
        
        # Use DeepSpeed's context manager to gather the partitioned parameters.
        with deepspeed.zero.GatheredParameters(batch_params, modifier_rank=None):
            logger.info(f"Gathering parameters {start} to {end}")
            if accelerator.is_local_main_process:
                with ThreadPoolExecutor() as executor:
                    # schedule all of the copies in parallel
                    futures = executor.map(_copy_to_cpu, batch)
                    for name, cpu_tensor in futures:
                        state_dict[name] = cpu_tensor
            
        torch.cuda.empty_cache()
        logger.info(f"Processed parameters {start} to {end}")
    
    return state_dict

