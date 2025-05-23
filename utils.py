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
import pynvml
pynvml.nvmlInit()

import logging
import torch
import psutil

class GPUFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', rank=0):
        super().__init__(fmt, datefmt, style)
        self.rank = int(str(rank))
        pynvml.nvmlInit()

    def format(self, record):
        if torch.cuda.is_available():
            gpu_id = self.rank % torch.cuda.device_count()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb  = info.used  / 1024**2
            total_mb = info.total / 1024**2
            pct      = used_mb / total_mb * 100
            record.gpu = f"GPU{gpu_id}: {used_mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)"
        else:
            record.gpu = "GPU: N/A"

        vm = psutil.virtual_memory()
        used_gb  = vm.used  / 1024**3
        total_gb = vm.total / 1024**3
        record.ram = f"RAM: {used_gb:.1f}/{total_gb:.1f} GB ({vm.percent:.1f}%)"

        record.rank = self.rank

        return super().format(record)

def init_logger(rank=0):
    logger = logging.getLogger("gpu_logger")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fmt = (
            "%(asctime)s | rank:%(rank)s | %(name)s | %(levelname)s | "
            "%(gpu)s | %(ram)s | %(message)s"
        )
        formatter = GPUFormatter(fmt=fmt, rank=rank)
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
        filter_fn = eval("lambda example: "+dataset.filter)
        dataset = load_dataset(dataset.name_or_path, dataset.subset, split=dataset.split)

        dataset = dataset.filter(
            filter_fn
        )
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


def gather_incremental_state_dict_to_cpu(accelerator, logger, model, batch_size=120):
    state_dict = {}
    
    def _copy_to_cpu(name_param):
        name, param = name_param
        return name, param.detach().cpu()
    
    params = list(model.named_parameters())
    total = len(params)
    logger.info(f"Gathering {total} parameters in batches of {batch_size}")

    #TODO: Could technically be improved by for each GPU gathering a part and then putting on shared memory and then
    # gathering the shared memory on the CPU.
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

