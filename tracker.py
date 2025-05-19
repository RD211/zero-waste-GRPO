import functools
import time, torch
from typing import Callable
from contextlib import contextmanager

@contextmanager
def gpu_tracker(name: str, step: int, log_to_wandb: bool, accel, extra: dict | None = None):
    device = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device)
    start_mem  = torch.cuda.memory_allocated(device)
    start_time = time.perf_counter()
    yield
    elapsed    = time.perf_counter() - start_time
    peak_mem   = torch.cuda.max_memory_allocated(device)

    if log_to_wandb and accel.is_main_process:
        import wandb
        wandb_log = {
            f"{name}/time_s"      : elapsed,
            f"{name}/mem_peak_mb" : peak_mem / 1e6,
            f"{name}/mem_start_mb": start_mem / 1e6,
            f"{name}/mem_diff_mb" : (peak_mem - start_mem) / 1e6,
            "step"                : step,
        }
        if extra: wandb_log.update(extra)
        wandb.log(wandb_log, step=step)

def gpu_profiler(name: str | None = None):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            tag   = name or func.__name__
            step  = getattr(self, "_current_step", 0)
            do_wb = getattr(self.cfg.logging, "wandb", False)
            accel = getattr(self, "accelerator", None)

            with gpu_tracker(tag, step, do_wb, accel):
                return func(self, *args, **kwargs)
        return wrapper
    return decorator