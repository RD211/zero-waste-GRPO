import transformer_engine.pytorch as te
import torch.nn as nn
import torch
import deepspeed


def fp8ify_module(module: nn.Module, skip_names: tuple[str] = ("lm_head",)):
    for name, child in module.named_children():
        if any(sn in name for sn in skip_names):
            continue

        if isinstance(child, nn.Linear):

            te_lin = te.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                params_dtype=child.weight.dtype,
            )
            with deepspeed.zero.GatheredParameters(
                [child.weight, child.bias], modifier_rank=None
            ):
                with torch.no_grad():
                    print(f"Shape of linear: {child.weight.shape}")
                    te_lin.weight.copy_(child.weight)
                    if child.bias is not None:
                        te_lin.bias.copy_(child.bias)
            setattr(module, name, te_lin)
        else:
            fp8ify_module(child, skip_names)
