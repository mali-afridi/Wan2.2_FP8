import transformer_engine.pytorch as te
import torch
import torch.nn as nn
import logging
from transformer_engine.common.recipe import (Format,
 DelayedScaling,
    Float8BlockScaling,
    Float8CurrentScaling,)
FP8_RECIPE_REGISTRY = {
    "Float8CurrentScaling": lambda: Float8CurrentScaling(fp8_format=Format.E4M3),
    "Float8BlockScaling": lambda: Float8BlockScaling(fp8_format=Format.E4M3),
    "DelayedScaling": lambda: DelayedScaling(
        fp8_format=Format.E4M3,
        amax_history_len=16,
        amax_compute_algo="max",
    ),
}

class CachedTELinear(te.Linear):
    """
    A Transformer Engine Linear layer that automatically caches FP8 weights 
    after the first forward pass to speed up inference.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_run_once = False

    def forward(self, x, *args, **kwargs):
        # The first time this runs, we let TE cast the weights (is_first_microbatch=True).
        # Every subsequent time, we tell TE to reuse the cached FP8 weights (is_first_microbatch=False).
        
        # Note: We must allow re-casting if the batch size changes or during warmup, 
        # but for fixed-batch inference, this is the optimal path.
        is_first = not self.has_run_once
        
        out = super().forward(x, *args, is_first_microbatch=is_first, **kwargs)
        
        self.has_run_once = True
        return out
def replace_with_te_layers(model, ignore_list=None):
    """
    Replaces torch.nn.Linear with CachedTELinear
    and torch.nn.LayerNorm with te.LayerNorm.

    Returns:
        dict with counts of replaced layers
    """
    if ignore_list is None:
        ignore_list = set()
    else:
        ignore_list = set(ignore_list)

    counts = {
        "linear": 0,
        "layernorm": 0,
    }

    def _replace(module):
        for name, child in module.named_children():
            if name in ignore_list:
                continue

            # ---- Linear ----
            if isinstance(child, nn.Linear):
                has_bias = child.bias is not None

                te_linear = CachedTELinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=has_bias,
                    params_dtype=torch.bfloat16,
                ).to(child.weight.device)

                te_linear.weight.data.copy_(child.weight.data)
                if has_bias:
                    te_linear.bias.data.copy_(child.bias.data)

                setattr(module, name, te_linear)
                counts["linear"] += 1

            # ---- LayerNorm ----
            elif isinstance(child, nn.LayerNorm):
                te_ln = te.LayerNorm(
                    child.normalized_shape[0],
                    eps=child.eps,
                    params_dtype=torch.bfloat16,
                )
                try:
                    device = (
                        child.weight.device
                        if child.weight is not None
                        else next(module.parameters()).device
                    )
                except StopIteration:
                    device = torch.device(f"cuda:{torch.distributed.get_rank()}")
                te_ln.to(device)

                if child.weight is not None:
                    te_ln.weight.data.copy_(child.weight.data)
                else:
                    te_ln.weight.data.fill_(1.0)

                if child.bias is not None:
                    te_ln.bias.data.copy_(child.bias.data)
                else:
                    te_ln.bias.data.zero_()

                setattr(module, name, te_ln)
                counts["layernorm"] += 1

            else:
                _replace(child)

    _replace(model)
    return counts

def replace_with_te_layers_modulelist(model, ignore_list=None):
    if ignore_list is None:
        ignore_list = set()
    else:
        ignore_list = set(ignore_list)

    counts = {"linear": 0, "layernorm": 0}

    def _replace(module):
        # iterate through child modules
        for name, child in module.named_children():
            if name in ignore_list:
                continue

            # --- SPECIAL HANDLING FOR MODULELIST ---
            # If the child is the 'blocks' ModuleList, handle it manually
            if isinstance(child, torch.nn.ModuleList):
                total_elements = len(child)
                for i, sub_module in enumerate(child):
                    # Skip the last block (index 39 for 40 blocks)
                    if i == total_elements - 1:
                        logging.info(f"Skipping final block index: {i}")
                        sub_module.norm3= sub_module.norm3.to(torch.float32)
                        continue
                    _replace(sub_module)
                continue # Skip the default recursion for this child

            # ---- Linear ----
            if isinstance(child, nn.Linear):
                has_bias = child.bias is not None
                te_linear = CachedTELinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=has_bias,
                    params_dtype=torch.bfloat16,
                ).to(child.weight.device)

                te_linear.weight.data.copy_(child.weight.data)
                if has_bias:
                    te_linear.bias.data.copy_(child.bias.data)

                setattr(module, name, te_linear)
                counts["linear"] += 1

            # ---- LayerNorm ----
            elif isinstance(child, nn.LayerNorm):
                te_ln = te.LayerNorm(
                    child.normalized_shape[0],
                    eps=child.eps,
                    params_dtype=torch.bfloat16,
                )
                
                device = child.weight.device if child.weight is not None else next(module.parameters()).device
                te_ln.to(device)

                if child.weight is not None:
                    te_ln.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    te_ln.bias.data.copy_(child.bias.data)

                setattr(module, name, te_ln)
                counts["layernorm"] += 1

            else:
                # Recurse for standard modules (WanAttentionBlock, Sequential, etc.)
                _replace(child)

    _replace(model)
    return counts