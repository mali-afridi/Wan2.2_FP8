# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy #fsdp2
from torch.distributed.tensor import DeviceMesh
from peft.tuners.lora.layer import Linear as LoRALinear
import logging
def shard_model(
    model,
    device_id,
    use_lora=False,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,

):
    logging.info(f"Sharding model with FSDP2...")

    # 1. Setup Device Mesh
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        device_mesh = DeviceMesh(
            "cuda",
            mesh=torch.arange(world_size).reshape(-1),
        )
    else:
        device_mesh = DeviceMesh("cuda", mesh=torch.tensor([device_id]))

    # 2. Define Mixed Precision Policy (Crucial for memory!)
    # FSDP2 uses 'MixedPrecisionPolicy', not the old 'MixedPrecision' class
    fp8_enabled = False
    base_layer_name = None
    has_lora = (hasattr(model.blocks[0], 'cross_attn') and (isinstance(model.blocks[0].cross_attn.q, LoRALinear)))
    if (hasattr(model.blocks[0], 'cross_attn')): #not t5
        #detected low_noise or high_noise model

        if has_lora:
            base_layer_name = type(model.blocks[0].cross_attn.q.base_layer).__name__
        else:
            base_layer_name = type(model.blocks[0].cross_attn.q).__name__
    
    
    
        if (base_layer_name == "CachedTELinear"):
            fp8_enabled = True
            logging.info(f"FP8 layers detected, cast_forward_inputs is True")
    
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, 
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=fp8_enabled
    )
    # 3. Apply sharding to INTERNAL blocks (The "Auto Wrap" replacement)
    # This assumes your transformer blocks are iterable, e.g., model.blocks or model.layers
    for layer_id, block in enumerate(model.blocks):
        fully_shard(
            block,
            mesh=device_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True
    )

    # 4. Apply sharding to the ROOT model
    # This handles embeddings, final layer norms, and heads usually found outside the blocks
    fully_shard(
        model,
        mesh=device_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True
    )

    return model

def free_model(model):

    del model
    gc.collect()
    torch.cuda.empty_cache()