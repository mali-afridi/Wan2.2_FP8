"""
Enhanced MagCache wrapper that works with FSDP and Sequence Parallel
This implementation preserves the residual caching mechanism even when 
FSDP or SP modifies the forward method.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Callable, Any, Dict, Tuple, Union
from functools import wraps
import logging
import inspect

logger = logging.getLogger(__name__)

MAGCACHE_RATIOS = [0.99191, 0.99144, 0.99356, 0.99337, 0.99326, 0.99285, 0.99251, 
                      0.99264, 0.99393, 0.99366, 0.9943, 0.9943, 0.99276, 0.99288, 
                      0.99389, 0.99393, 0.99274, 0.99289, 0.99316, 0.9931, 0.99379, 
                      0.99377, 0.99268, 0.99271, 0.99222, 0.99227, 0.99175, 0.9916, 
                      0.91076, 0.91046, 0.98931, 0.98933, 0.99087, 0.99088, 0.98852, 
                      0.98855, 0.98895, 0.98896, 0.98806, 0.98808, 0.9871, 0.98711, 
                      0.98613, 0.98618, 0.98434, 0.98435, 0.983, 0.98307, 0.98185, 
                      0.98187, 0.98131, 0.98131, 0.9783, 0.97835, 0.97619, 0.9762, 
                      0.97264, 0.9727, 0.97088, 0.97098, 0.96568, 0.9658, 0.96045, 
                      0.96055, 0.95322, 0.95335, 0.94579, 0.94594, 0.93297, 0.93311, 
                      0.91699, 0.9172, 0.89174, 0.89202, 0.8541, 0.85446, 0.79823, 0.79902]


def get_timesteps(
    num_inference_steps: int = 40,
    num_train_timesteps: int = 1000,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
    alphas_cumprod: Optional[np.ndarray] = None,
    use_dynamic_shifting: bool = False,
    final_sigmas_type: str = "sigma_min",  # or "zero"
    shift: float = 1.0,
    mu: Optional[float] = None,
    sigmas: Optional[List[float]] = None,
    device: Union[str, torch.device, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified version of the scheduler's `set_timesteps` function.

    Returns:
        sigmas (torch.FloatTensor): Noise levels, shape [num_inference_steps + 1]
        timesteps (torch.LongTensor): Discrete timesteps, shape [num_inference_steps]
    """
    if use_dynamic_shifting and mu is None:
        raise ValueError("`mu` must be provided when `use_dynamic_shifting` is True")

    if alphas_cumprod is None:
        # Default DDPM-style alphas_cumprod
        betas = np.linspace(1e-4, 0.02, num_train_timesteps, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

    if sigmas is None:
        sigmas = np.linspace(sigma_max, sigma_min, num_inference_steps + 1)[
            :-1
        ]  # shape: [num_inference_steps]
    if use_dynamic_shifting:
        # Simple example dynamic shift function
        sigmas = sigmas * np.exp(mu * (1 - sigmas))
    else:
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    if final_sigmas_type == "sigma_min":
        sigma_last = ((1 - alphas_cumprod[0]) / alphas_cumprod[0]) ** 0.5
    elif final_sigmas_type == "zero":
        sigma_last = 0.0
    else:
        raise ValueError(f"Invalid `final_sigmas_type`: {final_sigmas_type}")

    # Convert sigmas to tensor with final appended
    sigmas = np.concatenate([sigmas, [sigma_last]]).astype(
        np.float32
    )  # [num_inference_steps + 1]
    timesteps = sigmas[:-1] * num_train_timesteps  # [num_inference_steps]

    sigmas_tensor = torch.from_numpy(sigmas).to("cpu")
    timesteps_tensor = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

    return timesteps_tensor

def nearest_interp(src_array, target_length):
    """Nearest neighbor interpolation for magnitude ratios"""
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])
    
    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


class MagCacheManager:
    """
    MagCache manager that intercepts model blocks execution
    to implement residual caching without modifying forward methods
    """
    
    def __init__(
        self,
        mag_ratios: np.ndarray,
        num_steps: int,
        split_step: Optional[int] = None,
        mode: str = "t2v",
        magcache_thresh: float = 0.15,
        K: int = 10,
        retention_ratio: float = 0.5
    ):
        # Configuration
        self.mag_ratios = np.array([1.0] * 2 + list(mag_ratios))
        self.num_steps = num_steps * 2  # Account for conditional/unconditional
        self.split_step = split_step * 2 if split_step else None
        self.mode = mode
        self.magcache_thresh = magcache_thresh
        self.K = K
        self.retention_ratio = retention_ratio
        
        # State tracking
        self.cnt = 0
        self.accumulated_ratio = [1.0, 1.0]
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
        self.residual_cache = [None, None]
        
        # Interpolate mag_ratios if needed
        if len(self.mag_ratios)  != self.num_steps:
            mag_ratio_con = nearest_interp(np.array(self.mag_ratios[0::2]), num_steps)
            mag_ratio_ucon = nearest_interp(np.array(self.mag_ratios[1::2]), num_steps)
            self.mag_ratios = np.concatenate([
                mag_ratio_con.reshape(-1, 1), 
                mag_ratio_ucon.reshape(-1, 1)
            ], axis=1).reshape(-1).tolist()
        # Model references - now supports multiple models
        self.models = []  # List of models that share this manager
        self.original_forwards = {}  # model_id -> original forward
        self.original_blocks_forwards = {}  # model_id -> list of original block forwards
        self.is_active = False
        
    def attach(self, model: torch.nn.Module, model_name: str = None):
        """Attach MagCache to a model by wrapping block forward methods
        
        Args:
            model: Model to attach MagCache to
            model_name: Optional name for the model (for logging)
        """
        model_id = id(model)
        if model_id in self.original_forwards:
            logger.warning(f"MagCache already attached to this model")
            return
        
        self.models.append(model)
        self.original_blocks_forwards[model_id] = []
        
        # Wrap each block's forward method
        for i, block in enumerate(model.blocks):
            self.original_blocks_forwards[model_id].append(block.forward)
            block.forward = self._create_wrapped_block_forward(block, i)
        
        # Store and wrap the main model forward
        self.original_forwards[model_id] = model.forward
        model.forward = self._create_wrapped_model_forward(model)
        
        # Only log once for the first model
        if len(self.models) == 1:
            self.is_active = True
            logger.info("=" * 60)
            logger.info(f"✅ MagCache ACTIVATED")
            logger.info(f"   Mode: {self.mode}")
            logger.info(f"   Total steps: {self.num_steps}")
            logger.info(f"   Retention ratio: {self.retention_ratio}")
            logger.info(f"   Error threshold: {self.magcache_thresh}")
            logger.info(f"   Max skip steps (K): {self.K}")
            logger.info(f"   Split step: {self.split_step if self.split_step else 'None'}")
            logger.info(f"   Magnitude ratios length: {len(self.mag_ratios)}")
            logger.info(f"   First 10 mag ratios: {self.mag_ratios[:10] if len(self.mag_ratios) >= 10 else self.mag_ratios}")
        
        model_desc = f" ({model_name})" if model_name else ""
        logger.info(f"   Attached to model{model_desc}: {len(model.blocks)} blocks")
        
        if len(self.models) == 1:
            logger.info("=" * 60)
        
    def detach(self):
        """Detach MagCache from all models"""
        if not self.is_active:
            return
        
        # Restore original forwards for all models
        for model in self.models:
            model_id = id(model)
            
            # Restore block forwards
            if model_id in self.original_blocks_forwards:
                for block, original_forward in zip(model.blocks, self.original_blocks_forwards[model_id]):
                    block.forward = original_forward
            
            # Restore model forward
            if model_id in self.original_forwards:
                model.forward = self.original_forwards[model_id]
        
        self.models = []
        self.original_forwards = {}
        self.original_blocks_forwards = {}
        self.is_active = False
        logger.info("MagCache detached from all models")
        
    def reset(self):
        """Reset state for new video generation"""
        logger.info(f"🔄 MagCache RESET: cnt was {self.cnt}, resetting to 0")
        self.cnt = 0
        self.accumulated_ratio = [1.0, 1.0]
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
        self.residual_cache = [None, None]  # Clear cache for new generation
        
    def _should_use_magcache(self) -> bool:
        """Determine if MagCache should be used for current step"""
        use_magcache = True
        if self.split_step is not None:
            if self.mode == "i2v":
                if self.cnt<int(self.split_step+(self.num_steps-self.split_step)*self.retention_ratio):
                    use_magcache = False
            else:
                if self.cnt<int(self.split_step*self.retention_ratio) or (self.cnt<=((self.num_steps-self.split_step)*self.retention_ratio+self.split_step) and self.cnt>=self.split_step):
                    use_magcache = False
        else:  
            if self.cnt<int(self.num_steps*self.retention_ratio): # ti2v
                use_magcache = False
        return use_magcache

    
    def _check_skip_conditions(self) -> tuple[bool, str]:
        """Check if we should skip the blocks computation
        
        Returns:
            (can_skip, reason): Whether to skip and why we're computing if not skipping
        """
        idx = self.cnt % 2  # 0 for conditional, 1 for unconditional
        
        # Get current magnitude ratio
        cur_mag_ratio = self.mag_ratios[self.cnt] # conditional and unconditional in one list
            
        # CRITICAL: Update accumulated values BEFORE checking (following original)
        self.accumulated_ratio[idx] *= cur_mag_ratio
        self.accumulated_steps[idx] += 1  # Increment FIRST
        cur_skip_err = np.abs(1 - self.accumulated_ratio[idx])
        self.accumulated_err[idx] += cur_skip_err
        
        # Debug logging for threshold behavior
        if self.cnt % 10 == 0:
            logger.debug(f"Step {self.cnt}: accumulated_err[{idx}]={self.accumulated_err[idx]:.4f}, thresh={self.magcache_thresh}, steps={self.accumulated_steps[idx]}, K={self.K}")
        
        # Determine skip reason BEFORE resetting
        can_skip = False
        
        if self.accumulated_err[self.cnt%2]<self.magcache_thresh and self.accumulated_steps[self.cnt%2]<=self.K:
            can_skip = True
        
        # Reset accumulation if we can't skip
        else:
            self.accumulated_err[idx] = 0.0
            self.accumulated_steps[idx] = 0
            self.accumulated_ratio[idx] = 1.0
            
        return can_skip
    
    def _create_wrapped_block_forward(self, block, block_idx):
        """Create a wrapped forward for a single block"""
        original_forward = block.forward
        
        @wraps(original_forward)
        def wrapped_forward(*args, **kwargs):
            # Always execute normally - the skipping logic is in model forward
            return original_forward(*args, **kwargs)
        
        return wrapped_forward
    
    def _create_wrapped_model_forward(self, model):
        """Create a wrapped forward for the entire model with MagCache logic"""
        model_id = id(model)
        original_forward = self.original_forwards[model_id]
        manager = self  # Capture self for use in wrapper
        
        # Check if this is SP forward (which doesn't accept clip_fea) once at setup
        sig = inspect.signature(original_forward)
        params = list(sig.parameters.keys())
        uses_clip_fea = 'clip_fea' in params
        
        @wraps(original_forward)
        def wrapped_forward(x, t, context, seq_len, clip_fea=None, y=None):
            # Store original blocks' forward methods temporarily
            model_id = id(model)
            original_blocks_forwards = []
            for block in model.blocks:
                original_blocks_forwards.append(block.forward)
            
            # AUTO-RESET: If cnt is at or beyond num_steps, we're starting a new generation
            # This handles cases where the previous generation ended exactly at num_steps
            # or where extra forward passes happened after the main generation loop
            if manager.cnt >= manager.num_steps:
                logger.info(f"🔄 Auto-reset: New generation detected (cnt={manager.cnt} >= {manager.num_steps})")
                manager.reset()
            
            # Debug: log to track unexpected forward passes and model identity
            current_model_id = id(model)
            model_name = "unknown"
            for i, m in enumerate(manager.models):
                if id(m) == current_model_id:
                    model_name = f"model_{i}" if len(manager.models) > 1 else "model"
                    break
            
            if manager.cnt < 3 or manager.cnt % 20 == 0 or manager.cnt >= manager.num_steps - 3:
                logger.debug(f"🔍 Forward pass: cnt={manager.cnt}/{manager.num_steps} using {model_name}")
            
            # Track if we're skipping and need residual
            use_magcache = manager._should_use_magcache()
            skip_forward = False
            if use_magcache:

                skip_forward = manager._check_skip_conditions()
            
            # Always log for debugging with clearer step counting
            sample_step = manager.cnt // 2  # Convert to sampling step
            pass_type = "cond" if manager.cnt % 2 == 0 else "uncond"
            idx = manager.cnt % 2
            
            if skip_forward:
                logger.info(f"🚀 MagCache SKIPPING [{pass_type}] step {sample_step}/{manager.num_steps//2} (err: {manager.accumulated_err[idx]:.4f}<{manager.magcache_thresh}, steps: {manager.accumulated_steps[idx]}<={manager.K})")
            elif use_magcache and manager.cnt % 4 == 0:  # Log less frequently
                logger.info(f"⚙️  MagCache COMPUTING [{pass_type}] step {sample_step}")
            # elif not use_magcache and manager.cnt % 10 == 0:
            #     logger.info(f"⏳ MagCache WAITING at step {sample_step} (starts at {int(manager.num_steps * manager.retention_ratio)//2})")
            
            # Intercept blocks execution
            blocks_input = None
            blocks_output = None
            
            def capture_blocks_io(block_idx):
                """Create a wrapper for each block that captures input/output"""
                original_block_forward = original_blocks_forwards[block_idx]
                
                @wraps(original_block_forward)
                def block_wrapper(*args, **kwargs):
                    nonlocal blocks_input, blocks_output
                    
                    # Capture input at first block
                    if block_idx == 0 and len(args) > 0:
                        # Handle both tensor and other types
                        blocks_input = args[0].clone() if hasattr(args[0], 'clone') else args[0]
                        
                        # If skipping and have cached residual, apply it and skip all blocks
                        if skip_forward and manager.residual_cache[manager.cnt % 2] is not None:
                            # Apply cached residual to input
                            result = blocks_input + manager.residual_cache[manager.cnt % 2]
                            blocks_output = result  # This is the final output
                            # Pass through all remaining blocks without computation
                            return result
                    
                    # Normal execution
                    if skip_forward and block_idx > 0:
                        # Already applied residual in first block, just pass through
                        result = args[0]
                    else:
                        result = original_block_forward(*args, **kwargs)
                    
                    # Capture output at last block
                    if block_idx == len(model.blocks) - 1:
                        blocks_output = result
                    
                    return result
                
                return block_wrapper
            
            # Apply block wrappers
            for i, block in enumerate(model.blocks):
                block.forward = capture_blocks_io(i)
            try:
                # Call original forward (which handles all preprocessing and FSDP/SP)
                if uses_clip_fea:
                    # Regular forward with clip_fea
                    result = original_forward(x=x, t=t, context=context, seq_len=seq_len, clip_fea=clip_fea, y=y)
                else:
                    # SP forward without clip_fea (no clip_fea parameter)
                    result = original_forward(x=x, t=t, context=context, seq_len=seq_len, y=y)
                
                # After forward, handle residual caching
                # CRITICAL: Following original, ALWAYS update cache (even when skipping)
                if blocks_input is not None and blocks_output is not None:
                    if skip_forward and manager.residual_cache[manager.cnt % 2] is not None:
                        # We skipped - rewrite the SAME residual value (following original exactly)
                        residual = manager.residual_cache[manager.cnt % 2]
                        manager.residual_cache[manager.cnt % 2] = residual  # ALWAYS write
                        if manager.cnt % 10 == 0:
                            logger.debug(f"Used cached residual at step {manager.cnt}, rewrote cache")
                    elif hasattr(blocks_output, '__sub__') and hasattr(blocks_input, '__sub__'):
                        # We computed - cache the NEW residual
                        residual = blocks_output - blocks_input
                        manager.residual_cache[manager.cnt % 2] = residual  # ALWAYS write
                        if manager.cnt % 10 == 0:
                            logger.debug(f"Cached new residual at step {manager.cnt}")
                
                # Update counter
                manager.cnt += 1
                
                # Log statistics at the end of generation
                if manager.cnt == manager.num_steps:
                    # Calculate actual skip statistics
                    total_steps = manager.num_steps
                    sample_steps = total_steps // 2
                    eligible_steps = sum(1 for i in range(total_steps) if i >= int(total_steps * manager.retention_ratio))
                    
                    logger.info("=" * 60)
                    logger.info(f"📊 MagCache Statistics for Generation:")
                    logger.info(f"   Sampling steps: {sample_steps} (x2 for cond/uncond = {total_steps} passes)")
                    logger.info(f"   Eligible passes: {eligible_steps}/{total_steps}")
                    logger.info(f"   Settings: thresh={manager.magcache_thresh}, K={manager.K}")
                    logger.info("=" * 60)
                    # Note: Reset will happen automatically at the start of next generation
                
                return result
            except Exception as e:
                raise e
            finally:
                # Restore original block forwards
                for block, orig_fwd in zip(model.blocks, original_blocks_forwards):
                    block.forward = orig_fwd
        
        return wrapped_forward


def apply_magcache_to_model(
    model: torch.nn.Module,
    args: Any,
    cfg: Any = None,
    mode: str = "i2v"
) -> MagCacheManager:
    """
    Apply MagCache to a model that may have FSDP/SP already configured
    
    Args:
        model: Model to apply MagCache to
        args: Arguments with MagCache configuration
        cfg: Config with model settings
        mode: Generation mode ("t2v", "i2v", or "ti2v")
    
    Returns:
        MagCacheManager instance for control
    """
    # Default magnitude ratios (you can load from file)
    #TODO: handle different mag_ratios for other modes
    mag_ratios = [0.99191, 0.99144, 0.99356, 0.99337, 0.99326, 0.99285, 0.99251, 
                  0.99264, 0.99393, 0.99366, 0.9943, 0.9943, 0.99276, 0.99288, 
                  0.99389, 0.99393, 0.99274, 0.99289, 0.99316, 0.9931, 0.99379, 
                  0.99377, 0.99268, 0.99271, 0.99222, 0.99227, 0.99175, 0.9916, 
                  0.91076, 0.91046, 0.98931, 0.98933, 0.99087, 0.99088, 0.98852, 
                  0.98855, 0.98895, 0.98896, 0.98806, 0.98808, 0.9871, 0.98711, 
                  0.98613, 0.98618, 0.98434, 0.98435, 0.983, 0.98307, 0.98185, 
                  0.98187, 0.98131, 0.98131, 0.9783, 0.97835, 0.97619, 0.9762, 
                  0.97264, 0.9727, 0.97088, 0.97098, 0.96568, 0.9658, 0.96045, 
                  0.96055, 0.95322, 0.95335, 0.94579, 0.94594, 0.93297, 0.93311, 
                  0.91699, 0.9172, 0.89174, 0.89202, 0.8541, 0.85446, 0.79823, 0.79902]
    
    # Calculate split step if needed
    split_step = None
    if cfg and hasattr(cfg, 'boundary') and hasattr(args, 'sample_shift'):
        from wan.image2video_optimized import get_timesteps
        timesteps = get_timesteps(
            shift=args.sample_shift, 
            num_inference_steps=args.sample_steps
        )
        high_noise_steps = (timesteps >= (cfg.num_train_timesteps * cfg.boundary)).sum().item()
        split_step = high_noise_steps
    

    
    # Create and attach manager
    manager = MagCacheManager(
        mag_ratios=mag_ratios,
        num_steps=args.sample_steps,
        split_step=split_step,
        mode=mode,
        magcache_thresh=getattr(args, 'magcache_thresh', 0.15),
        K=getattr(args, 'magcache_K', 10),
        retention_ratio=getattr(args, 'retention_ratio', 0.5)
    )
    
    manager.attach(model)
    return manager


class MagCacheContext:
    """
    Context manager for temporary MagCache activation
    
    Usage:
        with MagCacheContext(model, args, cfg):
            # Run generation with MagCache
            output = generate_video(...)
    """
    
    def __init__(self, model, args, cfg=None, mode="i2v"):
        self.model = model
        self.args = args
        self.cfg = cfg
        self.mode = mode
        self.manager = None
    
    def __enter__(self):
        self.manager = apply_magcache_to_model(
            self.model, self.args, self.cfg, self.mode
        )
        return self.manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.manager:
            self.manager.detach()
            self.manager = None
