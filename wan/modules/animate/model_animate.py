# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import types
from copy import deepcopy
from einops import  rearrange
from typing import List
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import PeftAdapterMixin
import transformer_engine.pytorch as te
import torch.nn.functional as F
from ...distributed.sequence_parallel import (
    distributed_attention,
    gather_forward,
    get_rank,
    get_world_size,
)


from ..model import (
    Head,
    WanAttentionBlock,
    WanLayerNorm,
    WanRMSNorm,
    WanModel,
    WanSelfAttention,
    flash_attention,
    rope_params,
    sinusoidal_embedding_1d,
    rope_apply
)

from .face_blocks import FaceEncoder, FaceAdapter
from .motion_encoder import Generator
def fp8_pad_for_dim(x, dim, divisor=8):
    shape = x.shape
    prod_other = 1
    for i in range(len(shape) - 1):
        if i != dim:
            prod_other *= shape[i]

    p = 0
    while (prod_other * (shape[dim] + p)) % divisor != 0:
        p += 1
    return p
class HeadAnimate(Head):

    def forward(self, x, e,fp8_enabled=False):
        """
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32 if not fp8_enabled else e.dtype == torch.bfloat16
        if not fp8_enabled:
            with amp.autocast(dtype=torch.float32):
                e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
                x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        else:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            pad_x = fp8_pad_for_dim(x, dim=1)
            x = F.pad(x, (0, 0, 0, pad_x))
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))[:, :x.shape[1]-pad_x, :]
        return x


class WanAnimateSelfAttention(WanSelfAttention):

    def forward(self, x, seq_lens, grid_sizes, freqs):
        """
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAnimateCrossAttention(WanSelfAttention):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        use_img_emb=True
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps
        )
        self.use_img_emb = use_img_emb

        if use_img_emb:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        """
        x:              [B, L1, C].
        context:        [B, L2, C].
        context_lens:   [B].
        """
        if self.use_img_emb:
            context_img = context[:, :257] #!! fp8 needs division by 8
            context = context[:, 257:]
        else:
            context = context

        b, n, d = x.size(0), self.num_heads, self.head_dim

        pad_divisor = 8 
        seq_len = x.shape[1]
        seq_len_dim_context_0 = context.shape[0]
        seq_len_dim_context_img_0 = context_img.shape[0]
        seq_len_context = context.shape[1] * seq_len_dim_context_0
        seq_len_context_img = context_img.shape[1] * seq_len_dim_context_img_0
        pad_amt = (pad_divisor - (seq_len % pad_divisor)) % pad_divisor
        # torch.distributed.breakpoint()
        if hasattr(self.q, 'base_layer') and isinstance(self.q.base_layer, te.Linear):


            # Create zeros: [Batch, Pad_Amt, Hidden]
            # We grab dim 0 (Batch) and dim 2 (Hidden) dynamically from x
            padding = torch.zeros(
                x.shape[0], pad_amt, x.shape[2], 
                device=x.device, dtype=x.dtype
            )
            if seq_len_context % pad_divisor != 0:
                pad_amt_context = (pad_divisor - (seq_len_context % pad_divisor)) % pad_divisor
                context = F.pad(context, (0, 0, 0, 0, 0, pad_amt_context))
            
            if seq_len_context_img % pad_divisor != 0:
                pad_amt_context_img = (pad_divisor - (seq_len_context_img % pad_divisor)) % pad_divisor
                context_img = F.pad(context_img, (0, 0, 0, 0, 0, pad_amt_context_img))
            # Concatenate along DIM 1 (Sequence Length)
            x = torch.cat([x, padding], dim=1)
            q = self.norm_q(self.q(x))[:, :seq_len, :].view(b, -1, n, d)
            k = self.norm_k(self.k(context)).view(b, -1, n, d) if seq_len_context % pad_divisor == 0 else self.norm_k(self.k(context))[:seq_len_dim_context_0, :, :].view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d) if seq_len_context % pad_divisor == 0 else self.v(context)[:seq_len_dim_context_0, :, :].view(b, -1, n, d)
        elif type(self.q).__name__ == "CachedTELinear":
            padding = torch.zeros(
                x.shape[0], pad_amt, x.shape[2], 
                device=x.device, dtype=x.dtype
            )
            if seq_len_context % pad_divisor != 0:
                pad_amt_context = (pad_divisor - (seq_len_context % pad_divisor)) % pad_divisor
                context = F.pad(context, (0, 0, 0, 0, 0, pad_amt_context))
            if seq_len_context_img % pad_divisor != 0:
                pad_amt_context_img = (pad_divisor - (seq_len_context_img % pad_divisor)) % pad_divisor
                context_img = F.pad(context_img, (0, 0, 0, 0, 0, pad_amt_context_img))
            x = torch.cat([x, padding], dim=1)
            q = self.norm_q(self.q(x))[:, :seq_len, :].view(b, -1, n, d)
            k = self.norm_k(self.k(context)).view(b, -1, n, d) if seq_len_context % pad_divisor == 0 else self.norm_k(self.k(context))[:seq_len_dim_context_0, :, :].view(b, -1, n, d)
                
            v = self.v(context).view(b, -1, n, d) if seq_len_context % pad_divisor == 0 else self.v(context)[:seq_len_dim_context_0, :, :].view(b, -1, n, d)
        else:
            q = self.norm_q(self.q(x)).view(b, -1, n, d)
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        if self.use_img_emb:
            k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d) if seq_len_context_img % pad_divisor == 0 else self.norm_k_img(self.k_img(context_img))[:seq_len_dim_context_img_0, :, :].view(b, -1, n, d)
            v_img = self.v_img(context_img).view(b, -1, n, d) if seq_len_context_img % pad_divisor == 0 else self.v_img(context_img)[:seq_len_dim_context_img_0, :, :].view(b, -1, n, d)
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)

        if self.use_img_emb:
            img_x = img_x.flatten(2)
            x = x + img_x

        if hasattr(self.q, 'base_layer') and isinstance(self.q.base_layer, te.Linear):
            padding = torch.zeros(
                x.shape[0], pad_amt, x.shape[2], 
                device=x.device, dtype=x.dtype
            )

            # Concatenate along DIM 1 (Sequence Length)
            x = torch.cat([x, padding], dim=1)
            x = self.o(x)[:, :seq_len, :]
        elif type(self.q).__name__ == "CachedTELinear":
            padding = torch.zeros(
                x.shape[0], pad_amt, x.shape[2], 
                device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, padding], dim=1)
            x = self.o(x)[:, :seq_len, :]
        else:
            x = self.o(x)
        return x


class WanAnimateAttentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 use_img_emb=True):

        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanAnimateSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        
        self.norm3 = WanLayerNorm(
            dim, eps, elementwise_affine=True
        ) if cross_attn_norm else nn.Identity()
            
        self.cross_attn = WanAnimateCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, use_img_emb=use_img_emb)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        fp8_enabled=False
    ):
        """
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        fp8_enabled = (hasattr(self.self_attn.q, 'base_layer') and isinstance(self.self_attn.q.base_layer, te.Linear)) or \
                      (type(self.self_attn.q).__name__ == "CachedTELinear")
        assert e.dtype == torch.float32 if not fp8_enabled else e.dtype == torch.bfloat16
        if not fp8_enabled:
            with amp.autocast(dtype=torch.float32):
                e = (self.modulation + e).chunk(6, dim=1)
        else:
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32 if not fp8_enabled else e[0].dtype == torch.bfloat16

        # self-attention
        
        if not fp8_enabled:
            y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs
        )
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[2]
        else:
            y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs
        )
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e,fp8_enabled=False):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            
            if not fp8_enabled:
                y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
                with amp.autocast(dtype=torch.float32):
                    x = x + y * e[5]
            else:
                if hasattr(self.cross_attn.q, 'base_layer') and isinstance(self.cross_attn.q.base_layer, te.Linear):
                    pad_divisor = 8 
                    seq_len = x.shape[1]
                    pad_amt = (pad_divisor - (seq_len % pad_divisor)) % pad_divisor

                    # Only perform operations if padding is actually needed
                    if pad_amt > 0:
                        x = F.pad(x, (0, 0, 0, pad_amt))
                        e = [F.pad(es, (0, 0, 0, 0, 0, pad_amt)) for es in e]

                    # The rest of your calculation
                    # preserving the singleton dimension at index 2 until this squeeze.
                    y = self.ffn(
                        self.norm2(x) * (1 + e[4]) + e[3]
                    )[:, :seq_len, :]
                    if pad_amt > 0:
                        e[5] = e[5][:, :seq_len, :]
                    x=x[:, :seq_len, :]
                elif type(self.cross_attn.q).__name__ == "CachedTELinear":
                    pad_divisor = 8 
                    seq_len = x.shape[1]
                    pad_amt = (pad_divisor - (seq_len % pad_divisor)) % pad_divisor

                    # Only perform operations if padding is actually needed
                    if pad_amt > 0:
                        x = F.pad(x, (0, 0, 0, pad_amt))

                        # e = [F.pad(es, (0, 0, 0, 0, 0, pad_amt)) for es in e]

                    # The rest of your calculation
                    # preserving the singleton dimension at index 2 until this squeeze.
                    y = self.ffn(
                        self.norm2(x) * (1 + e[4]) + e[3]
                    )[:, :seq_len, :]
                    # if pad_amt > 0:
                    #     e[5] = e[5][:, :seq_len, :]
                    x=x[:, :seq_len, :]

                else:
                    y = self.ffn(
                        self.norm2(x).float() * (1 + e[4]) + e[3]
                    )

                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e,fp8_enabled)
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds,fp8_enabled=False):
        if fp8_enabled:
            padding_divisor = 8
            dim_1 = image_embeds.shape[1]
            padding_amt = (padding_divisor - (dim_1 % padding_divisor)) % padding_divisor
            if padding_amt > 0:
                image_embeds = F.pad(image_embeds, (0, 0, 0, padding_amt))
                clip_extra_context_tokens = self.proj(image_embeds)[:,:dim_1,:]
            else:
                image_embeds = image_embeds
                clip_extra_context_tokens = self.proj(image_embeds)
        else:
            clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class WanAnimateModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=36,
                 dim=5120,
                 ffn_dim=13824,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=40,
                 num_layers=40,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 motion_encoder_dim=512,
                 use_context_parallel=False,
                 use_img_emb=True):

        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.motion_encoder_dim = motion_encoder_dim
        self.use_context_parallel = use_context_parallel
        self.use_img_emb = use_img_emb

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.pose_patch_embedding = nn.Conv3d(
            16, dim, kernel_size=patch_size, stride=patch_size
        )

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAnimateAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps, use_img_emb) for _ in range(num_layers)
        ])

        # head
        self.head = HeadAnimate(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        self.img_emb = MLPProj(1280, dim)
        
        # initialize weights
        self.init_weights()

        self.motion_encoder = Generator(size=512, style_dim=512, motion_dim=20)
        self.face_adapter = FaceAdapter(
            heads_num=self.num_heads,
            hidden_dim=self.dim,
            num_adapter_layers=self.num_layers // 5,
        )

        self.face_encoder = FaceEncoder(
            in_dim=motion_encoder_dim,
            hidden_dim=self.dim,
            num_heads=4,
        )

    def after_patch_embedding(self, x: List[torch.Tensor], pose_latents, face_pixel_values,fp8_enabled=False):
        pose_latents = [self.pose_patch_embedding(u.unsqueeze(0)) for u in pose_latents]
        for x_, pose_latents_ in zip(x, pose_latents):
            x_[:, :, 1:] += pose_latents_
        
        b,c,T,h,w = face_pixel_values.shape
        face_pixel_values = rearrange(face_pixel_values, "b c t h w -> (b t) c h w")

        encode_bs = 8
        face_pixel_values_tmp = []
        for i in range(math.ceil(face_pixel_values.shape[0]/encode_bs)):
            face_pixel_values_tmp.append(self.motion_encoder.get_motion(face_pixel_values[i*encode_bs:(i+1)*encode_bs,:],fp8_enabled=fp8_enabled))
        motion_vec = torch.cat(face_pixel_values_tmp)
        
        motion_vec = rearrange(motion_vec, "(b t) c -> b t c", t=T)
        motion_vec = self.face_encoder(motion_vec)

        B, L, H, C = motion_vec.shape
        pad_face = torch.zeros(B, 1, H, C).type_as(motion_vec)
        motion_vec = torch.cat([pad_face, motion_vec], dim=1)
        return x, motion_vec


    def after_transformer_block(self, block_idx, x, motion_vec, motion_masks=None):
        if block_idx % 5 == 0:
            adapter_args = [x, motion_vec, motion_masks, self.use_context_parallel]
            residual_out = self.face_adapter.fuser_blocks[block_idx // 5](*adapter_args)
            x = residual_out + x
        return x


    def forward(
        self,
        x,
        t,
        clip_fea,
        context,
        seq_len,
        y=None,
        pose_latents=None, 
        face_pixel_values=None
    ):
        # params
        fp8_enabled = (hasattr(self.blocks[0].cross_attn.q, 'base_layer') and isinstance(self.blocks[0].cross_attn.q.base_layer, te.Linear)) or \
                      (type(self.blocks[0].cross_attn.q).__name__ == "CachedTELinear")
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        x, motion_vec = self.after_patch_embedding(x, pose_latents, face_pixel_values,fp8_enabled=fp8_enabled)

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        if not fp8_enabled:
            with amp.autocast(dtype=torch.float32):
            

                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t,fp8=fp8_enabled).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32 if not fp8_enabled else e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16
        else:
            input = sinusoidal_embedding_1d(self.freq_dim, t,fp8=fp8_enabled)
            padding_divisor = 8
            dim_0 = input.shape[0]
            padding_amt = (padding_divisor - (dim_0 % padding_divisor)) % padding_divisor
            if padding_amt > 0:
                input = F.pad(input, (0, 0, 0, padding_amt))
            else:
                input = input
            e = self.time_embedding(input)

            e0 = self.time_projection(e).unflatten(1, (6, self.dim))[:dim_0, :]
            e=e[:dim_0, :]
            assert e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if self.use_img_emb:
            context_clip = self.img_emb(clip_fea,fp8_enabled=fp8_enabled) # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        if self.use_context_parallel:
            x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]

        for idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            x = self.after_transformer_block(idx, x, motion_vec)

        # head
        x = self.head(x, e,fp8_enabled)

        if self.use_context_parallel:
            x = gather_forward(x, dim=1)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x] if not fp8_enabled else [u for u in x]


    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
