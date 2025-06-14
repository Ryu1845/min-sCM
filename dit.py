# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
# this is modeling code for DiT-LLaMA model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp

# Import the Flash Attention JVP kernel
from standalone_multihead_jvp_test import (
    flash_attention_jvp_multihead_triton_kernel_wrapper,
)


def jvp_wrap(fn):
    def closured_self(x):
        return fn(x)
    def closured_self_jvp(x, dx):
        return jvp(closured_self, (x,), (dx,))
    return closured_self_jvp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Pixel normalization for adaptive double normalization
def pnorm(v, eps=1e-8):
    return v * torch.rsqrt(torch.mean(v**2, dim=-1, keepdim=True) + eps)

# JVP for pnorm:
# Using torch.func.jvp directly on pnorm function would be cleaner if it can be made to work
# For now, let's make a jvp_wrap for pnorm
def jvp_wrap_pnorm(fn_to_wrap, eps=1e-8):
    def closured_fn(v_primal):
        return fn_to_wrap(v_primal, eps)
    def jvp_fn(v_primal, v_tangent):
        return jvp(closured_fn, (v_primal,), (v_tangent,))
    return jvp_fn

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=64):  # Reduced from 256 for positional embeddings
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, scale_factor=0.02):  # Add scale_factor for positional embedding
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        # Apply scale factor to reduce frequency magnitude (positional embedding as per sCM paper)
        args = (t[:, None] * freqs[None]) * scale_factor
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
            drop_ids = drop_ids.cuda()
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.n_rep = 1
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)

    def forward_with_jvp(self, x, t_x, freqs_cis):
        bsz, seqlen, _ = x.shape
        (xq, t_xq), (xk, t_xk), (xv, t_xv) = jvp_wrap(self.wq)(x, t_x), jvp_wrap(self.wk)(x, t_x), jvp_wrap(self.wv)(x, t_x)

        dtype = xq.dtype

        # Apply normalization to queries and keys (MISSING IN ORIGINAL!)
        (xq, t_xq), (xk, t_xk) = jvp_wrap(self.q_norm)(xq, t_xq), jvp_wrap(self.k_norm)(xk, t_xk)

        xq, xk, xv = xq.view(bsz, seqlen, self.n_heads, self.head_dim), xk.view(bsz, seqlen, self.n_heads, self.head_dim), xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        t_xq, t_xk, t_xv = t_xq.view(bsz, seqlen, self.n_heads, self.head_dim), t_xk.view(bsz, seqlen, self.n_heads, self.head_dim), t_xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        def apply_rotary_emb(xq, xk):
            return self.apply_rotary_emb(xq, xk, freqs_cis)
        (xq, t_xq), (xk, t_xk) = jvp(apply_rotary_emb, (xq, xk), (t_xq, t_xk))
        xq, xk = xq.to(dtype), xk.to(dtype)
        t_xq, t_xk = t_xq.to(dtype), t_xk.to(dtype)

        # Calculate scale for attention (same as in forward method)
        scale = 1.0 / (self.head_dim ** 0.5)

        # Permute for multi-head attention format (B, H, L, D)
        xq, xk, xv = xq.permute(0, 2, 1, 3), xk.permute(0, 2, 1, 3), xv.permute(0, 2, 1, 3)
        t_xq, t_xk, t_xv = t_xq.permute(0, 2, 1, 3), t_xk.permute(0, 2, 1, 3), t_xv.permute(0, 2, 1, 3)

        output, t_output = flash_attention_jvp_multihead_triton_kernel_wrapper(xq, xk, xv, t_xq, t_xk, t_xv, scale)

        # Permute back to (B, L, H, D)
        output, t_output = output.permute(0, 2, 1, 3), t_output.permute(0, 2, 1, 3)

        # Flatten heads dimension
        output, t_output = output.flatten(-2), t_output.flatten(-2)

        return jvp_wrap(self.wo)(output, t_output)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )

    def forward(self, x, freqs_cis, adaln_input=None):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

            # Apply pnorm to scales and shifts
            scale_msa = pnorm(scale_msa)
            shift_msa = pnorm(shift_msa)
            scale_mlp = pnorm(scale_mlp)
            shift_mlp = pnorm(shift_mlp)

            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis
            )
            x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
            )
        else:
            x = x + self.attention(self.attention_norm(x), freqs_cis)
            x = x + self.feed_forward(self.ffn_norm(x))

        return x

    def forward_with_jvp(self, x, t_x, freqs_cis, adaln_input=None):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

            # Apply pnorm to primal parts of scales and shifts
            # Assuming adaln_input is constant for JVP w.r.t. x, so tangents of mod_params are 0.
            # If adaln_input had a tangent, then mod_params_tangent would be non-zero,
            # and we'd need jvp_pnorm(scale_msa_primal, scale_msa_tangent)

            scale_msa_p = pnorm(scale_msa) # Apply to primal
            shift_msa_p = pnorm(shift_msa) # Apply to primal
            scale_mlp_p = pnorm(scale_mlp) # Apply to primal
            shift_mlp_p = pnorm(shift_mlp) # Apply to primal

            # Attention path
            x_norm_p, x_norm_t = jvp_wrap(self.attention_norm)(x, t_x)

            attn_input_primal = modulate(x_norm_p, shift_msa_p, scale_msa_p)
            # Corrected tangent for modulated attention input, using pnormed scales
            attn_input_tangent = x_norm_t * (1 + scale_msa_p.unsqueeze(1))

            attn_out, t_attn_out = self.attention.forward_with_jvp(
                attn_input_primal, attn_input_tangent, freqs_cis
            )
            x = x + gate_msa.unsqueeze(1) * attn_out
            t_x = t_x + gate_msa.unsqueeze(1) * t_attn_out

            # FFN path
            x_ffn_norm_p, x_ffn_norm_t = jvp_wrap(self.ffn_norm)(x, t_x)

            ffn_input_primal = modulate(x_ffn_norm_p, shift_mlp_p, scale_mlp_p)
            # Corrected tangent for modulated FFN input, using pnormed scales
            ffn_input_tangent = x_ffn_norm_t * (1 + scale_mlp_p.unsqueeze(1))

            ffn_out, t_ffn_out = jvp_wrap(self.feed_forward)(
                ffn_input_primal, ffn_input_tangent
            )
            x = x + gate_mlp.unsqueeze(1) * ffn_out
            # Corrected update of t_x
            t_x = t_x + gate_mlp.unsqueeze(1) * t_ffn_out
        else:
            # Attention path
            x_norm_p, x_norm_t = jvp_wrap(self.attention_norm)(x, t_x)
            attn_out, t_attn_out = self.attention.forward_with_jvp(x_norm_p, x_norm_t, freqs_cis)
            x = x + attn_out
            t_x = t_x + t_attn_out

            # FFN path
            x_ffn_norm_p, x_ffn_norm_t = jvp_wrap(self.ffn_norm)(x, t_x)
            ffn_out, t_ffn_out = jvp_wrap(self.feed_forward)(x_ffn_norm_p, x_ffn_norm_t)
            x = x + ffn_out
            t_x = t_x + t_ffn_out

        return x, t_x

    def forward_with_jvp_time(self, x_p, x_t, freqs_cis, adaln_input_p, adaln_input_t):
        """
        Forward pass with JVP with respect to time (through adaln_input).
        x_p: primal input tensor
        x_t: tangent input tensor
        freqs_cis: positional encodings (constant w.r.t. time)
        adaln_input_p: primal adaln_input
        adaln_input_t: tangent adaln_input (from time derivative)
        """
        if adaln_input_p is not None:
            # Compute modulation parameters and their tangents
            mod_p, mod_t = jvp_wrap(self.adaLN_modulation)(adaln_input_p, adaln_input_t)

            shift_msa_p, scale_msa_p, gate_msa_p, shift_mlp_p, scale_mlp_p, gate_mlp_p = mod_p.chunk(6, dim=1)
            shift_msa_t, scale_msa_t, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = mod_t.chunk(6, dim=1)

            # Apply pnorm to primal parts and compute tangents
            scale_msa_pnorm_p, scale_msa_pnorm_t = jvp_wrap_pnorm(pnorm)(scale_msa_p, scale_msa_t)
            shift_msa_pnorm_p, shift_msa_pnorm_t = jvp_wrap_pnorm(pnorm)(shift_msa_p, shift_msa_t)
            scale_mlp_pnorm_p, scale_mlp_pnorm_t = jvp_wrap_pnorm(pnorm)(scale_mlp_p, scale_mlp_t)
            shift_mlp_pnorm_p, shift_mlp_pnorm_t = jvp_wrap_pnorm(pnorm)(shift_mlp_p, shift_mlp_t)

            # Attention path
            x_norm_p, x_norm_t = jvp_wrap(self.attention_norm)(x_p, x_t)

            # Modulated attention input and its tangent
            attn_input_p = modulate(x_norm_p, shift_msa_pnorm_p, scale_msa_pnorm_p)
            attn_input_t = (
                x_norm_t * (1 + scale_msa_pnorm_p.unsqueeze(1)) +
                x_norm_p * scale_msa_pnorm_t.unsqueeze(1) +
                shift_msa_pnorm_t.unsqueeze(1)
            )

            attn_out_p, attn_out_t = self.attention.forward_with_jvp(attn_input_p, attn_input_t, freqs_cis)

            # Apply gating and residual connection
            residual_p = gate_msa_p.unsqueeze(1) * attn_out_p
            residual_t = (
                gate_msa_t.unsqueeze(1) * attn_out_p +
                gate_msa_p.unsqueeze(1) * attn_out_t
            )
            h_p = x_p + residual_p
            h_t = x_t + residual_t

            # FFN path
            h_ffn_norm_p, h_ffn_norm_t = jvp_wrap(self.ffn_norm)(h_p, h_t)

            ffn_input_p = modulate(h_ffn_norm_p, shift_mlp_pnorm_p, scale_mlp_pnorm_p)
            ffn_input_t = (
                h_ffn_norm_t * (1 + scale_mlp_pnorm_p.unsqueeze(1)) +
                h_ffn_norm_p * scale_mlp_pnorm_t.unsqueeze(1) +
                shift_mlp_pnorm_t.unsqueeze(1)
            )

            ffn_out_p, ffn_out_t = jvp_wrap(self.feed_forward)(ffn_input_p, ffn_input_t)

            ffn_residual_p = gate_mlp_p.unsqueeze(1) * ffn_out_p
            ffn_residual_t = (
                gate_mlp_t.unsqueeze(1) * ffn_out_p +
                gate_mlp_p.unsqueeze(1) * ffn_out_t
            )

            out_p = h_p + ffn_residual_p
            out_t = h_t + ffn_residual_t

            return out_p, out_t
        else:
            return self.forward_with_jvp(x_p, x_t, freqs_cis, adaln_input=None)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_Llama(nn.Module):
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        patch_size=2,
        dim=512,
        n_layers=5,
        n_heads=16,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        class_dropout_prob=0.1,
        num_classes=10,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size

        self.init_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
        )

        self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.freqs_cis = DiT_Llama.precompute_freqs_cis(dim // n_heads, 4096)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.size()
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def forward(self, x, t, y):
        self.freqs_cis = self.freqs_cis.to(x.device)

        x = self.init_conv_seq(x)

        x = self.patchify(x)
        x = self.x_embedder(x)

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        adaln_input = t.to(x.dtype) + y.to(x.dtype)

        for layer in self.layers:
            x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward_with_jvp(self, x, t_x, t, y_label):
        self.freqs_cis = self.freqs_cis.to(x.device)

        # Propagate JVP through init_conv_seq
        # Assuming init_conv_seq is a sequence of layers, and jvp_wrap handles nn.Sequential
        x_conv_p, x_conv_t = jvp_wrap(self.init_conv_seq)(x, t_x)

        # Patchify (reshaping, applied to both primal and tangent)
        x_patch_p = self.patchify(x_conv_p)
        x_patch_t = self.patchify(x_conv_t) # Assuming patchify is linear or JVP applies element-wise post-flattening

        # Propagate JVP through x_embedder
        x_embed_p, x_embed_t = jvp_wrap(self.x_embedder)(x_patch_p, x_patch_t)

        # Timestep and label embeddings (adaln_input)
        # adaln_input is treated as constant for the JVP w.r.t. x
        # If JVP w.r.t. t or y_label were needed, this would be different.
        time_emb = self.t_embedder(t)  # (N, D)
        label_emb = self.y_embedder(y_label, self.training)  # (N, D)
        adaln_input = time_emb.to(x_embed_p.dtype) + label_emb.to(x_embed_p.dtype)

        # Current x and its tangent
        current_x_p = x_embed_p
        current_x_t = x_embed_t

        for layer in self.layers:
            current_x_p, current_x_t = layer.forward_with_jvp(
                current_x_p, current_x_t, self.freqs_cis[: current_x_p.size(1)], adaln_input=adaln_input
            )

        # Propagate JVP through final_layer
        # final_layer.forward(x, c) where c is adaln_input (constant for this JVP path)
        # We need jvp(lambda _x: self.final_layer(_x, adaln_input), (current_x_p,), (current_x_t,))

        final_layer_fn = lambda _x: self.final_layer(_x, adaln_input)
        x_final_p, x_final_t = jvp(final_layer_fn, (current_x_p,), (current_x_t,))

        # Unpatchify (reshaping, applied to both primal and tangent)
        out_p = self.unpatchify(x_final_p)
        out_t = self.unpatchify(x_final_t)

        return out_p, out_t

    def forward_with_jvp_time(self, x, t, t_tangent, y_label):
        """
        Compute JVP with respect to time t.
        x: input tensor (constant w.r.t. time)
        t: time tensor
        t_tangent: tangent vector for time (usually ones)
        y_label: class labels (constant w.r.t. time)
        """
        self.freqs_cis = self.freqs_cis.to(x.device)

        # Process input (constant w.r.t. time)
        x_conv = self.init_conv_seq(x)
        x_patch = self.patchify(x_conv)
        x_embed = self.x_embedder(x_patch)

        # Timestep and label embeddings with JVP w.r.t. time
        # Convert t to float for JVP compatibility
        t_float = t.float()
        time_emb_p, time_emb_t = jvp_wrap(self.t_embedder)(t_float, t_tangent)
        label_emb = self.y_embedder(y_label, self.training)  # (N, D) - constant w.r.t. time

        # adaln_input has tangent only from time embedding
        adaln_input_p = time_emb_p.to(x_embed.dtype) + label_emb.to(x_embed.dtype)
        adaln_input_t = time_emb_t.to(x_embed.dtype)  # Only time contributes to tangent

        # Start with embedded input
        current_x_p = x_embed
        current_x_t = torch.zeros_like(current_x_p)

        for layer in self.layers:
            # Each layer's forward_with_jvp_time correctly computes both the output and its tangent w.r.t. time
            current_x_p, current_x_t = layer.forward_with_jvp_time(
                current_x_p, current_x_t, self.freqs_cis[: current_x_p.size(1)],
                adaln_input_p, adaln_input_t
            )

        # Propagate JVP through final_layer
        # The final layer depends on both current_x and adaln_input, both of which depend on time
        def final_layer_fn(x_param, adaln_param):
            return self.final_layer(x_param, adaln_param)

        final_out_p, final_out_t = jvp(
            final_layer_fn,
            (current_x_p, adaln_input_p),
            (current_x_t, adaln_input_t)
        )

        # Unpatchify
        out_p = self.unpatchify(final_out_p)
        out_t = self.unpatchify(final_out_t)

        return out_p, out_t


def DiT_Llama_600M_patch2(**kwargs):
    return DiT_Llama(patch_size=2, dim=256, n_layers=16, n_heads=16, **kwargs)


def DiT_Llama_3B_patch2(**kwargs):
    return DiT_Llama(patch_size=2, dim=3072, n_layers=32, n_heads=32, **kwargs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiT_Llama_600M_patch2().to(device)
    model.eval()
    x = torch.randn(2, 3, 32, 32, device=device)
    t = torch.randint(0, 100, (2,), device=device)
    y = torch.randint(0, 10, (2,), device=device)

    with torch.no_grad():
        out = model(x, t, y)
        print(out.shape)
        # Create a dummy tangent for x
        t_x = torch.ones_like(x)
        out_primal, out_tangent = model.forward_with_jvp(x, t_x, t, y)
        print(f"jvp(x) primal shape: {out_primal.shape}")
        print(f"jvp(x) tangent shape: {out_tangent.shape}")
        assert torch.allclose(out, out_primal, atol=1e-5)
        print("jvp(x) primal consistency check passed.")


        # Test forward_with_jvp_time
        print("\nTesting forward_with_jvp_time...")
        t_t = torch.ones_like(t).float()
        out_primal_time, out_tangent_time = model.forward_with_jvp_time(x, t, t_t, y)
        print(f"jvp(t) primal shape: {out_primal_time.shape}")
        print(f"jvp(t) tangent shape: {out_tangent_time.shape}")

        # The primal output from jvp should match the forward pass output
        assert torch.allclose(out, out_primal_time, atol=1e-5)
        print("jvp(t) primal consistency check passed.")


        out = model.forward_with_cfg(x, t, y, 0.5)
        print(f"\nforward_with_cfg output shape: {out.shape}")
