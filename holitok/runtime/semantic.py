from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Dropout, Linear


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        dropout: float = 0.0,
        norm_layer: str = "LayerNorm",
        **kwargs,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        norm_cls = getattr(nn, norm_layer)
        self.q_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.q_norm = norm_cls(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_cls(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = Dropout(attn_drop)
        self.o_proj = Linear(hidden_size, hidden_size)
        self.o_dropout = Dropout(dropout)

    def forward(self, q, k=None, v=None, mask=None, **kwargs):
        if k is None:
            k = q
        if v is None:
            v = q
        batch, target_len, channels = q.shape
        _, source_len, _ = v.shape
        if mask is not None:
            if mask.ndim == 2:
                assert target_len == source_len
                mask = mask[:, None, None, :].expand(-1, self.num_heads, target_len, -1)
            elif mask.ndim == 3:
                assert mask.size(1) == target_len and mask.size(2) == source_len
                mask = mask[:, None, :, :].expand(-1, self.num_heads, -1, -1)

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = q.view(batch, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        attn_bias = torch.zeros(batch, self.num_heads, target_len, source_len, dtype=q.dtype, device=q.device)
        if mask is not None:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, target_len, channels)
        return self.o_dropout(self.o_proj(out))


class Mlp(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size=4096, act_layer=nn.SiLU, dropout=0.0, **kwargs):
        super().__init__()
        self.fc1 = Linear(hidden_size, ffn_hidden_size)
        self.act = act_layer()
        self.fc2 = Linear(ffn_hidden_size, hidden_size)
        self.drop = Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=16,
        ffn_hidden_size=4096,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        norm_layer="LayerNorm",
        **kwargs,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, num_heads, attn_drop=attn_dropout, norm_layer=norm_layer)
        norm_cls = getattr(nn, norm_layer)
        self.attn_norm = norm_cls(hidden_size)
        self.ffn = Mlp(hidden_size, ffn_hidden_size, dropout=ffn_dropout)
        self.ffn_norm = norm_cls(hidden_size)
        self.hidden_size = hidden_size

    @staticmethod
    def _build_causal_mask(length: int, device):
        return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))

    @staticmethod
    def _build_padding_mask(x_lens, max_len: int, device):
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(x_lens.size(0), -1)
        return positions < x_lens.unsqueeze(1)

    @staticmethod
    def _fuse_attn_mask(causal_mask, padding_mask):
        if causal_mask is None and padding_mask is None:
            return None
        if causal_mask is None:
            row = padding_mask.unsqueeze(2)
            col = padding_mask.unsqueeze(1)
            return row & col
        if padding_mask is None:
            return causal_mask.unsqueeze(0)
        causal = causal_mask.unsqueeze(0)
        row = padding_mask.unsqueeze(2)
        col = padding_mask.unsqueeze(1)
        return causal & (row & col)

    def forward(self, x, x_lens=None, causal=True):
        batch, length, channels = x.shape
        assert channels == self.hidden_size
        causal_mask = self._build_causal_mask(length, x.device) if causal else None
        padding_mask = self._build_padding_mask(x_lens, length, x.device) if x_lens is not None else None
        fused_mask = self._fuse_attn_mask(causal_mask, padding_mask)
        h = self.attn_norm(x)
        x = x + self.attn(q=h, mask=fused_mask)
        h = self.ffn_norm(x)
        return x + self.ffn(h)


class SuperviseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.type = config.get("type", "transformer")
        self.hidden_size = config.get("hidden_size", 1024)
        if self.type == "transformer":
            self.layers = nn.ModuleList(
                [
                    TransformerEncoderLayer(
                        hidden_size=self.hidden_size,
                        num_heads=config.get("num_heads", 16),
                        ffn_hidden_size=config.get("ffn_hidden_size", 4096),
                    )
                    for _ in range(config.get("num_layers", 6))
                ]
            )
            self.causal = config.get("causal", False)
            self.learn_speaker_emb = config.get("learn_speaker_emb", False)
            if self.learn_speaker_emb:
                self.speaker_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
                nn.init.normal_(self.speaker_token, std=0.02)
            else:
                self.speaker_token = None
        elif self.type == "mlp":
            self.cond_layer = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            if config.get("learn_speaker_emb", False):
                self.speaker_layer = nn.Sequential(
                    nn.LayerNorm(self.hidden_size),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                )
        else:
            raise ValueError(f"Unknown encoder type: {self.type}")

    def forward(self, x, x_lens=None):
        batch, length, dim = x.shape
        if self.type == "mlp":
            content_out = self.cond_layer(x)
            spk_emb = self.speaker_layer(x).mean(dim=1) if hasattr(self, "speaker_layer") else None
            return content_out, spk_emb

        if x_lens is None:
            x_lens = torch.full((batch,), length, device=x.device, dtype=torch.long)
        if not self.learn_speaker_emb:
            for layer in self.layers:
                x = layer(x, x_lens=x_lens, causal=self.causal)
            return x, None

        x_extended = torch.cat([x, torch.zeros(batch, 1, dim, device=x.device, dtype=x.dtype)], dim=1)
        speaker_token = self.speaker_token.expand(batch, -1, -1).squeeze(1)
        batch_indices = torch.arange(batch, device=x.device)
        x_extended[batch_indices, x_lens] = speaker_token.to(dtype=x_extended.dtype)
        x_lens_new = x_lens + 1
        for layer in self.layers:
            x_extended = layer(x_extended, x_lens=x_lens_new, causal=self.causal)
        spk_emb = x_extended[batch_indices, x_lens]
        content_out = x_extended
        content_out[batch_indices, x_lens] = 0.0
        return content_out[:, :length, :], spk_emb

