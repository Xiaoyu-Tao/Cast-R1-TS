# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
iTransformer Model Implementation
Matching Time-Series-Library checkpoint structure.
Paper: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import math

from recipe.time_series_forecast.config_utils import get_default_lengths

DEFAULT_LOOKBACK_WINDOW, DEFAULT_FORECAST_HORIZON = get_default_lengths()


class DataEmbedding_inverted(nn.Module):
    """Inverted data embedding - matches Time-Series-Library structure."""
    
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        # Matches: enc_embedding.value_embedding
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, n_vars]
        # Transpose to [batch, n_vars, seq_len]
        x = x.permute(0, 2, 1)
        # Embed: [batch, n_vars, d_model]
        x = self.value_embedding(x)
        return self.dropout(x)


class FullAttention(nn.Module):
    """Full attention mechanism - matches Time-Series-Library structure."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Matches: encoder.attn_layers.*.attention.*_projection
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Attention
        scores = torch.einsum("blhd,bshd->bhls", queries, keys) * self.scale
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        
        out = out.reshape(B, L, -1)
        return self.out_projection(out)


class EncoderLayer(nn.Module):
    """Encoder layer - matches Time-Series-Library structure."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or d_model
        
        self.attention = FullAttention(d_model, n_heads, dropout)
        
        # Matches: encoder.attn_layers.*.conv1/conv2
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        
        # Matches: encoder.attn_layers.*.norm1/norm2
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # FFN with Conv1d
        y = x.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(1, 2)
        
        x = x + y
        x = self.norm2(x)
        
        return x


class Encoder(nn.Module):
    """Encoder - matches Time-Series-Library structure."""
    
    def __init__(self, layers: list, norm_layer: nn.Module = None):
        super().__init__()
        self.attn_layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.attn_layers:
            x = layer(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x


class iTransformer(nn.Module):
    """
    iTransformer model matching Time-Series-Library checkpoint structure.
    
    Note: Non-stationary Transformer normalization is built into forward().
    Input raw data, output is in original scale.
    """
    
    def __init__(
        self,
        seq_len: int = DEFAULT_LOOKBACK_WINDOW,
        pred_len: int = DEFAULT_FORECAST_HORIZON,
        enc_in: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        
        # Encoder
        self.encoder = Encoder(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # Projection
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, n_vars]
        
        Returns:
            Predictions of shape [batch, pred_len, n_vars]
        """
        # Non-stationary Transformer Normalization
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        
        _, _, N = x.shape
        
        # Embedding: [batch, n_vars, d_model]
        enc_out = self.enc_embedding(x)
        
        # Encoder
        enc_out = self.encoder(enc_out)
        
        # Project: [batch, n_vars, pred_len]
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        
        return dec_out


def create_itransformer_model(config: dict) -> iTransformer:
    """Create iTransformer model from config."""
    return iTransformer(
        seq_len=config.get('seq_len', DEFAULT_LOOKBACK_WINDOW),
        pred_len=config.get('pred_len', DEFAULT_FORECAST_HORIZON),
        enc_in=config.get('enc_in', 1),
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 8),
        e_layers=config.get('e_layers', 2),
        d_ff=config.get('d_ff', 128),
        dropout=config.get('dropout', 0.0),
    )
