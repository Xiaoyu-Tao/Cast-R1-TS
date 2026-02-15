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
PatchTST Model Implementation
Matching Time-Series-Library checkpoint structure.
Paper: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import math

from recipe.time_series_forecast.config_utils import get_default_lengths

DEFAULT_LOOKBACK_WINDOW, DEFAULT_FORECAST_HORIZON = get_default_lengths()


class PositionEmbedding(nn.Module):
    """Position embedding with learnable pe."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    """Patch embedding for PatchTST - matches Time-Series-Library structure."""
    
    def __init__(self, d_model: int, patch_len: int, stride: int, dropout: float = 0.0, padding: bool = True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        
        # Padding layer to match Time-Series-Library (pads end of sequence by stride)
        # This ensures patch_num = (seq_len + stride - patch_len) / stride + 1
        if padding:
            self.padding_layer = nn.ReplicationPad1d((0, stride))
        
        # Matches: patch_embedding.value_embedding.weight
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # Matches: patch_embedding.position_embedding.pe
        self.position_embedding = PositionEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, n_vars]
        n_vars = x.shape[-1]
        
        # Apply padding (Time-Series-Library style): pad along seq_len dimension
        if self.padding:
            # x: [batch, seq_len, n_vars] -> [batch, n_vars, seq_len] for padding
            x = x.permute(0, 2, 1)
            x = self.padding_layer(x)
            # x: [batch, n_vars, seq_len + stride] -> [batch, seq_len + stride, n_vars]
            x = x.permute(0, 2, 1)
        
        # Create patches
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # x: [batch, n_patches, n_vars, patch_len]
        
        # Reshape for channel independence
        batch_size, n_patches, n_vars, patch_len = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * n_vars, n_patches, patch_len)
        
        # Embedding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


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
        d_ff = d_ff or 4 * d_model
        
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


class Transpose(nn.Module):
    """Transpose module for use in nn.Sequential."""
    
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class FlattenHead(nn.Module):
    """Flatten head for prediction."""
    
    def __init__(self, n_vars: int, nf: int, pred_len: int):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch * n_vars, n_patches, d_model]
        x = self.flatten(x)
        x = self.linear(x)
        return x


class PatchTST(nn.Module):
    """
    PatchTST model matching Time-Series-Library checkpoint structure.
    
    Note: Non-stationary Transformer normalization is built into forward().
    Input raw data, output is in original scale.
    """
    
    def __init__(
        self,
        seq_len: int = DEFAULT_LOOKBACK_WINDOW,
        pred_len: int = DEFAULT_FORECAST_HORIZON,
        enc_in: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 1,
        d_ff: int = 2048,
        dropout: float = 0.0,
        patch_len: int = 16,
        stride: int = 8,
        **kwargs
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        
        # Number of patches (with padding: effective_seq_len = seq_len + stride)
        # This matches Time-Series-Library's PatchTST which pads by stride before patching
        self.patch_num = int((seq_len + stride - patch_len) / stride + 1)
        
        # Patch embedding (with padding enabled to match Time-Series-Library)
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout, padding=True)
        
        # Encoder
        # Note: norm_layer uses Transpose + BatchNorm1d(d_model) + Transpose to match Time-Series-Library checkpoint
        self.encoder = Encoder(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        )
        
        # Head
        self.head = FlattenHead(enc_in, self.patch_num * d_model, pred_len)

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
        
        # Patch embedding
        x, n_vars = self.patch_embedding(x)
        
        # Encoder
        x = self.encoder(x)
        
        # Reshape for head if needed
        if isinstance(x, tuple):
            x = x[0]
        
        # Prediction head
        x = self.head(x)
        
        # Reshape: [batch * n_vars, pred_len] -> [batch, pred_len, n_vars]
        batch_size = x.shape[0] // n_vars
        dec_out = x.reshape(batch_size, n_vars, -1).permute(0, 2, 1)
        
        # De-Normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        
        return dec_out


def create_patchtst_model(config: dict) -> PatchTST:
    """Create PatchTST model from config."""
    return PatchTST(
        seq_len=config.get('seq_len', DEFAULT_LOOKBACK_WINDOW),
        pred_len=config.get('pred_len', DEFAULT_FORECAST_HORIZON),
        enc_in=config.get('enc_in', 1),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        e_layers=config.get('e_layers', 1),
        d_ff=config.get('d_ff', 2048),
        dropout=config.get('dropout', 0.0),
        patch_len=config.get('patch_len', 16),
        stride=config.get('stride', 8),
    )
