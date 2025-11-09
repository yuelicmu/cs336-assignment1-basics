from __future__ import annotations

from jaxtyping import Bool, Float, Int

import torch
from torch import Tensor
from torch import linalg as LA

import torch.nn as nn
import numpy as np

from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        sigma = np.sqrt(2/(in_features+out_features))
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype), 
                mean=0.0, std=sigma, a=-3*sigma, b=3*sigma
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                mean=0.0, std=1.0, a=-3.0, b=3.0
            )
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x * (self.weight / rms)
        return result.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        sigma = np.sqrt(2/(d_model + d_ff))
        self.w1_weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_ff, d_model, device=device, dtype=dtype), 
                mean=0.0, std=sigma, a=-3*sigma, b=3*sigma
            )
        )
        self.w2_weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_ff, device=device, dtype=dtype), 
                mean=0.0, std=sigma, a=-3*sigma, b=3*sigma
            )
        )
        self.w3_weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_ff, d_model, device=device, dtype=dtype), 
                mean=0.0, std=sigma, a=-3*sigma, b=3*sigma
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        w1_x = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        silu = w1_x * torch.sigmoid(w1_x)
        w3_x = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        return einsum(silu * w3_x, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        buffer = self.construct_buffer()
        if device is not None:
            buffer = buffer.to(device)
        self.register_buffer("RoPE_buffer", buffer, persistent=False)

    def construct_buffer(self) -> Float[Tensor, "max_seq_len d_k_half 2 2"]:
        theta = self.theta
        d_k_half = self.d_k // 2
        max_seq_len = self.max_seq_len
        # Vectorized version
        i = torch.arange(max_seq_len, dtype=int)
        k = torch.arange(d_k_half, dtype=int)
        i = i[:, None]  # (max_seq_len, 1)
        k = k[None, :]  # (1, d_k_half)
        # Compute theta_ik for all i, k
        exponent = (2 * k) / (2 * d_k_half)
        theta_ik = i / (theta ** exponent)  # (max_seq_len, d_k_half)
        cos_val = torch.cos(theta_ik)
        sin_val = torch.sin(theta_ik)
        # Build the 2x2 rotation matrices
        out = torch.empty(max_seq_len, d_k_half, 2, 2, dtype=torch.float32)
        out[..., 0, 0] = cos_val
        out[..., 0, 1] = -sin_val
        out[..., 1, 0] = sin_val
        out[..., 1, 1] = cos_val
        return out
    
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        x_reshape = rearrange(x, "... (b1 b2) -> ... b1 b2", b2 = 2)
        y = einsum(x_reshape, self.RoPE_buffer[token_positions], " ... a, ... b a -> ... b")
        return rearrange(y, "... k_half b -> ... (k_half b)")
    
class Softmax(nn.Module):
    def __init__(
        self,
        dim: int
    ) -> None:
        super().__init__()
        self.dim = dim
    def forward(self, x: Tensor) -> Tensor:
        x = x - torch.amax(x, dim=self.dim, keepdim=True)
        x = torch.exp(x)
        x = x / torch.sum(x, dim=self.dim, keepdim=True)
        return x
        
class ScaledDotProductAttentionModule(nn.Module):
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    def __init__(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None
    ) -> None:
        super().__init__()
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask

    def forward(self) -> Tensor:
        d_k = self.Q.shape[-1]
        prod = einsum(self.Q, self.K, "... queries d_k, ... keys d_k -> ... queries keys") / np.sqrt(d_k)
        if self.mask is not None:
            prod[~self.mask] = -torch.inf
        softmax_module = Softmax(dim = -1)
        return einsum(softmax_module(prod), self.V, "... queries keys, ... keys d_v -> ... queries d_v")
    
class MultiHeadSelfAttention(nn.Module):
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rope: bool | None = False,
        theta: float | None = None,
        max_seq_len: int | None = None,
        token_positions: Int[Tensor, " ... sequence_length"] | None = None
    ) -> None:
        super().__init__()
        self.d_v = d_model // num_heads
        sigma = np.sqrt(2/(self.d_v+d_model))
        sigma_o = np.sqrt(1/d_model)

        self.w_q = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_heads, self.d_v, d_model, device=device, dtype=dtype),
                mean=0.0, std=sigma, a=-3*sigma, b=3*sigma
            )
        )
        self.w_k = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_heads, self.d_v, d_model, device=device, dtype=dtype),
                mean=0.0, std=sigma, a=-3*sigma, b=3*sigma
            )
        )
        self.w_v = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_heads, self.d_v, d_model, device=device, dtype=dtype),
                mean=0.0, std=sigma, a=-3*sigma, b=3*sigma
            )
        )
        self.w_o = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_model, device=device, dtype=dtype),
                mean=0.0, std=sigma_o, a=-3*sigma_o, b=3*sigma_o
            )
        )
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.token_positions = token_positions

    def get_mask(self, x):
        # ..., num_heads, seq_len, seq_len
        seq_len = x.shape[-2]
        base = torch.ones(seq_len, seq_len).to(torch.bool)
        return torch.tril(base).expand(*x.shape[:-2], self.num_heads, seq_len, seq_len)
    
    def forward(self, x: Float[Tensor, "... seq_len d_model"]):
        mask = self.get_mask(x)
        Q = einsum(self.w_q, x, "num_heads d_v d_model, ... seq_len d_model -> ... num_heads seq_len d_v")
        K = einsum(self.w_k, x, "num_heads d_v d_model, ... seq_len d_model -> ... num_heads seq_len d_v")
        
        if self.use_rope:
            RopeModule = RotaryPositionalEmbedding(theta = self.theta, d_k = self.d_v, max_seq_len = self.max_seq_len)
            u = self.token_positions
            token_positions_expanded = u.unsqueeze(-3).expand(*u.shape[:-2], self.num_heads, *u.shape[-2:])
            Q = RopeModule(Q, token_positions_expanded)
            K = RopeModule(K, token_positions_expanded)

        V = einsum(self.w_v, x, "num_heads d_v d_model, ... seq_len d_model -> ... num_heads seq_len d_v")
        scaled_dot_product_attn_module = ScaledDotProductAttentionModule(Q, K, V, mask)
        attn = scaled_dot_product_attn_module() # ... num_heads seq_len d_v
        attn_reshape = rearrange(attn, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        return einsum(self.w_o, attn_reshape, "... d_out d_model, ... seq_len d_model -> ... seq_len d_out")
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        theta: float, 
        max_seq_len: int, 
        # token_positions: Int[Tensor, " ... sequence_length"],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.mha = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
            use_rope=True,
            theta=theta,
            max_seq_len=max_seq_len,
            token_positions=None  # Will be set in forward
        )
        self.rms1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.swiGLU = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.rms2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... seq_len d_model"]) -> Float[Tensor, "... seq_len d_model"]:
        token_positions = torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape[0], -1)
        self.mha.token_positions = token_positions
        x = x + self.mha(self.rms1(x))
        x = x + self.swiGLU(self.rms2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    theta=theta,
                    max_seq_len=max_seq_len,
                    device=device,
                    dtype=dtype
                )
            )
        self.rms_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Int[Tensor, " batch sequence_length"]) -> Float[Tensor, " batch sequence_length vocab_size"]:
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.rms_final(x)
        logits = self.lm_head(x)
        return logits