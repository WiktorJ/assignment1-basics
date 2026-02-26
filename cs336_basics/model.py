import torch
import numpy as np
import einx


def silu(x):
    return x * torch.sigmoid(x)


def softmax(x, dim):
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None):
    dk = K.shape[-1]
    qk = einx.dot("... q d_k, ... k d_k -> ... q k", Q, K) / np.sqrt(dk)
    if mask is not None:
        qk.masked_fill_(~mask, -torch.inf)
    return einx.dot("... q k, ... k d_v -> ... q d_v", softmax(qk, dim=-1), V)


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        std = np.sqrt(2 / (in_features + out_features))
        W = torch.empty((out_features, in_features), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(W, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W = torch.nn.Parameter(W)

    def forward(self, x):
        return einx.dot("d_out d_in, ... d_in -> ... d_out", self.W, x)


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        W = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(W, mean=0, std=1, a=-3, b=3)
        self.W = torch.nn.Parameter(W)

    def forward(self, x):
        return self.W[x]


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.g = torch.nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2 + self.eps, dim=-1, keepdim=True))
        return x / rms * self.g


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        return self.W2(silu(self.W1(x)) * self.W3(x))


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        base = 1.0 / theta ** (torch.arange(0, d_k, 2) / d_k).to(device, dtype=dtype)
        seq_lens = torch.arange(max_seq_len, device=device, dtype=dtype)
        angles = einx.multiply("s, d_2 -> s d_2", seq_lens, base)
        # [x1, x2, ...] -> [x1, x1, x2, x2, ...]
        angles = einx.rearrange("s d_2 -> s (d_2 2)", angles)
        self.register_buffer("cos", angles.cos().to(device), persistent=False)
        self.register_buffer("sin", angles.sin().to(device), persistent=False)
        self.R = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dtype).to(device)

    def forward(self, x: torch.Tensor, token_position: torch.Tensor):
        # dim(x) = [..., seq_len, d_k]
        # dim(token_position) = [..., seq_len]
        # Naive operation:
        # R*q (dxd * dx1) = x (dx1) = [cos*x1 + -sin*x2, sin*x1 + cos*x2, ...]

        cos = self.cos[token_position]
        sin = self.sin[token_position]

        # [-x2, x1, -x4, x3, ...]
        x_flipped = einx.dot("... (d_2 r_in), r_out r_in -> ... (d_2 r_out)", x, self.R)
        # [cos*x1, cos*x2, cos*x3, cos*x4, ...] + [-sin*x2, sin*x1, -sin*x4, sin*x3, ...]
        return x * cos + x_flipped * sin


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.W = Linear(d_model, 3 * d_model, device=None, dtype=None)
        self.Wo = Linear(d_model, d_model, device=None, dtype=None)

    def forward(self, x: torch.Tensor, rope: RoPE | None = None, token_position: torch.Tensor | None = None):
        # dim(x) = [..., seq_len, d_model]
        # QKV = einx.rearrange("... s (n dm qkv) -> ... n s dm qkv", self.W(x), qkv=self.d_model, n=self.n_heads)
        seq_len = x.shape[-2]
        Q, K, V = einx.rearrange("... s (qkv n dh) -> qkv ... n s dh", self.W(x), n=self.n_heads, dh=self.d_k)
        if rope is not None:
            Q = rope(Q, token_position)
            K = rope(K, token_position)
        mask = ~torch.triu(torch.ones((seq_len, seq_len), device=Q.device, dtype=Q.dtype), diagonal=1).bool()
        att = scaled_dot_product_attention(Q, K, V, mask)
        att = einx.rearrange("... n s dv -> ... s (n dv)", att, n=self.n_heads)
        return self.Wo(att)
