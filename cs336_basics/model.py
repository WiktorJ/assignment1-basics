import torch
import numpy as np
import einx


def silu(x):
    # FLOPs count:
    # 2C * bs * seq_len * d_model
    return x * torch.sigmoid(x)


def softmax(x, dim):
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None):
    dk = K.shape[-1]
    # FLOPs count:
    # 2 * bs * num_heads * seq_len^2 * d_k  = 2 * bs * seq_len^2 * d_model
    qk = einx.dot("... q d_k, ... k d_k -> ... q k", Q, K) / np.sqrt(dk)
    if mask is not None:
        qk.masked_fill_(~mask, -torch.inf)
    # FLOPs count:
    # 2 * bs * num_heads * seq_len^2 * d_v  = 2 * bs * seq_len^2 * d_model
    # Total = 4 * bs * seq_len^2 * d_model
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
        # Param count:
        # vocab_size * d_model
        self.W = torch.nn.Parameter(W)

    def forward(self, x):
        return self.W[x]


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # Param count:
        # d_model
        self.g = torch.nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        # FLOPs count:
        # bs * seq_len * d_model
        rms = torch.sqrt(torch.mean(x**2 + self.eps, dim=-1, keepdim=True))
        # FLOPs count:
        # 2 * bs * seq_len * d_model
        # Total:
        # 3 * bs * seq_len * d_model
        return x / rms * self.g


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        # Param count:
        # d_ff * d_model
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # d_ff * d_model
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        # d_ff * d_model
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # Total: 3 * d_ff * d_model

    def forward(self, x):
        # FLOPs count:
        # (6 * bs * seq_len * d_model * d_ff) + (2C * bs * seq_len * d_model)
        # ~= (6 * bs * seq_len * d_model * d_ff)
        return self.W2(silu(self.W1(x)) * self.W3(x))


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        base = 1.0 / theta ** (torch.arange(0, d_k, 2) / d_k).to(device, dtype=dtype)
        seq_lens = torch.arange(max_seq_len, device=device, dtype=dtype)
        angles = einx.multiply("s, d_2 -> s d_2", seq_lens, base)
        # [x1, x2, ...] -> [x1, x1, x2, x2, ...]
        angles = einx.rearrange("s d_2 -> s (d_2 2)", angles)
        # Param count:
        # d_k * max_seq_len
        self.register_buffer("cos", angles.cos().to(device), persistent=False)
        self.register_buffer("sin", angles.sin().to(device), persistent=False)
        self.R = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dtype).to(device)

    def forward(self, x: torch.Tensor, token_position: torch.Tensor | None):
        # dim(x) = [..., seq_len, d_k]
        # dim(token_position) = [..., seq_len]
        # Naive operation:
        # R*q (dxd * dx1) = x (dx1) = [cos*x1 + -sin*x2, sin*x1 + cos*x2, ...]

        cos = self.cos[token_position]
        sin = self.sin[token_position]

        # FLOPs count:
        # (2 * bs * seq_len * d_k * 2) + (2 * bs * seq_len * d_k) = 8 * bs * seq_len * d_k

        # [-x2, x1, -x4, x3, ...]
        x_flipped = einx.dot("... (d_2 r_in), r_out r_in -> ... (d_2 r_out)", x, self.R)
        # [cos*x1, cos*x2, cos*x3, cos*x4, ...] + [-sin*x2, sin*x1, -sin*x4, sin*x3, ...]
        return x * cos + x_flipped * sin


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        # Param count:
        # (d_model x (3 * d_model)) + (d_model x d_model) = (4 * d_model**2)
        self.W = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.Wo = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, rope: RoPE | None = None, token_position: torch.Tensor | None = None):
        # dim(x) = [..., seq_len, d_model]
        # QKV = einx.rearrange("... s (n dm qkv) -> ... n s dm qkv", self.W(x), qkv=self.d_model, n=self.n_heads)
        seq_len = x.shape[-2]
        # FLOps count:
        # 3 * 2 * bs * seq_len * d_model^2 = 6 * bs * seq_len * d_model^2
        Q, K, V = einx.rearrange("... s (qkv n dh) -> qkv ... n s dh", self.W(x), n=self.n_heads, dh=self.d_k)
        if rope is not None:
            if token_position is None:
                token_position = torch.arange(seq_len, device=Q.device)
            # FLOPs count:
            # 2 * RoPE flops = 16 * bs * seq_len * (d_model / n_heads)
            Q = rope(Q, token_position)
            K = rope(K, token_position)
        mask = ~torch.triu(torch.ones((seq_len, seq_len), device=Q.device, dtype=Q.dtype), diagonal=1).bool()
        # FLOPs count:
        # 4 * bs * seq_len^2 * d_model
        att = scaled_dot_product_attention(Q, K, V, mask)
        att = einx.rearrange("... n s dv -> ... s (n dv)", att, n=self.n_heads)
        # FLOPs count:
        # 2 * bs * seq_len * d_model^2 = 2 * bs * seq_len * d_model^2
        # Total = (6 * bs * seq_len * d_model^2) + (16 * bs * seq_len * (d_model / n_heads)) +  (4 * bs * seq_len^2 * d_model) + (2 * bs * seq_len * d_model^2)
        #   ~= (8 * bs * seq_len * d_model^2) + (4 * bs * seq_len^2 * d_model)
        return self.Wo(att)


class TransformerBlock(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, rope: RoPE | None = None, device=None, dtype=None
    ) -> None:
        super().__init__()
        # Params count:
        # d_model
        self.rms_norm_att = RMSNorm(d_model, device=device, dtype=dtype)
        # d_model
        self.rms_norm_ff = RMSNorm(d_model, device=device, dtype=dtype)
        # (4 * d_model**2)
        self.att = MultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        # 3 * d_ff * d_model
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        # (d_mode / num_head) * max_seq_len
        self.rope = rope
        # Total:
        # (2 * d_model) + (4 * d_model**2) + (3 * d_ff * d_model) + ((d_model / num_head) * max_seq_len)
        # Trainable: (2 * d_model) + (4 * d_model**2) + (3 * d_ff * d_model)

    def forward(self, x: torch.Tensor, token_position: torch.Tensor | None = None):
        # dim(x) = [..., seq_len, d_model]
        # FLOPs count:
        # (4 * bs * seq_len * d_model) + ((8 * bs * seq_len * d_model^2) + (4 * bs * seq_len^2 * d_model))
        x = x + self.att(self.rms_norm_att(x), self.rope, token_position)
        # FLOPs count:
        # (4 * bs * seq_len * d_model) + (6 * bs * seq_len * d_model * d_ff)
        # Total:
        # (8 * bs * seq_len * d_model)  + (8 * bs * seq_len * d_model^2) + (4 * bs * seq_len^2 * d_model) + (6 * bs * seq_len * d_model * d_ff)
        return x + self.ff(self.rms_norm_ff(x))


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        # Param count:
        # vocab_size * d_model
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # num_layers * ((2 * d_model) + (4 * d_model**2) + (3 * d_ff * d_model))
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    RoPE(rope_theta, d_model // num_heads, context_length, device=device, dtype=dtype),
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        # d_model
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        # d_model * vocab_size
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        # Total:
        # d_model + (2 * d_model * vocab_size) + (num_layers * ((2 * d_model) + (4 * d_model**2) + (3 * d_ff * d_model)))

    def forward(self, x: torch.Tensor):
        # dim(x) = [batch_size, seq_len]
        # embedding [vocab_size x d_model]
        # layer:
        #
        x = self.token_embeddings(x)
        # FLOPs count:
        # num_layers * ((8 * bs * seq_len * d_model)  + (8 * bs * seq_len * d_model^2) + (4 * bs * seq_len^2 * d_model) + (6 * bs * seq_len * d_model * d_ff))
        for layer in self.layers:
            x = layer(x)
        # FLOPs count:
        # 3 * bs * seq_len * d_model
        x = self.norm(x)
        # FLOPs count:
        # 2 * bs * seq_len * d_model * vocab_size
        # Total:
        # (num_layers * ((8 * bs * seq_len * d_model)  + (8 * bs * seq_len * d_model^2) + (4 * bs * seq_len^2 * d_model)
        #   + (6 * bs * seq_len * d_model * d_ff))) + (3 * bs * seq_len * d_model) + (2 * bs * seq_len * d_model * vocab_size)
        return self.lm_head(x)
