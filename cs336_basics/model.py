import torch
import numpy as np
import einx


def silu(x):
    return x * torch.sigmoid(x)


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
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        return self.W2(silu(self.W1(x)) * self.W3(x))
