from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class Optimizer(torch.optim.Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, {"lr": lr})

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = closure() if closure else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                statep = self.state[p]
                t = statep.get("t", 0)
                grad = p.grad.data
                p.data = p.data - (group["lr"] / math.sqrt(t + 1)) * grad
                statep["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = closure() if closure else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                statep = self.state[p]
                t = statep.get("t", 1)
                m = statep.get("m", torch.zeros_like(p.data))
                v = statep.get("v", torch.zeros_like(p.data))
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                grad = p.grad.data

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                alpha = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data = p.data - alpha * (m / (v.sqrt() + eps))
                p.data = p.data - lr * weight_decay * p.data
                statep["t"] = t + 1
                statep["m"] = m
                statep["v"] = v
        return loss
