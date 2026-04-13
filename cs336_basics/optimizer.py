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
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = closure() if closure else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                statep = self.state[p]
                if len(statep) == 0:
                    statep["t"] = 1
                    statep["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    statep["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                t = statep["t"]
                m = statep["m"]
                v = statep["v"]
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                grad = p.grad

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                alpha = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.addcdiv_(m, v.sqrt() + eps, value=-alpha)
                p.mul_(1 - lr * weight_decay)
                statep["t"] = t + 1
        return loss
