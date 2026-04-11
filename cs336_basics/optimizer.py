from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class Optimizer(torch.optim.Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, {"lr": lr})

    def set(self, closure: Callable[[], float] | None = None) -> float | None:
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
