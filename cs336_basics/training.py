import torch
import numpy as np
import einx


@torch.compile
def cross_entropy_loss(logits, targets, dim=-1):
    logits = logits - torch.max(logits, dim=dim, keepdim=True)[0]
    log_exp_sum = torch.sum(torch.exp(logits), dim=dim, keepdim=True)
    logits = torch.gather(logits, dim=dim, index=targets.unsqueeze(dim))
    return torch.mean(-logits + torch.log(log_exp_sum))


def perplexity(losses):
    return np.exp(losses.mean())
