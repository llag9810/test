import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from collections.abc import Callable, Iterable
from typing import Optional
import typing
import math
import numpy as np
import random
import os

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    right_label_score = inputs[torch.arange(inputs.shape[0]), targets]
    max_value = inputs.max(dim=-1, keepdim=True).values
    inputs_minus_max = inputs - max_value
    log_exp_sum = torch.log(torch.exp(inputs_minus_max).sum(dim=-1))
    result = -right_label_score + max_value.squeeze() + log_exp_sum
    return result.mean(dim=0)

class AdamW(Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        defaults = {"alpha": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "lam": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lam = group["lam"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros(size=p.shape))
                v = state.get("v", torch.zeros(size=p.shape))
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.pow(grad, 2)
                alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data -= alpha * lam * p.data
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss
    
def lr_cosine_schedule(t, alpha_max, alpha_min, t_w, t_c):
    if t < t_w:
        return t * alpha_max / t_w
    elif t <= t_c:
        return alpha_min + (1 + math.cos((t - t_w) * math.pi / (t_c - t_w))) * (alpha_max - alpha_min) / 2
    else:
        return alpha_min
    
def gradient_clipping(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    norm = 0.0

    for g in grads:
        norm += (g**2).sum()

    norm = torch.sqrt(norm)
    clip_coef = min(1, max_norm / (norm + 1e-6))
    for g in grads:
        g *= clip_coef
        
def get_batch(dataset, batch_size, context_length, device):
    x = torch.empty(batch_size, context_length, dtype=torch.long, device=device)
    y = torch.empty(batch_size, context_length, dtype=torch.long, device=device)

    for i in range(batch_size):
        start_idx = random.randint(0, len(dataset) - context_length - 1)
        chunk = torch.tensor(dataset[start_idx : start_idx + context_length + 1], dtype=torch.long)
        x[i] = chunk[:-1]
        y[i] = chunk[1:]

    return x, y

def save_checkpoint(model: nn.Module, optimizer: Optimizer, iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)
    
def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: nn.Module, optimizer: Optimizer) -> int:
    
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]