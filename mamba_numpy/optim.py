"""AdamW and global-norm clipping, implemented with NumPy arrays."""

import math
import numpy as np


def clip_grad_norm(parameters, max_norm=1.0):
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be positive and finite")
    params = list(parameters)
    total = math.sqrt(sum(float(np.sum(p.grad.astype(np.float64)**2)) for p in params))
    if not math.isfinite(total):
        raise FloatingPointError("nonfinite gradients; optimizer update was not performed")
    if total > max_norm:
        factor = max_norm / total
        for p in params:
            p.grad *= factor
    return total


class AdamW:
    def __init__(self, named_parameters, lr=0.003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("lr and eps must be positive and finite")
        if len(betas) != 2 or not all(0 <= b < 1 for b in betas):
            raise ValueError("Adam betas must be in [0,1)")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")
        items = list(named_parameters)
        self.params = dict(items)
        if len(self.params) != len(items):
            raise ValueError("parameter names must be unique")
        if len({id(p) for p in self.params.values()}) != len(self.params):
            raise ValueError("provide each Parameter exactly once")
        self.lr, self.betas, self.eps, self.weight_decay = lr, tuple(betas), eps, weight_decay
        self.t = 0
        self.m = {name: np.zeros_like(p.data) for name, p in self.params.items()}
        self.v = {name: np.zeros_like(p.data) for name, p in self.params.items()}

    def step(self, *, lr=None):
        rate = self.lr if lr is None else lr
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("learning rate must be positive and finite")
        if any(not np.all(np.isfinite(p.grad)) for p in self.params.values()):
            raise FloatingPointError("nonfinite gradients; no parameters were changed")
        self.t += 1
        b1, b2 = self.betas
        for name, p in self.params.items():
            m, v = self.m[name], self.v[name]
            m *= b1
            m += (1 - b1) * p.grad
            v *= b2
            v += (1 - b2) * p.grad**2
            if p.decay:
                p.data *= 1 - rate * self.weight_decay
            m_hat, v_hat = m / (1 - b1**self.t), v / (1 - b2**self.t)
            p.data -= rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def hyperparameters(self):
        return {"lr": self.lr, "betas": list(self.betas), "eps": self.eps,
                "weight_decay": self.weight_decay}
