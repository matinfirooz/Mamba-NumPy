"""Small neural-network primitives. Every derivative is implemented explicitly.

Only one retained forward can be outstanding per module. A backward consumes
that cache. Inference (retain=False) does not replace a pending training cache.
"""

import numpy as np


class Parameter:
    def __init__(self, data: np.ndarray, *, decay: bool = True):
        self.data = np.asarray(data)
        self.grad = np.zeros_like(self.data)
        self.decay = decay


def sigmoid(x):
    # exp(-logaddexp(0, -x)) avoids exponential overflow at either extreme.
    return np.exp(-np.logaddexp(0, -x))


def silu(x):
    return x * sigmoid(x)


def silu_backward(x, grad):
    s = sigmoid(x)
    return grad * (s + x * s * (1 - s))


def softplus(x):
    return np.logaddexp(0, x)


class Linear:
    def __init__(self, in_dim, out_dim, rng, dtype, *, bias=False, scale=1.0):
        self.weight = Parameter((rng.standard_normal((in_dim, out_dim))
                                 * (scale / np.sqrt(in_dim))).astype(dtype))
        self.bias = Parameter(np.zeros(out_dim, dtype=dtype), decay=False) if bias else None
        self._cache = None

    def named_parameters(self, prefix=""):
        yield prefix + "weight", self.weight
        if self.bias is not None:
            yield prefix + "bias", self.bias

    def forward(self, x, *, retain=False):
        if retain:
            self._cache = x
        out = x @ self.weight.data
        if self.bias is not None:
            out = out + self.bias.data
        return out

    def backward(self, grad):
        if self._cache is None:
            raise RuntimeError("Linear.backward needs a retained forward")
        x, self._cache = self._cache, None
        self.weight.grad += x.reshape(-1, x.shape[-1]).T @ grad.reshape(-1, grad.shape[-1])
        if self.bias is not None:
            self.bias.grad += grad.sum(axis=tuple(range(grad.ndim - 1)))
        return grad @ self.weight.data.T


class RMSNorm:
    def __init__(self, dim, dtype, eps=1e-5):
        self.weight = Parameter(np.ones(dim, dtype=dtype), decay=False)
        self.eps = eps
        self._cache = None

    def named_parameters(self, prefix=""):
        yield prefix + "weight", self.weight

    def forward(self, x, *, retain=False):
        inv = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        if retain:
            self._cache = (x, inv)
        return x * inv * self.weight.data

    def backward(self, grad):
        if self._cache is None:
            raise RuntimeError("RMSNorm.backward needs a retained forward")
        (x, inv), self._cache = self._cache, None
        self.weight.grad += np.sum(grad * x * inv, axis=tuple(range(grad.ndim - 1)))
        g = grad * self.weight.data
        return g * inv - x * inv**3 * np.mean(g * x, axis=-1, keepdims=True)


class CausalDepthwiseConv1D:
    """Independent channel filters; weight[0] is the oldest tap.

    x has shape (batch, time, channels). The current input uses weight[-1].
    The streaming state holds exactly kernel_size-1 previous raw inputs.
    """

    def __init__(self, channels, kernel_size, rng, dtype):
        self.kernel_size = kernel_size
        self.weight = Parameter((rng.standard_normal((kernel_size, channels))
                                 / np.sqrt(kernel_size)).astype(dtype))
        self.bias = Parameter(np.zeros(channels, dtype=dtype), decay=False)
        self._cache = None

    def named_parameters(self, prefix=""):
        yield prefix + "weight", self.weight
        yield prefix + "bias", self.bias

    def forward(self, x, *, retain=False):
        if retain:
            self._cache = x
        length = x.shape[1]
        out = np.zeros_like(x) + self.bias.data
        for lag in range(min(self.kernel_size, length)):
            out[:, lag:] += x[:, :length - lag] * self.weight.data[-1 - lag]
        return out

    def backward(self, grad):
        if self._cache is None:
            raise RuntimeError("convolution backward needs a retained forward")
        x, self._cache = self._cache, None
        length = x.shape[1]
        dx = np.zeros_like(x)
        self.bias.grad += grad.sum(axis=(0, 1))
        for lag in range(min(self.kernel_size, length)):
            g = grad[:, lag:]
            dx[:, :length - lag] += g * self.weight.data[-1 - lag]
            self.weight.grad[-1 - lag] += np.sum(g * x[:, :length - lag], axis=(0, 1))
        return dx

    def step(self, x, history):
        expected = (x.shape[0], self.kernel_size - 1, x.shape[-1])
        if history.shape != expected:
            raise ValueError(f"convolution state must have shape {expected}")
        window = np.concatenate((history, x[:, None, :]), axis=1)
        out = np.sum(window * self.weight.data[None, :, :], axis=1) + self.bias.data
        history[...] = window[:, 1:]
        return out


def cross_entropy(logits, targets, *, ignore_index=-100):
    """Mean token loss and dlogits; ignored targets contribute no gradient.

    logits: (..., vocabulary); targets: matching leading dimensions, integers.
    """
    if logits.shape[:-1] != targets.shape or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("integer targets must match the leading dimensions of logits")
    v = logits.shape[-1]
    flat = logits.reshape(-1, v)
    labels = targets.reshape(-1)
    mask = labels != ignore_index
    n = int(mask.sum())
    if n == 0:
        raise ValueError("cross entropy needs at least one non-ignored target")
    selected = labels[mask]
    if np.any(selected < 0) or np.any(selected >= v):
        raise ValueError("target outside the vocabulary")
    shifted = flat[mask] - flat[mask].max(axis=-1, keepdims=True)
    log_denom = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - log_denom
    loss = -float(np.sum(log_probs[np.arange(n), selected], dtype=np.float64) / n)
    probs = np.exp(log_probs)
    probs[np.arange(n), selected] -= 1
    grad = np.zeros_like(flat)
    grad[mask] = probs / n
    return loss, grad.reshape(logits.shape)
