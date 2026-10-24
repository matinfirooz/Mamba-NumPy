"""Mamba-1 mixer, residual stack, language model, and recurrent inference."""

from dataclasses import dataclass
import numpy as np

from .config import MambaConfig
from .nn import (Parameter, Linear, RMSNorm, CausalDepthwiseConv1D,
                 silu, silu_backward, softplus, sigmoid, cross_entropy)
from .scan import selective_scan, selective_scan_backward


@dataclass
class LayerState:
    conv: np.ndarray  # (batch, K-1, E), oldest -> newest raw input
    ssm: np.ndarray   # (batch, E, N)

    def copy(self):
        return LayerState(self.conv.copy(), self.ssm.copy())

    @property
    def nbytes(self):
        return self.conv.nbytes + self.ssm.nbytes


class MambaMixer:
    def __init__(self, config: MambaConfig, rng=None):
        self.config = config
        rng = np.random.default_rng(config.seed) if rng is None else rng
        e, n, r = config.d_inner, config.d_state, config.resolved_dt_rank
        dtype = config.np_dtype
        self.in_proj = Linear(config.d_model, 2 * e, rng, dtype)
        self.conv = CausalDepthwiseConv1D(e, config.d_conv, rng, dtype)
        self.x_proj = Linear(e, r + 2 * n, rng, dtype)
        self.dt_proj = Linear(r, e, rng, dtype, bias=True)
        self.dt_proj.weight.data[...] = rng.uniform(-r**-0.5, r**-0.5, (r, e))
        dt = np.exp(rng.uniform(np.log(config.dt_min), np.log(config.dt_max), e))
        self.dt_proj.bias.data[...] = dt + np.log(-np.expm1(-dt))
        self.A_log = Parameter(np.broadcast_to(
            np.log(np.arange(1, n + 1, dtype=dtype)), (e, n)).copy(), decay=False)
        self.D = Parameter(np.ones(e, dtype=dtype), decay=False)
        self.out_proj = Linear(e, config.d_model, rng, dtype,
                               scale=config.n_layers**-0.5)
        self._cache = None

    def named_parameters(self, prefix=""):
        for name in ("in_proj", "conv", "x_proj", "dt_proj", "out_proj"):
            yield from getattr(self, name).named_parameters(prefix + name + ".")
        yield prefix + "A_log", self.A_log
        yield prefix + "D", self.D

    def forward(self, x, *, retain=False, scan_mode="sequential"):
        xz = self.in_proj.forward(x, retain=retain)
        raw, z = np.split(xz, 2, axis=-1)
        conv = self.conv.forward(raw, retain=retain)
        u = silu(conv)
        projected = self.x_proj.forward(u, retain=retain)
        r, n = self.config.resolved_dt_rank, self.config.d_state
        low_dt, B, C = np.split(projected, [r, r + n], axis=-1)
        pre_dt = self.dt_proj.forward(low_dt, retain=retain)
        delta = softplus(pre_dt)
        A = -np.exp(self.A_log.data)
        y, _, scan_cache = selective_scan(u, delta, A, B, C, self.D.data,
                                          retain=retain, mode=scan_mode)
        out = self.out_proj.forward(y * silu(z), retain=retain)
        if retain:
            self._cache = (z, conv, pre_dt, A, y, scan_cache)
        return out

    def backward(self, grad):
        if self._cache is None:
            raise RuntimeError("MambaMixer.backward needs a retained forward")
        (z, conv, pre_dt, A, y, scan_cache), self._cache = self._cache, None
        dgated = self.out_proj.backward(grad)
        dz = silu_backward(z, dgated * y)
        gs = selective_scan_backward(dgated * silu(z), scan_cache)
        self.A_log.grad += gs["A"] * A  # derivative of -exp(A_log) is A
        self.D.grad += gs["D"]
        dlow_dt = self.dt_proj.backward(gs["delta"] * sigmoid(pre_dt))
        du = gs["u"] + self.x_proj.backward(np.concatenate((dlow_dt, gs["B"], gs["C"]), axis=-1))
        draw = self.conv.backward(silu_backward(conv, du))
        return self.in_proj.backward(np.concatenate((draw, dz), axis=-1))

    def init_state(self, batch_size):
        c = self.config
        return LayerState(np.zeros((batch_size, c.d_conv - 1, c.d_inner), dtype=c.np_dtype),
                          np.zeros((batch_size, c.d_inner, c.d_state), dtype=c.np_dtype))

    def step(self, x, state: LayerState):
        """Consume one (B,D) token; mutate caller-owned inference state."""
        expected = (x.shape[0], self.config.d_inner, self.config.d_state)
        if state.ssm.shape != expected:
            raise ValueError(f"SSM state must have shape {expected}")
        raw, z = np.split(self.in_proj.forward(x), 2, axis=-1)
        u = silu(self.conv.step(raw, state.conv))
        r, n = self.config.resolved_dt_rank, self.config.d_state
        low_dt, B, C = np.split(self.x_proj.forward(u), [r, r + n], axis=-1)
        delta = softplus(self.dt_proj.forward(low_dt))
        A = -np.exp(self.A_log.data)
        state.ssm[...] = (np.exp(delta[..., None] * A[None]) * state.ssm
                          + (delta * u)[..., None] * B[:, None, :])
        y = np.sum(state.ssm * C[:, None, :], axis=-1) + self.D.data * u
        return self.out_proj.forward(y * silu(z))


class ResidualBlock:
    def __init__(self, config, rng):
        self.norm = RMSNorm(config.d_model, config.np_dtype, config.norm_eps)
        self.mixer = MambaMixer(config, rng)

    def named_parameters(self, prefix=""):
        yield from self.norm.named_parameters(prefix + "norm.")
        yield from self.mixer.named_parameters(prefix + "mixer.")

    def forward(self, x, *, retain=False, scan_mode="sequential"):
        return x + self.mixer.forward(self.norm.forward(x, retain=retain),
                                      retain=retain, scan_mode=scan_mode)

    def backward(self, grad):
        return grad + self.norm.backward(self.mixer.backward(grad))

    def step(self, x, state):
        return x + self.mixer.step(self.norm.forward(x), state)


class MambaLM:
    """Causal token-to-logits model with optional tied embedding/head weights."""

    def __init__(self, config: MambaConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.embedding = Parameter((rng.standard_normal((config.vocab_size, config.d_model))
                                    * 0.2).astype(config.np_dtype))
        self.blocks = [ResidualBlock(config, rng) for _ in range(config.n_layers)]
        self.norm = RMSNorm(config.d_model, config.np_dtype, config.norm_eps)
        self.head = None if config.tie_embeddings else Linear(
            config.d_model, config.vocab_size, rng, config.np_dtype)
        self._cache = None

    def named_parameters(self):
        # Tied weights appear once; gradients accumulate from both uses.
        yield "embedding.weight", self.embedding
        for i, block in enumerate(self.blocks):
            yield from block.named_parameters(f"blocks.{i}.")
        yield from self.norm.named_parameters("norm.")
        if self.head is not None:
            yield from self.head.named_parameters("head.")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0)

    @property
    def num_parameters(self):
        return sum(p.data.size for p in self.parameters())

    def _validate_ids(self, ids, ndim):
        ids = np.asarray(ids)
        if ids.ndim != ndim or any(s == 0 for s in ids.shape):
            raise ValueError(f"token IDs must be a nonempty {ndim}-dimensional array")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("token IDs must be integers")
        if np.any(ids < 0) or np.any(ids >= self.config.vocab_size):
            raise ValueError("token ID outside the vocabulary")
        return ids

    def forward(self, ids, *, retain=False, scan_mode="sequential"):
        ids = self._validate_ids(ids, 2)
        if retain and self._cache is not None:
            raise RuntimeError("consume backward before another retained forward")
        x = self.embedding.data[ids]
        for block in self.blocks:
            x = block.forward(x, retain=retain, scan_mode=scan_mode)
        hidden = self.norm.forward(x, retain=retain)
        if self.head is None:
            logits = hidden @ self.embedding.data.T
        else:
            logits = self.head.forward(hidden, retain=retain)
        if retain:
            self._cache = (ids.copy(), hidden)
        return logits

    def backward(self, grad_logits):
        if self._cache is None:
            raise RuntimeError("MambaLM.backward needs forward(retain=True)")
        ids, hidden = self._cache
        if grad_logits.shape != (*ids.shape, self.config.vocab_size):
            raise ValueError("logit gradient has the wrong shape")
        self._cache = None
        if self.head is None:
            self.embedding.grad += (grad_logits.reshape(-1, self.config.vocab_size).T
                                     @ hidden.reshape(-1, self.config.d_model))
            grad = grad_logits @ self.embedding.data
        else:
            grad = self.head.backward(grad_logits)
        grad = self.norm.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        np.add.at(self.embedding.grad, ids, grad)

    def loss_and_grad(self, ids, targets):
        ids = self._validate_ids(ids, 2)
        targets = np.asarray(targets)
        if targets.shape != ids.shape or not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("integer targets must have the same shape as token IDs")
        mask = targets != -100
        if not np.any(mask) or np.any(targets[mask] < 0) or np.any(targets[mask] >= self.config.vocab_size):
            raise ValueError("need at least one non-ignored target, with all target IDs inside the vocabulary")
        self.zero_grad()
        logits = self.forward(ids, retain=True)
        loss, grad = cross_entropy(logits, targets)
        self.backward(grad)
        return loss

    def init_state(self, batch_size=1):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return [block.mixer.init_state(batch_size) for block in self.blocks]

    def step(self, ids, state):
        ids = self._validate_ids(ids, 1)
        if len(state) != len(self.blocks):
            raise ValueError("one inference state is required per block")
        x = self.embedding.data[ids]
        for block, layer_state in zip(self.blocks, state):
            x = block.step(x, layer_state)
        x = self.norm.forward(x)
        logits = x @ self.embedding.data.T if self.head is None else self.head.forward(x)
        return logits, state

    def prefill(self, ids, state=None):
        """Consume a prompt or chunk using bounded recurrent workspace.

        Returns all chunk logits and the updated state. Supplying the returned
        state to another prefill continues the sequence without resetting it.
        """
        ids = self._validate_ids(ids, 2)
        state = self.init_state(ids.shape[0]) if state is None else state
        logits = []
        for t in range(ids.shape[1]):
            out, state = self.step(ids[:, t], state)
            logits.append(out)
        return np.stack(logits, axis=1), state

    def state_bytes(self, batch_size=1):
        c = self.config
        return batch_size * c.n_layers * c.d_inner * (c.d_state + c.d_conv - 1) * c.np_dtype.itemsize
