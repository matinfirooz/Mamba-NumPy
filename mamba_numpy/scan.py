"""Selective recurrence, associative prefix demo, and analytic reverse scan.

Layout: u/delta=(B,L,E), A=(E,N), B/C=(B,L,N), D=(E,).
The variable named B is the input-dependent write vector, not batch size.
"""

from dataclasses import dataclass
import numpy as np


def affine_prefix(a, b, initial_state=None):
    """Inclusive Hillis-Steele scan of affine maps h -> a*h+b.

    a,b have shape (batch, length, ...). Returns all recurrent states.
    Composition: (a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1).
    This vectorized teaching algorithm uses O(L log L) work and O(L) storage.
    It is NOT the work-efficient fused scan used by the official CUDA kernels.
    """
    if a.shape != b.shape or a.ndim < 2 or a.shape[1] < 1:
        raise ValueError("a,b must have identical (batch, nonempty time, ...) shapes")
    pa, pb = a.copy(), b.copy()
    offset = 1
    while offset < a.shape[1]:
        # Compute both right-hand sides before modifying the current level.
        right_a = pa[:, offset:].copy()
        next_b = pb[:, offset:] + right_a * pb[:, :-offset]
        next_a = right_a * pa[:, :-offset]
        pa[:, offset:], pb[:, offset:] = next_a, next_b
        offset *= 2
    if initial_state is not None:
        expected = (a.shape[0],) + a.shape[2:]
        if initial_state.shape != expected:
            raise ValueError(f"initial_state must have shape {expected}")
        pb += pa * initial_state[:, None]
    return pb


@dataclass
class ScanCache:
    u: np.ndarray
    delta: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    states: np.ndarray  # includes initial state at time zero


def selective_scan(u, delta, A, B, C, D, *, initial_state=None,
                   retain=False, mode="sequential"):
    """Mamba-1 real diagonal scan; returns (output, final_state, cache_or_None).

    a_t = exp(delta_t * A)
    h_t = a_t * h_(t-1) + delta_t * B_t * u_t
    y_t = sum_n(C_t * h_t) + D * u_t

    A uses the exponential discretization; the input term uses delta*B*u,
    following the official reference implementation's simplified input rule.
    """
    if u.ndim != 3 or u.shape[1] < 1 or A.ndim != 2:
        raise ValueError("u must be (batch, nonempty time, channels); A must be (channels, state)")
    batch, length, channels = u.shape
    n = A.shape[1]
    if (delta.shape != u.shape or A.shape[0] != channels
            or B.shape != (batch, length, n) or C.shape != B.shape or D.shape != (channels,)):
        raise ValueError("inconsistent selective scan shapes")
    dtype = np.result_type(u, delta, A, B, C, D)
    h = np.zeros((batch, channels, n), dtype=dtype)
    if initial_state is not None:
        if initial_state.shape != h.shape:
            raise ValueError(f"initial_state must have shape {h.shape}")
        h[...] = initial_state
    states = None
    if retain:
        states = np.empty((batch, length + 1, channels, n), dtype=dtype)
        states[:, 0] = h
    if mode == "sequential":
        y = np.empty(u.shape, dtype=dtype)
        for t in range(length):
            a = np.exp(delta[:, t, :, None] * A[None])
            h = a * h + (delta[:, t] * u[:, t])[:, :, None] * B[:, t, None, :]
            y[:, t] = np.sum(h * C[:, t, None, :], axis=-1) + D * u[:, t]
            if retain:
                states[:, t + 1] = h
    elif mode == "parallel":
        a = np.exp(delta[..., None] * A[None, None])
        b = (delta * u)[..., None] * B[:, :, None, :]
        all_h = affine_prefix(a, b, h)
        y = np.sum(all_h * C[:, :, None, :], axis=-1) + u * D
        h = all_h[:, -1].copy()
        if retain:
            states[:, 1:] = all_h
    else:
        raise ValueError("mode must be sequential or parallel")
    cache = ScanCache(u, delta, A, B, C, D, states) if retain else None
    return y, h, cache


def selective_scan_backward(grad_y, cache, grad_last=None):
    """Reverse-time analytic differentiation, including initial/final state."""
    u, delta, A, B, C, D, states = (
        cache.u, cache.delta, cache.A, cache.B, cache.C, cache.D, cache.states
    )
    if grad_y.shape != u.shape:
        raise ValueError("grad_y must have the same shape as u")
    du, ddelta = np.zeros_like(u), np.zeros_like(delta)
    dA, dB, dC, dD = (np.zeros_like(x) for x in (A, B, C, D))
    carry = np.zeros_like(states[:, 0])
    if grad_last is not None:
        if grad_last.shape != carry.shape:
            raise ValueError("grad_last has the wrong shape")
        carry[...] = grad_last
    for t in range(u.shape[1] - 1, -1, -1):
        dt = delta[:, t, :, None]
        ut = u[:, t, :, None]
        bt = B[:, t, None, :]
        a = np.exp(dt * A[None])
        gy = grad_y[:, t]
        dh = carry + gy[:, :, None] * C[:, t, None, :]
        dC[:, t] = np.sum(gy[:, :, None] * states[:, t + 1], axis=1)
        dD += np.sum(gy * u[:, t], axis=0)
        du[:, t] = gy * D + np.sum(dh * dt * bt, axis=-1)
        dB[:, t] = np.sum(dh * dt * ut, axis=1)
        ddelta[:, t] = np.sum(dh * (a * states[:, t] * A[None] + bt * ut), axis=-1)
        dA += np.sum(dh * a * states[:, t] * dt, axis=0)
        carry = dh * a
    return {"u": du, "delta": ddelta, "A": dA, "B": dB,
            "C": dC, "D": dD, "initial_state": carry}
