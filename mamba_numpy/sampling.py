"""Streaming autoregressive sampling with temperature, top-k, and top-p."""

import math
import numpy as np


def sample_token(logits, rng, *, temperature=1.0, top_k=None, top_p=1.0):
    if logits.ndim != 1 or logits.size == 0 or not np.all(np.isfinite(logits)):
        raise ValueError("sampling expects a finite nonempty logit vector")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be nonnegative and finite (zero = greedy)")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0,1]")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be None or a positive integer")
    if temperature == 0:
        return int(np.argmax(logits))
    values = (logits.astype(np.float64) - np.max(logits)) / temperature
    order = np.argsort(-values, kind="stable")
    if top_k is not None:
        order = order[:min(top_k, logits.size)]
    probs = np.exp(values[order] - np.max(values[order]))
    probs /= probs.sum()
    if top_p < 1:
        # Keep the token crossing the cumulative threshold, including top-1.
        count = int(np.searchsorted(np.cumsum(probs), top_p, side="left")) + 1
        order, probs = order[:count], probs[:count]
        probs /= probs.sum()
    return int(rng.choice(order, p=probs))


def generate(model, prompt, max_new_tokens=100, *, temperature=0.8, top_k=None,
             top_p=1.0, seed=42, eos_id=None):
    prompt = np.asarray(prompt)
    if prompt.ndim != 1 or prompt.size == 0:
        raise ValueError("prompt must be a nonempty one-dimensional integer sequence")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    # Validation remains active even when no output tokens are requested.
    prompt = model._validate_ids(prompt, 1)
    result = prompt.tolist()
    if max_new_tokens == 0:
        return np.asarray(result, dtype=np.int64)
    rng = np.random.default_rng(seed)
    state = model.init_state(1)
    for token in prompt:
        logits, state = model.step(np.array([token]), state)
    for i in range(max_new_tokens):
        token = sample_token(logits[0], rng, temperature=temperature, top_k=top_k, top_p=top_p)
        result.append(token)
        if token == eos_id:
            break
        if i + 1 < max_new_tokens:
            logits, state = model.step(np.array([token]), state)
    return np.asarray(result, dtype=np.int64)
