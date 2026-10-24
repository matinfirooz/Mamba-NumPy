"""Text windows and a procedural selective-recall task."""

import numpy as np


def text_batch(tokens, batch_size, seq_len, rng):
    if batch_size < 1 or seq_len < 1 or len(tokens) <= seq_len:
        raise ValueError("need positive batch/length and more than seq_len corpus tokens")
    starts = rng.integers(0, len(tokens) - seq_len, size=batch_size)
    windows = tokens[starts[:, None] + np.arange(seq_len + 1)[None]]
    return windows[:, :-1].copy(), windows[:, 1:].copy()


def selective_recall_batch(batch_size, seq_len, rng, *, n_values=4, min_delay=4, max_delay=9):
    """Store selected values, ignore distractors, answer only at query tokens.

    Tokens 1..V write a value. V+1..2V are distractors; 2V+1 queries the most
    recent write. Targets are -100 everywhere except queries, where they are
    the selected value. Delays count intervening distractors. min_delay=4
    places the write outside a four-tap convolution's receptive field.
    Token zero is reserved; vocabulary size is 2*V+2. Values never appear as
    distractors. This is selective recall, not the paper's full copy benchmark.
    """
    if (batch_size < 1 or n_values < 1 or min_delay < 0 or max_delay < min_delay
            or seq_len < min_delay + 2):
        raise ValueError("invalid selective-recall sizes or delay range")
    x = rng.integers(n_values + 1, 2 * n_values + 1, (batch_size, seq_len))
    y = np.full_like(x, -100)
    query = 2 * n_values + 1
    for b in range(batch_size):
        position = 0
        while position + min_delay + 2 <= seq_len:
            max_here = min(max_delay, seq_len - position - 2)
            delay = int(rng.integers(min_delay, max_here + 1))
            selected = int(rng.integers(1, n_values + 1))
            x[b, position] = selected
            position += delay + 1
            x[b, position] = query
            y[b, position] = selected
            position += 1
    return x.astype(np.int64), y.astype(np.int64)
