# Python API

Run commands from the repository root or install the package with
`python -m pip install -e .`. Python's standard library plus NumPy are sufficient.

## Model

```python
import numpy as np
from mamba_numpy import MambaConfig, MambaLM

config = MambaConfig(vocab_size=32, d_model=32, n_layers=2,
                     d_state=8, d_conv=4, expand=2, dt_rank=2)
model = MambaLM(config)
ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
logits = model.forward(ids)   # (1, 4, 32)
```

`MambaConfig` is a validated frozen dataclass. `dt_rank=None` resolves to
`ceil(d_model/16)`. The supported dtypes are `float32` and `float64`.
The vocabulary must contain every supplied integer ID. Batch and time
dimensions must be nonempty. There is no positional embedding or fixed
maximum sequence length.

## Training

```python
from mamba_numpy import AdamW, clip_grad_norm

targets = np.array([[2, 3, 4, 5]])
optimizer = AdamW(model.named_parameters(), lr=0.003)
loss = model.loss_and_grad(ids, targets)  # clears gradients first
preclip_norm = clip_grad_norm(model.parameters(), max_norm=1.0)
optimizer.step()
```

Targets equal to `-100` are ignored. The loss averages over non-ignored tokens.
At least one target must be valid. Every parameter has `.data`, `.grad`, and
`.decay` attributes. Biases, RMSNorm weights, `A_log`, and `D` do not decay.
Tied embedding weights appear only once in `named_parameters()`.

For a custom objective:

```python
model.zero_grad()
logits = model.forward(ids, retain=True)
# Compute dlogits with the same shape as logits.
dlogits = np.ones_like(logits) / logits.size
model.backward(dlogits)
```

One retained forward may be outstanding. Backward consumes it. Inference
does not overwrite a retained forward. Parameter updates must occur after
backward; do not mutate inputs or parameters between retained forward/backward.
Full-model training starts with zero recurrent state on each sampled window.
The low-level scan exposes initial/final-state gradients, but the model API
does not implement cross-window truncated BPTT.

## Streaming

```python
state = model.init_state(batch_size=1)
for token in (1, 2, 3, 4):
    current_logits, state = model.step(np.array([token]), state)

first, state = model.prefill(ids[:, :2])
second, state = model.prefill(ids[:, 2:], state)
```

`step` consumes `(B,)` IDs and returns `(B,V)` logits. `prefill` consumes
`(B,L)` IDs and returns `(B,L,V)` logits. Both mutate the supplied state.
`init_state` resets a stream. `[s.copy() for s in state]` makes an independent
branch. Use the same batch layout and model parameters throughout a stream.
`model.state_bytes(B)` counts persistent conv/SSM arrays, excluding logits and weights.

## Generation

```python
from mamba_numpy import generate

tokens = generate(model, np.array([1, 2, 3]), max_new_tokens=20,
                  temperature=0.8, top_k=10, top_p=0.95, seed=42)
```

The result includes the prompt. `temperature=0` is greedy. Top-k is applied
before top-p; the first token crossing the top-p cumulative threshold stays
eligible. `eos_id` can end generation early. The fitted character tokenizer
uses `<unk>` at ID zero and otherwise one Unicode character per token.
For text, load the tokenizer with the checkpoint and use its encode/decode.

## Checkpoints

```python
from mamba_numpy.checkpoint import save_checkpoint, load_checkpoint

save_checkpoint("runs/model.npz", model, optimizer=optimizer,
                metadata={"step": optimizer.t})
ckpt = load_checkpoint("runs/model.npz", load_optimizer=True)
restored_model, restored_optimizer = ckpt.model, ckpt.optimizer
```

Checkpoints contain numeric arrays and a JSON header, loaded with
`allow_pickle=False`. Architecture, parameter shapes/dtypes, array keys,
finite values, and tokenizer size are checked. Saves use a same-directory
temporary file and atomic replace. A model-only save is supported by omitting
the optimizer; requesting an absent optimizer on load raises an error.

The CLI trainer also stores tokenizer, corpus hash, configuration, optimizer
moments/step, and NumPy generator state. Resume preserves the data stream and
warmup schedule. `--steps` is the total update target. Results can still vary
between machines or NumPy/BLAS versions. If moving the repository, pass the
relocated corpus with `--data`; identical contents are required for resume.

## Low-level scan

```python
from mamba_numpy.scan import selective_scan, selective_scan_backward

# See docs/math.md for all shapes. initial_state and grad_last are optional.
y, final_state, cache = selective_scan(u, delta, A, B, C, D,
                                      initial_state=h0, retain=True)
grads = selective_scan_backward(dy, cache, grad_last=dh_final)
```

The gradient dictionary contains `u`, `delta`, `A`, `B`, `C`, `D`, and
`initial_state`. This low-level API expects real floating-point arrays with
the documented shapes. `mode="parallel"` enables the associative teaching
implementation. `MambaLM.forward(..., scan_mode="parallel")` uses it in every
layer. It changes arithmetic order and uses additional workspace.
