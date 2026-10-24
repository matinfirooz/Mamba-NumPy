"""A complete trainable Mamba model in a few API calls."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from mamba_numpy import MambaConfig, MambaLM, AdamW, clip_grad_norm, generate

model = MambaLM(MambaConfig(vocab_size=16, d_model=16, n_layers=2, d_state=8))
rng = np.random.default_rng(42)
x = rng.integers(0, 16, (4, 12))
y = (x + 1) % 16  # toy per-token mapping to demonstrate the training API
optimizer = AdamW(model.named_parameters(), lr=0.005)
for step in range(20):
    loss = model.loss_and_grad(x, y)
    clip_grad_norm(model.parameters(), 1.0)
    optimizer.step()
    if step in (0, 9, 19):
        print(f"step={step+1:2d} loss={loss:.4f}")
print("logits:", model.forward(x).shape)
print("parameters:", model.num_parameters)
print("recurrent state per sequence:", model.state_bytes(), "bytes")
print("sampled token IDs:", generate(model, np.array([1, 2, 3]), max_new_tokens=8))
