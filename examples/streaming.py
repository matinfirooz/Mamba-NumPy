"""Prove chunk continuation and full inference produce equivalent logits."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from mamba_numpy import MambaConfig, MambaLM

model = MambaLM(MambaConfig(vocab_size=32, d_model=16, n_layers=2, dtype="float64"))
tokens = np.random.default_rng(1).integers(0, 32, (2, 37))
full = model.forward(tokens)
left, state = model.prefill(tokens[:, :11])
right, state = model.prefill(tokens[:, 11:], state)
chunked = np.concatenate((left, right), axis=1)
np.testing.assert_allclose(full, chunked, atol=1e-12, rtol=1e-12)
print("maximum full/chunked error:", np.max(np.abs(full - chunked)))
print("actual recurrent state bytes:", sum(s.nbytes for s in state))
print("state formula:", model.state_bytes(batch_size=2))
print("copy state to branch a continuation:", len([s.copy() for s in state]), "layers")
