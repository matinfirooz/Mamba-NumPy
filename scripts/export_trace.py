"""Export the first layer's floating-point SSM operands and golden states.

Intended for studying a software-to-hardware mapping. This is NOT a quantized
implementation or an integer arithmetic accuracy/energy evaluation.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mamba_numpy.checkpoint import load_checkpoint
from mamba_numpy.nn import silu, softplus
from mamba_numpy.scan import selective_scan


def export_trace(model, ids, output):
    ids = model._validate_ids(np.asarray(ids), 2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    block = model.blocks[0]
    mixer = block.mixer
    x = block.norm.forward(model.embedding.data[ids])
    raw, z = np.split(mixer.in_proj.forward(x), 2, axis=-1)
    u = silu(mixer.conv.forward(raw))
    c = model.config
    low_dt, B, C = np.split(mixer.x_proj.forward(u),
                             [c.resolved_dt_rank, c.resolved_dt_rank + c.d_state], axis=-1)
    delta = softplus(mixer.dt_proj.forward(low_dt))
    A = -np.exp(mixer.A_log.data)
    y, last, cache = selective_scan(u, delta, A, B, C, mixer.D.data, retain=True)
    np.savez_compressed(output / "trace.npz", token_ids=ids, u=u, delta=delta, A=A,
                        B=B, C=C, D=mixer.D.data, z=z, states=cache.states,
                        scan_output=y, final_state=last,
                        mixer_output=mixer.out_proj.forward(y * silu(z)))
    with (output / "channel0.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "token_id", "u_0", "delta_0", "state_0_0", "scan_y_0"])
        for t in range(ids.shape[1]):
            writer.writerow([t, ids[0, t], u[0, t, 0], delta[0, t, 0],
                             cache.states[0, t + 1, 0, 0], y[0, t, 0]])
    metadata = {"model_config": c.to_dict(), "layer": 0, "dtype": c.dtype,
                "note": "floating-point reference operands; no quantization or hardware claims",
                "shapes": {"u/delta": list(u.shape), "A": list(A.shape), "B/C": list(B.shape),
                           "D": list(mixer.D.data.shape), "states": list(cache.states.shape)},
                "state_index": "states[:,0] is zero initial state; states[:,t+1] follows input t"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output / "trace.npz"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--text", default="mamba reads a stream")
    p.add_argument("--out", type=Path, default=Path("results/trace"))
    args = p.parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    if ckpt.tokenizer is None:
        p.error("export this text trace from a character-model checkpoint")
    print(export_trace(ckpt.model, ckpt.tokenizer.encode(args.text)[None], args.out))


if __name__ == "__main__":
    main()
