"""Held-out recall at several delays, plus a state-readout ablation.

The ablation zeros the C projection in every trained block at inference. It is
a diagnostic of dependence on the state path, not a retrained competitive model.
"""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mamba_numpy.checkpoint import load_checkpoint
from mamba_numpy.training import TrainConfig, TaskData, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/recall_evaluation"))
    p.add_argument("--batches", type=int, default=16)
    args = p.parse_args()
    if args.batches < 1:
        p.error("batches must be positive")
    ckpt = load_checkpoint(args.checkpoint)
    base = TrainConfig(**ckpt.metadata["train_config"])
    if base.task != "recall":
        p.error("this command requires a selective-recall checkpoint")
    model = ckpt.model
    rows = []
    original = [b.mixer.x_proj.weight.data.copy() for b in model.blocks]
    for mode in ("trained", "state_readout_disabled"):
        try:
            if mode == "state_readout_disabled":
                c = model.config
                for b in model.blocks:
                    b.mixer.x_proj.weight.data[:, c.resolved_dt_rank + c.d_state:] = 0
            for seq_len, low, high in ((32, 4, 9), (64, 12, 20), (128, 32, 48), (256, 64, 96)):
                config = replace(base, seq_len=seq_len, min_delay=low, max_delay=high)
                metrics = evaluate(model, TaskData(config), batches=args.batches, seed=20261023)
                row = {"mode": mode, "seq_len": seq_len, "min_delay": low,
                       "max_delay": high, "loss": metrics["loss"],
                       "query_accuracy": metrics["accuracy"], "queries": metrics["tokens"]}
                rows.append(row)
                print(f"{mode:24s} delay {low:2d}..{high:2d} | "
                      f"accuracy {metrics['accuracy']:.2%} | {metrics['tokens']} queries", flush=True)
        finally:
            for b, value in zip(model.blocks, original):
                b.mixer.x_proj.weight.data[...] = value
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "delay_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"seed": 20261023, "batches": args.batches, "n_values": base.n_values,
                "chance_if_selecting_uniformly_among_values": 1 / base.n_values,
                "training_delay_range": [base.min_delay, base.max_delay],
                "max_local_receptive_field_tokens": 1 + model.config.n_layers * (model.config.d_conv - 1),
                "ablation": "zeros C-projection columns in every trained block; no retraining",
                "scope": "small procedural task; one training seed; not the original Mamba copy benchmark"}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
