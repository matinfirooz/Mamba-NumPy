"""Measure full, prefix-scan, and streaming inference on the SAME model."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mamba_numpy import MambaConfig, MambaLM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/benchmark"))
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--d-model", type=int, default=32)
    args = p.parse_args()
    if args.repeats < 1 or args.batch_size < 1 or any(l < 1 for l in args.lengths):
        p.error("lengths, batch size, repeats must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = MambaConfig(vocab_size=32, d_model=args.d_model, n_layers=2, d_state=8)
    model = MambaLM(cfg)
    rng = np.random.default_rng(91)
    rows = []
    for length in args.lengths:
        ids = rng.integers(0, cfg.vocab_size, (args.batch_size, length))
        reference = model.forward(ids)
        functions = {"sequential": lambda: model.forward(ids),
                     "parallel_demo": lambda: model.forward(ids, scan_mode="parallel"),
                     "streaming": lambda: model.prefill(ids)[0]}
        for name, function in functions.items():
            result = function()  # warm up; includes cache allocation for streaming
            error = float(np.max(np.abs(result - reference)))
            np.testing.assert_allclose(result, reference, atol=5e-6, rtol=5e-5)
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                function()
                times.append(time.perf_counter() - start)
            median = statistics.median(times)
            row = {"length": length, "method": name, "batch_size": args.batch_size,
                   "median_ms": median * 1000, "tokens_per_second": args.batch_size * length / median,
                   "max_abs_error": error, "recurrent_state_bytes": model.state_bytes(args.batch_size),
                   "training_ssm_history_bytes": args.batch_size * cfg.n_layers * (length + 1)
                       * cfg.d_inner * cfg.d_state * cfg.np_dtype.itemsize}
            rows.append(row)
            print(f"L={length:4d} {name:14s} {median*1000:9.3f} ms | max error {error:.2e}")
    with (args.out / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "platform": platform.platform(), "machine": platform.machine(),
                   "processor": platform.processor(), "model_config": cfg.to_dict(),
                   "repeats": args.repeats, "warmup_runs": 1,
                   "thread_environment": {k: os.environ.get(k) for k in
                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                   "timing_scope": "end-to-end chunk inference including logits and state allocation; CPU wall clock",
                   "memory_scope": "analytic array bytes for recurrent state / saved SSM history only; not peak RSS",
                   "parallel_demo": "Hillis-Steele, O(L log L) work; not the official fused work-efficient scan"}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                environment["processor"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    (args.out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
