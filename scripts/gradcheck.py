"""Central finite differences for every parameter tensor and all scan inputs."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mamba_numpy import MambaConfig, MambaLM
from mamba_numpy.nn import cross_entropy
from mamba_numpy.scan import selective_scan, selective_scan_backward


def compare(array, analytic, objective, rng, samples=5, eps=1e-5):
    positions = rng.choice(array.size, size=min(samples, array.size), replace=False)
    rows = []
    for i in positions:
        old = array.flat[i].copy()
        try:
            array.flat[i] = old + eps
            plus = objective()
            array.flat[i] = old - eps
            minus = objective()
        finally:
            array.flat[i] = old
        numerical = (plus - minus) / (2 * eps)
        exact = float(analytic.flat[i])
        error = abs(exact - numerical)
        # The absolute tolerance protects near-zero derivatives from cancellation.
        tolerance = 2e-8 + 2e-5 * max(abs(exact), abs(numerical))
        rows.append({"index": int(i), "analytic": exact, "numerical": numerical,
                     "absolute_error": error, "tolerance": tolerance,
                     "passed": bool(error <= tolerance)})
    return rows


def run_gradcheck(samples=5):
    rng = np.random.default_rng(941)
    model = MambaLM(MambaConfig(vocab_size=7, d_model=4, n_layers=2, d_state=3,
                                expand=2, d_conv=3, dt_rank=2, dtype="float64", seed=7))
    ids = rng.integers(0, 7, (2, 5))
    targets = rng.integers(0, 7, (2, 5))
    targets[0, 1] = -100
    model.loss_and_grad(ids, targets)
    report = []
    for name, p in model.named_parameters():
        rows = compare(p.data, p.grad.copy(),
                       lambda: cross_entropy(model.forward(ids), targets)[0], rng, samples)
        report.append({"name": "model/" + name, "checks": rows})
    # Isolate recurrence derivatives with nonzero incoming and outgoing state.
    b, l, e, n = 2, 4, 3, 2
    arrays = {"u": rng.normal(size=(b, l, e)), "delta": rng.uniform(0.1, 0.8, (b, l, e)),
              "A": -rng.uniform(0.1, 1.0, (e, n)), "B": rng.normal(size=(b, l, n)),
              "C": rng.normal(size=(b, l, n)), "D": rng.normal(size=e),
              "initial_state": rng.normal(size=(b, e, n))}
    gy, gh = rng.normal(size=(b, l, e)), rng.normal(size=(b, e, n))

    def objective():
        y, last, _ = selective_scan(**arrays)
        return float(np.sum(y * gy) + np.sum(last * gh))

    _, _, cache = selective_scan(**arrays, retain=True)
    grads = selective_scan_backward(gy, cache, gh)
    for name, a in arrays.items():
        report.append({"name": "scan/" + name,
                       "checks": compare(a, grads[name], objective, rng, samples)})
    checks = [c for r in report for c in r["checks"]]
    return {"passed": all(c["passed"] for c in checks), "num_checks": len(checks),
            "max_absolute_error": max(c["absolute_error"] for c in checks),
            "method": "central differences, eps=1e-5, float64",
            "tolerance": "abs_error <= 2e-8 + 2e-5 * max(abs(analytic), abs(numerical))",
            "tensors": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    result = run_gradcheck(args.samples)
    print(f"Gradient check: {'PASS' if result['passed'] else 'FAIL'}; "
          f"{result['num_checks']} coordinates; max absolute error {result['max_absolute_error']:.3e}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        for row in result["tensors"]:
            if any(not c["passed"] for c in row["checks"]):
                print(row)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
