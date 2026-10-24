# Contributing

Keep numerical computation in NumPy and utilities in the Python standard library.
Explain shape conventions and any changes to the recurrence in the documentation.

Before opening a pull request, run:

```bash
python -m unittest discover -s tests -v
python scripts/gradcheck.py --samples 6
python examples/streaming.py
```

A new operator needs a meaningful finite-difference or independent reference
check. Changes to state layout also need full/streaming/chunk equivalence tests.
Report the Python/NumPy versions, seed, and command when describing a numerical
issue. Keep timing claims tied to the measured machine and measurement scope.

Do not commit private datasets, model credentials, or large downloaded weights.
Use `runs/` for your own experiments. The small bundled checkpoints are educational
demo models trained from scratch on the included data/procedural task.
