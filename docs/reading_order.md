# Read the code in this order

1. Run `python examples/scan_walkthrough.py`. The scalar example makes each
   write, decay, and read visible as a number.
2. Read `docs/math.md` and `mamba_numpy/scan.py`. Follow the sequential forward
   loop first. Notice that `B_t/C_t` are shared across expanded channels.
3. Read `mamba_numpy/nn.py`. Trace RMSNorm, causal filtering, and stable loss.
4. Read `MambaMixer.forward` in `model.py`. Match every tensor to the diagram.
5. Read `selective_scan_backward`, then `MambaMixer.backward`. Follow how a
   single upstream derivative splits into state, gate, and direct paths.
6. Run `python scripts/gradcheck.py --samples 6`. Open the JSON report to see
   analytic and numerical derivatives rather than just a PASS label.
7. Run `python examples/quickstart.py` and then the text-training command.
   Inspect the loss, optimizer state, tokenizer, and saved configuration.
8. Run `python examples/streaming.py`. Split the sequence at other positions,
   carry the returned state, and compare with full forward.
9. Train selective recall, then sweep the delay and disable state readout.
   Separate in-distribution success from extrapolation behavior.
10. Inspect `affine_prefix`. Work through the composition of three affine
    maps, then compare its work/storage with the simple recurrent loop.

Useful experiments: vary state size, time-step rank, convolution width, depth,
or tied weights one at a time. Keep seeds, total updates, data streams, and
evaluation conditions recorded. If you modify an equation, rerun numerical
checks and streaming equivalence before interpreting task accuracy.
