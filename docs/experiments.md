# Experiments and evidence

All reported numbers in the README come from files included in this package.
They describe small educational experiments, not a reproduction of the
Mamba paper's scale, pretrained quality, or kernel performance.

## Character language modeling

Command for the final configuration:

```bash
python -m mamba_numpy train-text --data data/tiny_corpus.txt --out runs/text --config configs/tiny.json --steps 600 --log-every 50
```

The bundled run initially trained for 300 updates, then resumed from its
optimizer/RNG checkpoint to 600. The learning rate is constant after 10 warmup
updates, so extending the total update target does not change earlier learning
rates. The exact-resume test verifies split and uninterrupted training produce
the same weights in one numerical environment.

The original, included corpus has intentionally repeated passages. Split by
character position: first 85% training, last 15% validation. Sample shifted
windows of 48 characters, batch size 8. Fit the character vocabulary on training
text only. Evaluation uses four fixed batches, yielding 1,536 targets; it is
sampled evaluation, not exhaustive evaluation over the full validation split.
The train probe uses a different fixed seed from the validation probe.

Configuration: width 32, two layers, state size 8, expansion 2, convolution 4,
time-step rank 2, tied embeddings, float32, seed 42, AdamW rate 0.003, decay
0.01, clipping 1.0. There is no dropout. Raw configs/logs are in
`results/text_demo/`; the final full training state is
`checkpoints/text_demo.npz`.

Final fixed-probe loss: 0.455922606; character perplexity: 1.57762824;
token accuracy: 87.955729%. These numbers show that the implementation learns
the supplied distribution. Corpus repetition limits any generalization claim.

## Selective recall

```bash
python -m mamba_numpy train-recall --out runs/recall --config configs/tiny.json --steps 600 --seq-len 32 --batch-size 16 --log-every 100 --save-every 200
```

Ten-token vocabulary: zero reserved, values 1..4, distractors 5..8, and query 9.
A segment contains a uniformly sampled selected value, 4..9 distractor tokens,
and a query. Several segments are packed into each sequence. The query target
is the selected value; all other targets are `-100` and contribute no loss.
At least one complete segment exists per sequence. Values are never used as
distractors, so writes are unambiguous. Model/training settings match the text
run except batch size 16, sequence length 32, vocabulary size, and logging.

Train and validation examples are generated from independent deterministic
random streams. The fixed validation probe has 231 query targets in four
batches. The trained checkpoint reaches 100% accuracy on that probe at
update 100 and retains it at 600; final loss is 0.000395560.

This is not the paper's selective-copy benchmark: it requests the latest
selected item at each query rather than recalling a selected subsequence.
There is one training seed, and no confidence intervals or multi-seed claims.

## Delay sweep and state ablation

```bash
python scripts/evaluate_recall.py --checkpoint checkpoints/recall_demo.npz --out runs/recall_evaluation --batches 16
```

The evaluation seed is 20261023, distinct from both training and validation
probe seeds. Each setting uses 16 batches at batch size 16; segment packing
causes query counts to vary. Sequence lengths/delays are 32/(4..9),
64/(12..20), 128/(32..48), and 256/(64..96). Inputs and targets are identical
between the trained and ablated models at each setting.

The inference ablation zeros the `C_t` projection columns in **every** layer.
The state may still be updated but cannot affect the output. All other learned
weights are retained; the ablated network is not retrained. Treat the result as
a dependence diagnostic, not a competitive local-only baseline.

Two four-tap local convolutions can carry information across at most six
intervening token steps beyond the current input, for a receptive field of
seven tokens. Thus the 12..20 delay condition exceeds the local-only path's
receptive field. The trained model gets 821/824 queries correct there; the
ablation gets 204/824. At delays 32..48 and 64..96 the trained checkpoint
falls near 25% chance. This is a measured extrapolation limit; it does not
contradict the fixed-size recurrent state or establish a limit of Mamba in
general.

Raw outcomes and sample counts are in `results/recall_evaluation/`.

## CPU benchmark

```bash
# Linux/macOS: set these before Python starts.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python scripts/benchmark.py --out runs/benchmark --repeats 5
```

PowerShell equivalents:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python scripts/benchmark.py --out runs/benchmark --repeats 5
```

The default model is randomly initialized, with vocabulary 32, width 32, two
layers, state size 8, expansion 2, convolution 4, and float32. Batch size is
two. For each sequence length 16..512, each method is warmed up once, checked
against sequential logits, and timed over repeated complete chunk calls.
The median is saved along with tokens/second and maximum absolute logit error.

Streaming timing includes creation of the zero initial state and return of
all logits. Sequential timing uses batched projections and a Python scan.
The prefix demo uses Hillis–Steele composition with O(L log L) work; it is not
the official fused GPU scan and need not outperform sequential NumPy.
Hardware, versions, repetition count, and thread variables are recorded in
`results/benchmark/environment.json`. CPU scheduling and virtualization can
affect these small timings. No comparison to Transformers or GPU frameworks
is claimed.

The memory fields count only explicit recurrent conv/SSM arrays and the full
SSM state history used by manual training. They exclude weights, output logits,
other caches, gradients, moments, and temporary arrays; they are not peak RSS.

## Numerical verification

`results/gradcheck.json` contains per-coordinate finite differences in float64.
The 165 checks sample six coordinates per tensor where available, covering
every model tensor and all scan inputs. The scan check includes nonzero
initial state and a final-state loss contribution. The maximum absolute
error is approximately 4.40e-10. It is sampled checking, not an exhaustive
proof over every coefficient or a comparison to official PyTorch checkpoints.

The unit suite tests independent properties including forward/stream/chunk
agreement for several convolution widths and both tied/untied heads, causality,
loss masking, convolution tap orientation, finite-loss training, checkpoint
validation, and exact optimizer/RNG continuation. The CI matrix is supplied;
it has not been run on GitHub by this ZIP's authoring environment.

## Regenerate charts and inspect traces

```bash
python scripts/render_assets.py
python scripts/export_trace.py --checkpoint checkpoints/text_demo.npz --text "mamba reads a stream" --out runs/trace
```

SVG rendering uses only the standard library and reads the checked-in CSVs.
The exported trace is a floating-point software reference, with zero state
at index zero and each post-update state at index `t+1`. Loading uses
`np.load(path, allow_pickle=False)`.
