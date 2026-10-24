# Demo checkpoints

`text_demo.npz` and `recall_demo.npz` are tiny Mamba-1 models trained from
scratch with this repository. Each contains model weights, architecture,
AdamW moments/update count, metadata, and the training generator's state.
The text checkpoint also contains the fitted character tokenizer.

They are included so generation and evaluation work immediately. They are
not official pretrained Mamba weights. The teaching corpus is original to
this repository; recall data is generated procedurally. See the main README
and `docs/experiments.md` for commands and limitations.

```bash
python -m mamba_numpy info --checkpoint checkpoints/text_demo.npz
python -m mamba_numpy generate --checkpoint checkpoints/text_demo.npz --prompt "mamba"
python -m mamba_numpy evaluate --checkpoint checkpoints/text_demo.npz --data data/tiny_corpus.txt
```

To continue the bundled model, choose your own output directory:

```bash
python -m mamba_numpy resume --checkpoint checkpoints/text_demo.npz --out runs/continued_text --steps 900 --data data/tiny_corpus.txt
```
