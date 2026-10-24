"""Reproducible text and selective-recall training, evaluation, and resume."""

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from .checkpoint import load_checkpoint, save_checkpoint
from .config import MambaConfig
from .data import text_batch, selective_recall_batch
from .model import MambaLM
from .nn import cross_entropy
from .optim import AdamW, clip_grad_norm
from .tokenizer import CharTokenizer


@dataclass(frozen=True)
class TrainConfig:
    task: str = "text"
    steps: int = 300
    batch_size: int = 8
    seq_len: int = 48
    lr: float = 0.003
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup: int = 10
    log_every: int = 25
    save_every: int = 100
    eval_batches: int = 4
    seed: int = 42
    val_fraction: float = 0.15
    n_values: int = 4
    min_delay: int = 4
    max_delay: int = 9

    def __post_init__(self):
        if self.task not in ("text", "recall"):
            raise ValueError("task must be text or recall")
        for name in ("steps", "batch_size", "seq_len", "log_every", "save_every", "eval_batches", "n_values"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup", "min_delay", "max_delay", "seed"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("lr", "grad_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be in (0,1)")
        if self.max_delay < self.min_delay or (self.task == "recall" and self.seq_len < self.min_delay + 2):
            raise ValueError("invalid recall delays or sequence length")


class TaskData:
    def __init__(self, config, *, path=None, tokenizer=None):
        self.config = config
        self.tokenizer = None
        self.fingerprint = None
        self.path = None
        if config.task == "text":
            if path is None:
                raise ValueError("text training/evaluation needs a UTF-8 data path")
            self.path = str(Path(path))
            raw = Path(path).read_bytes()
            text = raw.decode("utf-8")
            self.fingerprint = hashlib.sha256(raw).hexdigest()
            # Fit on the training portion only. Validation-only symbols map to UNK.
            split = int(len(text) * (1 - config.val_fraction))
            self.tokenizer = tokenizer or CharTokenizer.fit(text[:split])
            self.train_tokens = self.tokenizer.encode(text[:split])
            self.val_tokens = self.tokenizer.encode(text[split:])
            if min(len(self.train_tokens), len(self.val_tokens)) <= config.seq_len:
                raise ValueError("both train/validation splits need more than seq_len characters")
            self.vocab_size = self.tokenizer.vocab_size
        else:
            self.vocab_size = 2 * config.n_values + 2

    def batch(self, rng, *, split="train"):
        c = self.config
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        if c.task == "text":
            tokens = self.train_tokens if split == "train" else self.val_tokens
            return text_batch(tokens, c.batch_size, c.seq_len, rng)
        return selective_recall_batch(c.batch_size, c.seq_len, rng,
                                      n_values=c.n_values, min_delay=c.min_delay, max_delay=c.max_delay)


def evaluate(model, data, *, split="validation", batches=None, seed=None):
    config = data.config
    rng = np.random.default_rng(config.seed + 1000 if seed is None else seed)
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for _ in range(config.eval_batches if batches is None else batches):
        x, y = data.batch(rng, split=split)
        logits = model.forward(x)
        loss, _ = cross_entropy(logits, y)
        mask = y != -100
        n = int(mask.sum())
        total_loss += loss * n
        total_correct += int(np.sum((logits.argmax(axis=-1) == y) & mask))
        total_tokens += n
    if not total_tokens:
        raise ValueError("evaluation requires a positive number of batches")
    loss = total_loss / total_tokens
    result = {"loss": loss, "accuracy": total_correct / total_tokens, "tokens": total_tokens}
    if config.task == "text":
        result["perplexity"] = math.exp(min(loss, 700))
    return result


def train(model_config, train_config, out_dir, *, data_path=None, resume_path=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    if resume_path is None and (metrics_path.exists() or (out_dir / "last.npz").exists()):
        raise FileExistsError("output already contains a training run; choose a new --out or resume")
    rng = np.random.default_rng(train_config.seed)
    step = 0
    previous_seconds = 0.0
    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, load_optimizer=True)
        model, optimizer = ckpt.model, ckpt.optimizer
        saved = ckpt.metadata
        old_config = TrainConfig(**saved["train_config"])
        # Total target steps and logging/save cadence may be changed on resume.
        changeable = {"steps", "log_every", "save_every", "eval_batches"}
        for name, value in asdict(old_config).items():
            if name not in changeable and getattr(train_config, name) != value:
                raise ValueError(f"resume cannot change {name}")
        data = TaskData(train_config, path=data_path or saved.get("data_path"), tokenizer=ckpt.tokenizer)
        if data.fingerprint != saved.get("data_sha256"):
            raise ValueError("resume corpus content differs from checkpoint")
        if ckpt.rng_state is None:
            raise ValueError("checkpoint is missing training RNG state")
        rng.bit_generator.state = ckpt.rng_state
        step = int(saved["step"])
        previous_seconds = float(saved.get("train_seconds", 0))
        if optimizer.t != step:
            raise ValueError("optimizer step differs from training step")
    else:
        data = TaskData(train_config, path=data_path)
        config_values = model_config.to_dict()
        config_values["vocab_size"] = data.vocab_size
        model = MambaLM(MambaConfig.from_dict(config_values))
        optimizer = AdamW(model.named_parameters(), lr=train_config.lr,
                          weight_decay=train_config.weight_decay)
    if train_config.steps <= step:
        raise ValueError(f"target steps must exceed saved step {step}")
    if model.config.vocab_size != data.vocab_size:
        raise ValueError("model vocabulary differs from task vocabulary")
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "platform": platform.platform(), "processor": platform.processor()}
    run = {"model_config": model.config.to_dict(), "train_config": asdict(train_config),
           "data_path": data.path, "data_sha256": data.fingerprint, "environment": environment,
           "parameters": model.num_parameters, "resume_from": str(resume_path) if resume_path else None}
    # A fresh output directory can receive resumed runs; existing history must match.
    append = resume_path is not None and metrics_path.exists()
    if append:
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            prior_rows = list(csv.DictReader(stream))
        if prior_rows and int(prior_rows[-1]["step"]) > step:
            raise ValueError("metrics contain steps newer than the resume checkpoint; choose another out directory")
    (out_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    fields = ["step", "train_loss", "val_loss", "val_accuracy", "train_seconds", "grad_norm"]
    start_time = time.perf_counter()
    last_norm = 0.0

    def seconds():
        return previous_seconds + time.perf_counter() - start_time

    def save(name):
        save_checkpoint(out_dir / name, model, optimizer=optimizer, tokenizer=data.tokenizer,
                        metadata={**run, "step": step, "train_seconds": seconds()},
                        rng_state=rng.bit_generator.state)

    def record(writer, stream):
        train_eval = evaluate(model, data, split="train", seed=train_config.seed + 2000)
        val_eval = evaluate(model, data)
        row = {"step": step, "train_loss": train_eval["loss"], "val_loss": val_eval["loss"],
               "val_accuracy": val_eval["accuracy"], "train_seconds": seconds(), "grad_norm": last_norm}
        writer.writerow(row)
        stream.flush()
        print(f"step {step:5d} | train {row['train_loss']:.4f} | val {row['val_loss']:.4f} | "
              f"val acc {row['val_accuracy']:.1%} | {row['train_seconds']:.1f}s", flush=True)

    print(f"{train_config.task}: {model.num_parameters:,} parameters | {model.config.dtype} | "
          f"state {model.state_bytes():,} bytes / sequence", flush=True)
    with metrics_path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not append:
            writer.writeheader()
            record(writer, stream)
        while step < train_config.steps:
            x, y = data.batch(rng)
            loss = model.loss_and_grad(x, y)
            if not math.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            last_norm = clip_grad_norm(model.parameters(), train_config.grad_clip)
            scale = min(1.0, (step + 1) / max(1, train_config.warmup))
            optimizer.step(lr=train_config.lr * scale)
            step += 1
            if step % train_config.log_every == 0 or step == train_config.steps:
                record(writer, stream)
            if step % train_config.save_every == 0 or step == train_config.steps:
                save("last.npz")
    save("last.npz")
    final = evaluate(model, data)
    summary = {"task": train_config.task, "step": step, "parameters": model.num_parameters,
               "validation": final, "train_seconds": seconds(), "state_bytes_per_sequence": model.state_bytes()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return model, summary
