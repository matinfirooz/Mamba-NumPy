"""Portable model/optimizer checkpoints: numeric NPZ arrays + JSON, no pickle."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .config import MambaConfig
from .model import MambaLM
from .optim import AdamW
from .tokenizer import CharTokenizer


@dataclass
class Checkpoint:
    model: MambaLM
    optimizer: AdamW | None
    tokenizer: CharTokenizer | None
    metadata: dict
    rng_state: dict | None


def save_checkpoint(path, model, *, optimizer=None, tokenizer=None, metadata=None, rng_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer is not None and tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("tokenizer and model vocabulary sizes differ")
    if optimizer is not None and (
        set(optimizer.params) != set(dict(model.named_parameters()))
        or any(optimizer.params[n] is not p for n, p in model.named_parameters())
    ):
        raise ValueError("optimizer belongs to a different model")
    header = {"format_version": 1, "config": model.config.to_dict(),
              "tokenizer": tokenizer.to_dict() if tokenizer else None,
              "metadata": metadata or {}, "rng_state": rng_state,
              "optimizer": None if optimizer is None else {
                  **optimizer.hyperparameters(), "t": optimizer.t}}
    arrays = {"header_json": np.array(json.dumps(header, ensure_ascii=False, allow_nan=False))}
    for name, p in model.named_parameters():
        arrays["parameter/" + name] = p.data
        if optimizer is not None:
            arrays["adam_m/" + name] = optimizer.m[name]
            arrays["adam_v/" + name] = optimizer.v[name]
    # Same-directory temporary + atomic replace prevents partially written saves.
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, *, load_optimizer=False):
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header_json"].item()))
        if header.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        model = MambaLM(MambaConfig.from_dict(header["config"]))
        expected = {"header_json"}
        opt_data = header.get("optimizer")
        for name, p in model.named_parameters():
            key = "parameter/" + name
            expected.add(key)
            if opt_data is not None:
                expected.update(("adam_m/" + name, "adam_v/" + name))
            if key not in archive:
                raise ValueError(f"missing checkpoint parameter: {name}")
            value = archive[key]
            if value.shape != p.data.shape or value.dtype != p.data.dtype or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid checkpoint parameter: {name}")
            p.data[...] = value
        if set(archive.files) != expected:
            raise ValueError("checkpoint array keys do not match the model and optimizer")
        tokenizer = None if header["tokenizer"] is None else CharTokenizer.from_dict(header["tokenizer"])
        if tokenizer is not None and tokenizer.vocab_size != model.config.vocab_size:
            raise ValueError("checkpoint tokenizer does not match the model")
        optimizer = None
        if load_optimizer:
            if opt_data is None:
                raise ValueError("checkpoint contains no optimizer state")
            optimizer = AdamW(model.named_parameters(), **{k: v for k, v in opt_data.items() if k != "t"})
            if not isinstance(opt_data["t"], int) or opt_data["t"] < 0:
                raise ValueError("invalid optimizer step")
            optimizer.t = opt_data["t"]
            for name, p in model.named_parameters():
                for prefix, target in (("adam_m/", optimizer.m), ("adam_v/", optimizer.v)):
                    value = archive[prefix + name]
                    if (value.shape != p.data.shape or value.dtype != p.data.dtype
                            or not np.all(np.isfinite(value))
                            or (prefix == "adam_v/" and np.any(value < 0))):
                        raise ValueError(f"invalid optimizer state: {prefix}{name}")
                    target[name][...] = value
    return Checkpoint(model, optimizer, tokenizer, header["metadata"], header["rng_state"])
