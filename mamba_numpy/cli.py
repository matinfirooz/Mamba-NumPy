"""Command-line interface; see python -m mamba_numpy --help."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .checkpoint import load_checkpoint
from .config import MambaConfig
from .sampling import generate
from .training import TaskData, TrainConfig, evaluate, train


def parser():
    root = argparse.ArgumentParser(description="Mamba-1 from scratch with NumPy only")
    sub = root.add_subparsers(dest="command", required=True)
    for command, task in (("train-text", "text"), ("train-recall", "recall")):
        p = sub.add_parser(command, help=f"train a {task} model")
        p.set_defaults(task=task)
        p.add_argument("--out", type=Path, required=True)
        if task == "text":
            p.add_argument("--data", type=Path, required=True)
        p.add_argument("--config", type=Path, help="JSON MambaConfig; vocabulary is inferred from data")
        for flag in ("d-model", "n-layers", "d-state", "d-conv", "expand", "dt-rank"):
            p.add_argument("--" + flag, type=int)
        p.add_argument("--dtype", choices=("float32", "float64"))
        p.add_argument("--untied", action="store_true", default=None, help="use an independent output head")
        for flag in ("steps", "batch-size", "seq-len", "warmup", "log-every", "save-every",
                     "eval-batches", "seed", "n-values", "min-delay", "max-delay"):
            p.add_argument("--" + flag, type=int)
        for flag in ("lr", "weight-decay", "grad-clip", "val-fraction"):
            p.add_argument("--" + flag, type=float)
    p = sub.add_parser("resume", help="continue saved optimizer and RNG state")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True, help="total target updates, not additional updates")
    p.add_argument("--data", type=Path, help="relocated corpus; SHA-256 must match")
    p = sub.add_parser("generate", help="generate text from a character-model checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prompt", default="mamba")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("evaluate", help="evaluate a saved model on fixed held-out batches")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, help="UTF-8 corpus used for a text run")
    p.add_argument("--batches", type=int, default=10)
    p = sub.add_parser("info", help="show architecture, parameter count and recurrent memory")
    p.add_argument("--checkpoint", type=Path, required=True)
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command.startswith("train-"):
            values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
            for key in ("d_model", "n_layers", "d_state", "d_conv", "expand", "dt_rank", "dtype", "seed"):
                value = getattr(args, key)
                if value is not None:
                    values[key] = value
            if args.untied:
                values["tie_embeddings"] = False
            config = MambaConfig.from_dict(values)
            training = asdict(TrainConfig(task=args.task))
            for key in training:
                if hasattr(args, key) and getattr(args, key) is not None:
                    training[key] = getattr(args, key)
            train(config, TrainConfig(**training), args.out, data_path=getattr(args, "data", None))
        elif args.command == "resume":
            ckpt = load_checkpoint(args.checkpoint)
            training = dict(ckpt.metadata["train_config"], steps=args.steps)
            train(ckpt.model.config, TrainConfig(**training), args.out,
                  data_path=args.data, resume_path=args.checkpoint)
        elif args.command == "generate":
            ckpt = load_checkpoint(args.checkpoint)
            if ckpt.tokenizer is None:
                raise ValueError("text generation requires a character-tokenizer checkpoint")
            tokens = generate(ckpt.model, ckpt.tokenizer.encode(args.prompt), args.tokens,
                              temperature=args.temperature, top_k=args.top_k,
                              top_p=args.top_p, seed=args.seed)
            print(ckpt.tokenizer.decode(tokens))
        elif args.command == "evaluate":
            if args.batches < 1:
                raise ValueError("batches must be positive")
            ckpt = load_checkpoint(args.checkpoint)
            training = TrainConfig(**ckpt.metadata["train_config"])
            data = TaskData(training, path=args.data or ckpt.metadata.get("data_path"), tokenizer=ckpt.tokenizer)
            print(json.dumps(evaluate(ckpt.model, data, batches=args.batches), indent=2))
        else:
            ckpt = load_checkpoint(args.checkpoint)
            model = ckpt.model
            print(json.dumps({"config": model.config.to_dict(), "parameters": model.num_parameters,
                              "parameter_bytes": sum(p.data.nbytes for p in model.parameters()),
                              "state_bytes_per_sequence": model.state_bytes(),
                              "step": ckpt.metadata.get("step"),
                              "has_tokenizer": ckpt.tokenizer is not None}, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, FloatingPointError) as error:
        root.error(str(error))


if __name__ == "__main__":
    main()
