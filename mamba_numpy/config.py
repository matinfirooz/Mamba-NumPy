"""Configuration for the real-valued Mamba-1 architecture."""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MambaConfig:
    vocab_size: int = 32
    d_model: int = 32
    n_layers: int = 2
    d_state: int = 8
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | None = None
    dt_min: float = 0.001
    dt_max: float = 0.1
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dtype: str = "float32"
    seed: int = 42

    def __post_init__(self):
        for name in ("vocab_size", "d_model", "n_layers", "d_state", "d_conv", "expand"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dt_rank is not None and (
            isinstance(self.dt_rank, bool) or not isinstance(self.dt_rank, int) or self.dt_rank < 1
        ):
            raise ValueError("dt_rank must be None or a positive integer")
        if not (math.isfinite(self.dt_min) and math.isfinite(self.dt_max)
                and 0 < self.dt_min <= self.dt_max):
            raise ValueError("require finite 0 < dt_min <= dt_max")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive and finite")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(self.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be a bool")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def resolved_dt_rank(self) -> int:
        return self.dt_rank or math.ceil(self.d_model / 16)

    @property
    def np_dtype(self):
        return np.dtype(self.dtype)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "MambaConfig":
        return cls(**values)
