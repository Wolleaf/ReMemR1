"""Deterministic seed derivation for reproduction runs and worker processes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import random
from typing import Any


MAX_SEED = 2**32 - 1

ROLLOUT_GLOBAL_STEP_KEY = "rollout_global_step"
ROLLOUT_SAMPLE_INDEX_KEY = "rollout_sample_index"
ROLLOUT_TRAJECTORY_INDEX_KEY = "rollout_trajectory_index"
ROLLOUT_TURN_INDEX_KEY = "rollout_turn_index"


def _validate_seed(seed: int, *, name: str = "seed") -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"{name} must be an int")
    if seed < 0 or seed > MAX_SEED:
        raise ValueError(f"{name} must be in [0, {MAX_SEED}]")
    return seed


def validate_seed(seed: int, *, name: str = "seed") -> int:
    """Validate a configured seed without accepting bools, floats, or strings."""

    return _validate_seed(seed, name=name)


def _validate_nonnegative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def derive_seed(run_seed: int, namespace: str, *coordinates: Any) -> int:
    """Derive a stable 32-bit seed without depending on Python's hash salt."""

    _validate_seed(run_seed, name="run_seed")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")
    payload = json.dumps(
        {
            "schema_version": 1,
            "run_seed": run_seed,
            "namespace": namespace,
            "coordinates": coordinates,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def derive_rollout_generation_seed(
    rollout_seed: int,
    global_step: int,
    sample_index: int,
    trajectory_index: int,
    turn_index: int,
) -> int:
    """Derive an isolated RNG stream for one recurrent generation call."""

    rollout_seed = _validate_seed(rollout_seed, name="rollout_seed")
    for name, value in (
        ("global_step", global_step),
        ("sample_index", sample_index),
        ("trajectory_index", trajectory_index),
        ("turn_index", turn_index),
    ):
        _validate_nonnegative_int(value, name=name)

    trajectory_seed = derive_seed(
        rollout_seed,
        "trajectory",
        global_step,
        sample_index,
        trajectory_index,
    )
    return derive_seed(trajectory_seed, "turn", turn_index)


def make_rollout_coordinate_tensors(
    expanded_batch_size: int,
    trajectories_per_sample: int,
    *,
    sample_offset: int = 0,
    device=None,
):
    """Build coordinates for an interleaved ``DataProto.repeat`` expansion."""

    expanded_batch_size = _validate_nonnegative_int(
        expanded_batch_size,
        name="expanded_batch_size",
    )
    trajectories_per_sample = _validate_nonnegative_int(
        trajectories_per_sample,
        name="trajectories_per_sample",
    )
    sample_offset = _validate_nonnegative_int(sample_offset, name="sample_offset")
    if trajectories_per_sample == 0:
        raise ValueError("trajectories_per_sample must be positive")
    if expanded_batch_size % trajectories_per_sample:
        raise ValueError(
            "expanded_batch_size must be divisible by trajectories_per_sample"
        )

    import torch

    expanded_index = torch.arange(expanded_batch_size, dtype=torch.long, device=device)
    return {
        ROLLOUT_SAMPLE_INDEX_KEY: expanded_index // trajectories_per_sample + sample_offset,
        ROLLOUT_TRAJECTORY_INDEX_KEY: expanded_index % trajectories_per_sample,
    }


@dataclass(frozen=True, slots=True)
class ReproductionSeeds:
    """Explicit seed namespaces shared by B/C and checkpoint metadata."""

    run: int
    data: int
    model_init: int
    rollout: int

    @classmethod
    def from_run_seed(cls, run_seed: int) -> "ReproductionSeeds":
        run_seed = _validate_seed(run_seed, name="run_seed")
        return cls(
            run=run_seed,
            data=derive_seed(run_seed, "data"),
            model_init=derive_seed(run_seed, "model_init"),
            rollout=derive_seed(run_seed, "rollout"),
        )

    def rollout_for(self, global_step: int, sample_index: int, trajectory_index: int) -> int:
        for name, value in (
            ("global_step", global_step),
            ("sample_index", sample_index),
            ("trajectory_index", trajectory_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        return derive_seed(
            self.rollout,
            "trajectory",
            global_step,
            sample_index,
            trajectory_index,
        )

    def rollout_generation_for(
        self,
        global_step: int,
        sample_index: int,
        trajectory_index: int,
        turn_index: int,
    ) -> int:
        return derive_rollout_generation_seed(
            self.rollout,
            global_step,
            sample_index,
            trajectory_index,
            turn_index,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def seed_process(seed: int, *, include_cuda: bool = True) -> None:
    """Seed Python, NumPy, Torch CPU, and available CUDA generators."""

    seed = _validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if include_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


__all__ = [
    "MAX_SEED",
    "ROLLOUT_GLOBAL_STEP_KEY",
    "ROLLOUT_SAMPLE_INDEX_KEY",
    "ROLLOUT_TRAJECTORY_INDEX_KEY",
    "ROLLOUT_TURN_INDEX_KEY",
    "ReproductionSeeds",
    "derive_seed",
    "derive_rollout_generation_seed",
    "make_rollout_coordinate_tensors",
    "seed_process",
    "validate_seed",
]
