"""Dependency-light loader for the trainer's pilot evidence contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[2] / "verl" / "utils" / "reproduction_pilot.py"
_MODULE_NAME = "_rememr1_reproduction_pilot_contract"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - repository corruption.
    raise ImportError(f"cannot load pilot evidence contract: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault(_MODULE_NAME, _MODULE)
_SPEC.loader.exec_module(_MODULE)

for _name in _MODULE.__all__:
    globals()[_name] = getattr(_MODULE, _name)

__all__ = list(_MODULE.__all__)
