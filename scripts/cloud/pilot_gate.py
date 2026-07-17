"""Validate paired three-step pilots and publish a scientific gate decision."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.cloud.pilot_evidence import (  # noqa: E402
    PilotEvidenceError,
    create_gate_report,
    publish_gate_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-evidence", type=Path, required=True)
    parser.add_argument("--c-evidence", type=Path, required=True)
    parser.add_argument("--b-step-zero", type=Path, required=True)
    parser.add_argument("--c-step-zero", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = create_gate_report(
            b_evidence_path=args.b_evidence,
            c_evidence_path=args.c_evidence,
            b_step_zero_path=args.b_step_zero,
            c_step_zero_path=args.c_step_zero,
        )
        publish_gate_report(args.output, report)
    except Exception as exc:
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "status": "blocked"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["outcome"] == "pass" else 42


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PilotEvidenceError", "main"]
