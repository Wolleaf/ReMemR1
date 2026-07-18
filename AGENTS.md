# ReMemR1 Repository Guide

This file applies to the entire repository. It is an agent-facing guide, not a
replacement for the active experiment plan, resolved configs, or tests.

## Project Mission

This repository started from the official implementation of the ICLR 2026
paper *Look Back to Reason Forward: Revisitable Memory for Long-Context LLM
Agents*. The active branch adapts that work into a resource-constrained,
portfolio-oriented mechanism reproduction:

> ReMemR1 on one RTX 5090 32 GB with fixed-revision Qwen3.5-2B and LoRA-GRPO.

Treat this as a reproduction and engineering adaptation of upstream work, not
as an original algorithm implementation. Preserve the upstream license and
paper attribution. The repository is MIT-licensed and includes MemAgent-derived
Apache-2.0 components; consult `LICENSE`, `THIRD_PARTY/`, and the root README
before redistributing code. The goal is a small but auditable project with
reproducible training, evaluation, resource accounting, and honest resume
claims; it is not to match the paper's full-scale numbers.

Never turn a plan, CPU fixture, synthetic sample, expected capacity, or paper
number into a claimed local result. Negative or inconclusive results remain
valid results and must not be hidden by selecting favorable seeds, checkpoints,
or evaluation cells.

## Source Of Truth

Read these before making a broad change:

1. `docs/rtx5090_2b_reproduction_plan_zh.md` is the sole active implementation
   contract for scientific choices, hardware, stages, budgets, and claim scope.
2. `docs/reproduction_implementation_handoff_zh.md` is the short session entry
   point, and `scripts/cloud/README.md` is the operational cloud runbook.
3. `2509.23040v5.pdf` defines the paper method. Current code and tests show what
   is actually implemented; resolve disagreements with evidence and update the
   plan, config, and tests together.
4. `.agents/CLAUDE.md` and `.agents/skills/` are useful upstream architecture
   references, but some defaults predate the active reproduction profile.

The root `README.md`, `notes.md`, `docs/final_reproduction_plan_zh.md`, and the
3B/7B launch scripts describe upstream or historical workflows. In particular,
the old Qwen3.5-4B / RTX PRO 6000 plan is not the active profile. Do not copy its
model size, hardware, batch sizes, paths, claims, or install commands into the
2B/5090 workflow.

## Fixed Active Contract

- Branch/profile: `reproduction/rtx5090-2b` and
  `rtx5090-32g-qwen35-2b-v1`.
- Model: `Qwen/Qwen3.5-2B`, revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`, text-only.
- Hardware: exactly one GeForce RTX 5090 with 32 GB VRAM for GPU evidence.
- Training: LoRA-GRPO, LoRA rank 32 / alpha 64 / dropout 0, no QLoRA. The
  active configs retain an FP32 actor master with BF16 compute/reference and
  use the synchronous Hugging Face rollout path, not the upstream SGLang path.
- Scale: train/mini batch 2/2, GRPO group 4, eight trajectories per optimizer
  step, `5000 x 6` recurrent chunks, and memory/final generation limits
  768/512. Verify exact values in active configs rather than duplicating them.
- Paired comparison: B is outcome-only (`algorithm.alpha=1.0`) and C is
  outcome plus state reward (`algorithm.alpha=0.8`). B and C must otherwise
  keep model revision, data, seeds, schedule, decode, and runtime contract
  paired.
- Evaluation: HotpotQA and 2WikiMultiHopQA at 200/800-document settings, with
  `learned`, `none`, and `fixed_question` callback modes.
- Protocol: every intermediate response has exactly one non-empty `<update>`
  and at most one non-empty `<recall>`; the final response has a boxed answer.
  Do not add `compress_context`, switch actions to JSON, or change action
  semantics inside this profile.
- Delivery: B/C40 is the minimum formal L1 experiment. B/C80 is an optional,
  synchronized continuation after capacity, scientific, and budget gates.

Changing one of these is a project decision, not a routine refactor. Make the
new profile or plan explicit instead of silently weakening the active contract.

## Repository Map

- `recurrent/protocol.py`: canonical action parsing, `MemoryRecord`,
  paper-aligned word recall, and deterministic retrieval.
- `recurrent/rewards.py`: CPU-testable paper-aligned state reward primitives.
- `recurrent/impls/memory_revisit.py`: recurrent memory agent, callback modes,
  ordered history, and provenance wiring.
- `verl/`: the modified verl training stack. The main entry is
  `verl/trainer/main_ppo.py`; recurrent reward and GRPO wiring live under
  `verl/trainer/ppo/`.
- `verl/trainer/config/reproduction/`: Hydra configs for G0/G1/G2, B/C stages,
  evaluation, and offload profiles. Derived configs inherit from shorter-stage
  configs, so inspect the full defaults chain before editing.
- `taskutils/data_synthesis/`: deterministic reproduction bundle and manifest
  construction.
- `taskutils/memory_eval/reproduction_runner.py` and
  `reproduction_metrics.py`: active Transformers evaluation and metrics path.
  Older files under `taskutils/memory_eval/utils/` may be upstream compatibility
  paths; do not assume they define the reproduction contract.
- `scripts/reproduction/`: environment checks, asset prefetch, and adapter
  export helpers.
- `scripts/cloud/`: AutoDL preparation, immutable stage state, gates, recovery,
  telemetry, evidence verification, evaluation, export, and safe shutdown.
- `tests/reproduction/` and `tests/cloud/`: the primary CPU-verifiable contract
  suites. `recurrent/test/` contains older focused tests and notebooks.
- `environment/`: pinned direct dependency sets for the active CUDA 13.0 and
  `sm_120` environment. In particular, keep
  `reproduction-cu130.lock.json`, `reproduction-assets.json`, and the matching
  requirements files synchronized through their existing generators/checks.

## Development Workflow

不要过度设计。

Do not overdesign. Optimize for the shortest total wall-clock time to a
trustworthy experiment result; once infrastructure is adequate, prioritize
training and add machinery only for a demonstrated blocker or required contract.

Start by reading the active plan section relevant to the task and running
`git status --short`. The worktree may contain concurrent user or agent work:
preserve it, inspect overlapping edits carefully, and never reset, discard, or
rewrite unrelated changes. Do not commit, push, pull, switch branches, or amend
history unless the user explicitly asks.

The current workspace may have a Python 3.12.2 `.venv` on Windows; it is local
and ignored, not part of the repository. Prefer that interpreter when it
exists, and run commands from the repository root:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/reproduction tests/cloud
```

The local environment may intentionally lack Torch, NumPy, or other cloud
dependencies. Report that limitation instead of installing an ad hoc mixture
or treating collection failures as test passes. The active cloud environment
is separately created and frozen by `scripts/cloud/install_env.sh` with Python
3.12.2 and the `environment/reproduction-*` locks; the root README's Python
3.11/CUDA 12.6 setup belongs to the upstream workflow.

Run the smallest relevant test file first while iterating, then the full two
suites for cross-cutting protocol, config, checkpoint, evaluation, or cloud
changes. The CPU gate also compiles the active Python trees:

```powershell
.venv\Scripts\python.exe -m compileall -q scripts taskutils recurrent verl
```

Do not use a bare repository-wide `pytest`: legacy files such as data/eval
scripts and a reward-manager concurrency test can download a tokenizer,
initialize Ray, or behave as CLIs during collection. Many cloud tests also skip
outside their target platform; a skip is not GPU or AutoDL evidence. There is
no repository-wide formatter/linter configuration, so do not invent a
mass-formatting pass.

For reproduction config changes, run the focused config-matrix tests and the
existing `scripts/cloud/resolve_configs.py` composition in its intended
CPU-finalize context. A YAML parse alone does not validate the defaults chain,
exact key set, manifest binding, or complete resolved-config inventory.

When editing active cloud shell scripts, validate their syntax from Bash/WSL
and preserve tracked modes:

```bash
git ls-files -z 'scripts/cloud/*.sh' 'scripts/cloud/lib/*.sh' | xargs -0 -n1 bash -n
git diff --check
```

Cloud scripts target Bash on Ubuntu/AutoDL, not local PowerShell. The normal
production surface is deliberately only:

```bash
bash /root/autodl-tmp/ReMemR1/scripts/cloud/prepare_cpu.sh
bash /root/autodl-tmp/ReMemR1/scripts/cloud/run_gpu.sh
```

Do not launch either command merely to validate a code change. They download
assets, operate on a persistent volume, may consume paid GPU time, and may
request shutdown. Use unit tests and test-mode fixtures for routine validation.
Use a documented dry run only on the intended host/path after the user asks to
operate the cloud workflow: even `--dry-run` creates persistent state and is not
a generic read-only local check. Production requires a clean checkout at one
complete 40-character commit; after CPU handoff starts, do not pull, switch, or
edit it until the GPU workflow ends. The user handles Git checkout on AutoDL;
launchers must not clone, fetch, pull, or checkout.

The public GPU entry is fixed to R0 and stops after the L0 G0/G1/G2 capacity,
resume, and artifact gates. It does not automatically authorize R1, B/C40,
B/C80, formal evaluation, or result export; those are separately gated paid
research stages.

## Implementation Conventions

- Follow the existing Python style: type hints, dataclasses for structured
  config/state, small pure helpers for scientific logic, and explicit errors.
- Keep protocol parsing centralized in `recurrent/protocol.py`; training,
  memory state, logging, and evaluation must not grow divergent regex parsers.
- Preserve ordered memory records, duplicates, step IDs, source provenance,
  and deterministic earliest-step tie breaking. Do not change history back to
  a `set` or make retrieval dependent on hash iteration order.
- Preserve the paper's `word_recall` argument direction and max-over-valid-gold
  behavior. Equation-level tests must distinguish reversed arguments, max vs.
  average, multiline tags, empty/duplicate actions, and ambiguous final answers.
- Preserve compatibility spellings that are part of an interface, including
  `get_bactch_keys`, unless all callers and tests are migrated together.
- Preserve action-type compatibility: 0 is final, 1 is recall, and 2 is memory
  update, even when one active path does not emit every legacy type.
- Keep active paths fail-closed: invalid manifests, hashes, revisions, resume
  endpoints, capacity evidence, and config bindings must stop rather than fall
  back to placeholders or an old profile.
- Keep execution deterministic where promised. Seeds, manifest identity,
  config identity, callback mode, and checkpoint lineage are experiment data,
  not incidental metadata.
- Use Python logging rather than ad hoc prints in library code. Never log API
  keys, tokens, machine identifiers, or other secrets.
- Avoid broad changes inside vendored/modified `verl` when a narrow adapter is
  sufficient, but test the actual trainer wiring when behavior crosses that
  boundary.
- Preserve neighboring line endings and file modes. In particular, active
  `scripts/cloud/*.sh` files use LF and executable bits; do not normalize them
  while making an unrelated edit.

## Config, Evidence, And Cloud Safety

Hydra source configs may contain placeholder manifest hashes that are replaced
and exact-key checked during the sealed CPU handoff. Do not hand-edit resolved
configs or weaken hash checks to make a run start. Keep B/C stage directories,
checkpoints, attempts, logs, and evidence isolated.

Formal cloud training binds and verifies sealed configs through
`scripts/cloud/run_resolved_training.py`; do not bypass that contract by
launching a source YAML directly. Re-read the tracked environment and asset
locks at execution time. Any `UNVERIFIED` or `BLOCKED` asset/build field means
the corresponding GPU or formal-data claim is still unavailable.

Never manually edit generated handoffs, resolved config indexes, immutable
attempts, stage pointers, terminal records, capacity approvals, cost
projections, or verified result packages. These artifacts use canonical JSON,
self-hashes, atomic publication, and create-if-absent semantics. Fix the source
or generator and create a new valid artifact instead.

The root `.gitignore` has non-obvious behavior: new `*.json`/`*.jsonl` files are
globally ignored, while the leading `./data`, `./models`, and `./results`
patterns do not reliably protect every root artifact. Before adding a new JSON
contract, use `git check-ignore -v --no-index <path>` and make its tracking
intent explicit. Independently inspect status for datasets, weights,
checkpoints, results, logs, caches, and large binaries; never rely on ignore
rules as the safety boundary.

Only a captured PyTorch CUDA OOM can support the documented capacity fallback.
Host OOM, SIGKILL, disconnect, or disappearance is not equivalent evidence.
Scientific stop (42) and capacity stop (43) are non-retryable outcomes; do not
loop retries until a gate appears to pass. R1 requires its documented host RAM,
fresh launcher, projection, explicit one-time approval, and nonce.

Guest `shutdown` does not prove AutoDL billing stopped. Any real run handoff
must remind the operator to verify instance state, balance, and billing in the
provider control plane. Never publish local host inventory, provider IDs, GPU
UUIDs, environment files, credentials, raw datasets, checkpoints, or logs.

## Validation And Claim Levels

For a code or config change, report exactly which tests/checks ran and which
could not run. A passing process exit alone is not enough when the stage also
requires finite loss/gradients, artifact hashes, resume lineage, or scientific
thresholds.

- Code/CPU only: after the required CPU checks pass, may claim an implemented
  and CPU-validated reproduction framework. May not claim the RTX 5090 path ran
  or cite measured GPU numbers.
- L0: requires verified real G0/G1/G2 attempts and supports only the capacity,
  recovery, and artifact-closure claims documented in the active plan.
- L1: requires paired pilots, B/C40, the complete 32-QA-per-cell matrix, and a
  verified package before reporting the formal paired experiment.
- L2: requires synchronized B/C80 continuation, 64 QA per cell, and the final
  verified package.

Resume text and project reports must say what was inherited from ReMemR1, what
was adapted here, the achieved evidence level, the actual measured metrics and
costs, and the limitations of a single-seed 2B LoRA reproduction. Never fill a
resume template's bracketed values until a verified package supplies them.
