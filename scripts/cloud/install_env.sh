#!/usr/bin/env bash
# Install and prove the persistent Python 3.12.2 reproduction environment.
set -euo pipefail

CLOUD_ENV="${REMEMR1_CLOUD_ENV:-/root/autodl-tmp/rememr1-cloud.env}"
[[ -f "${CLOUD_ENV}" ]] || {
    echo "cloud environment is missing" >&2
    exit 1
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/runtime.sh"
rememr1_load_cloud_env "${CLOUD_ENV}"
rememr1_require_cloud_env
cd "${REMEMR1_PROJECT_DIR}"

ENV_PREFIX="${REMEMR1_ENV_PREFIX}"
PYTHON_VERSION="3.12.2"
CONDA_BIN="${REMEMR1_CONDA_BIN:-}"
if [[ -z "${CONDA_BIN}" ]]; then
    for candidate in \
        "$(command -v conda 2>/dev/null || true)" \
        /root/miniconda3/bin/conda \
        /opt/conda/bin/conda; do
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            CONDA_BIN="${candidate}"
            break
        fi
    done
fi
[[ -n "${CONDA_BIN}" && -x "${CONDA_BIN}" ]] || {
    echo "conda was not found; use an AutoDL image with Miniconda or set REMEMR1_CONDA_BIN" >&2
    exit 1
}

mkdir -p "$(dirname "${ENV_PREFIX}")" "${PIP_CACHE_DIR}"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    "${CONDA_BIN}" create --yes --prefix "${ENV_PREFIX}" \
        "python=${PYTHON_VERSION}" pip
fi
PYTHON="${ENV_PREFIX}/bin/python"
"${PYTHON}" -c \
    'import sys; actual = sys.version_info[:3]; expected = (3, 12, 2); sys.exit(f"expected Python {expected}, got {actual}") if actual != expected else None'

"${PYTHON}" -m pip install --upgrade "pip==26.1.2"
"${PYTHON}" -m pip install \
    -r environment/reproduction-cu130.requirements.txt \
    -r environment/reproduction-cloud.requirements.txt
"${PYTHON}" -m pip check
"${PYTHON}" scripts/reproduction/verify_environment.py --lock-only

# These imports cover the CPU builder and the unconditional trainer import
# chain. GPU kernels are deliberately installed and evidenced on the GPU host.
"${PYTHON}" - <<'PY'
import os

import datasets
import hydra
import huggingface_hub
import peft
import pyarrow
import ray
import tensordict
import torch
import transformers
import uvloop

from taskutils.data_synthesis import reproduction_builder

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CPU preparation requires CUDA_VISIBLE_DEVICES to be empty")
if os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void":
    raise RuntimeError("CPU preparation requires NVIDIA_VISIBLE_DEVICES=void")
if torch.cuda.is_available() or torch.cuda.device_count() != 0:
    raise RuntimeError("CPU preparation unexpectedly exposed a CUDA device")
if not reproduction_builder.BUNDLE_KIND:
    raise RuntimeError("reproduction bundle contract is unavailable")
print("CPU import preflight passed with CUDA unavailable")
PY

freeze="${REMEMR1_PERSIST_ROOT}/evidence/pip-freeze.cpu.txt"
mkdir -p "$(dirname "${freeze}")"
"${PYTHON}" -m pip freeze --all | LC_ALL=C sort > "${freeze}.tmp.$$"
mv "${freeze}.tmp.$$" "${freeze}"
sha256sum "${freeze}" > "${freeze}.sha256.tmp.$$"
mv "${freeze}.sha256.tmp.$$" "${freeze}.sha256"
rememr1_sync_file "${freeze}"

echo "Environment ready: ${ENV_PREFIX}"
echo "Dependency evidence: ${freeze}"
