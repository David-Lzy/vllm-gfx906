#!/usr/bin/env bash
set -euo pipefail

export PATH="/root/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --no-modify-path
fi

venv=/artifacts/venv
if [[ ! -x "$venv/bin/python" ]]; then
  uv venv --system-site-packages --python /usr/bin/python3 "$venv"
fi

uv pip install --python "$venv/bin/python" -r python/requirements.txt
uv pip install --python "$venv/bin/python" --no-build-isolation --editable .
"$venv/bin/python" /workspace/phase19/tools/gfx906_triton37_smoke.py
