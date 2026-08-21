#!/usr/bin/env bash
set -euo pipefail

export PATH="/root/.local/bin:$PATH"
export UV_NO_MODIFY_PATH=1
export UV_LINK_MODE=copy
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

venv=/tmp/triton37-wheel-smoke
uv venv --system-site-packages --python /usr/bin/python3 "$venv"
uv pip install --python "$venv/bin/python" --no-deps /wheel/triton-*.whl

"$venv/bin/python" /workspace/phase19/tools/gfx906_triton37_smoke.py
