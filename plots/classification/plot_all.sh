#!/usr/bin/env bash
# Regenerate all three noise-robustness figures into ../figures/ from ../data/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

for d in smnist tidigits dvs; do
    echo "=== plotting ${d} ==="
    "$PYTHON" "draw_${d}.py" "$@"
done

echo "All figures written to ../figures/"
