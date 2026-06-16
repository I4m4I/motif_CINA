#!/usr/bin/env bash
# Regenerate all four reward-curve figures into ../figures/ from ../data/.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"

for env in walker ant ip idp; do
    echo "=== plotting ${env} ==="
    "$PYTHON" "draw_${env}.py" "$@"
done

echo "All figures written to ../figures/"
