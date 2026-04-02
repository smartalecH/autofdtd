#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_ITERATIONS="${1:-100}"
python3 "$SCRIPT_DIR/state/runtime.py" --bundle-dir "$SCRIPT_DIR" --max-iterations "$MAX_ITERATIONS"
