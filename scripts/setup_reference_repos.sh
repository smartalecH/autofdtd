#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_DIR="$(cd "$REPO_ROOT/.." && pwd)"

UPDATE_MODE="fetch"
if [[ "${1:-}" == "--clone-only" ]]; then
  UPDATE_MODE="skip"
elif [[ "${1:-}" == "--pull" ]]; then
  UPDATE_MODE="pull"
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--clone-only|--pull]" >&2
  exit 2
fi

REPOS=(
  "meep|https://github.com/NanoComp/meep.git"
  "tidy3d|https://github.com/flexcompute/tidy3d.git"
  "warp|https://github.com/NVIDIA/warp.git"
  "fdtdx|https://github.com/ymahlau/fdtdx.git"
  "Khronos.jl|https://github.com/facebookresearch/Khronos.jl.git"
  "GeometryPrimitives.jl|https://github.com/stevengj/GeometryPrimitives.jl.git"
  "VectorModesolver.jl|https://github.com/ianmatthewhammond/VectorModesolver.jl.git"
  "libctl|https://github.com/NanoComp/libctl.git"
  "fdtd-pipeline|https://github.com/JPPhotonics/fdtd-pipeline.git"
)

echo "Repo root: $REPO_ROOT"
echo "Reference checkout root: $PARENT_DIR"
echo

for entry in "${REPOS[@]}"; do
  name="${entry%%|*}"
  url="${entry#*|}"
  target_dir="$PARENT_DIR/$name"

  if [[ -d "$target_dir/.git" ]]; then
    echo "[present] $name -> $target_dir"
    if [[ "$UPDATE_MODE" == "fetch" ]]; then
      git -C "$target_dir" fetch --all --tags --prune
      echo "  fetched latest refs"
    elif [[ "$UPDATE_MODE" == "pull" ]]; then
      branch="$(git -C "$target_dir" rev-parse --abbrev-ref HEAD)"
      git -C "$target_dir" pull --ff-only origin "$branch"
      echo "  pulled origin/$branch"
    else
      echo "  leaving existing checkout unchanged"
    fi
  else
    echo "[clone] $url -> $target_dir"
    git clone "$url" "$target_dir"
  fi
done

echo
echo "Paper references are checked into this repo:"
echo "  $REPO_ROOT/papers/meep_paper.pdf"
echo "  $REPO_ROOT/papers/gpu_benchmarking.pdf"
echo "  $REPO_ROOT/papers/systolic_fdtd.pdf"
