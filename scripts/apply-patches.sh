#!/usr/bin/env bash
# Replay the maintained KVMem diff against the pinned beellama.cpp submodule.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="${KVMEM_LLAMA_DIR:-$ROOT/llama.cpp}"
PATCH="$ROOT/patches/llama-kvmem-current.patch"

cd "$LLAMA"

if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "already applied: $PATCH"
    exit 0
fi

git apply --check "$PATCH"
git apply "$PATCH"
echo "applied: $PATCH"
