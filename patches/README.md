# llama.cpp patch replay (beellama.cpp base)

`llama-kvmem-current.patch` is the cumulative KVMem diff against the pinned
**Anbeeld/beellama.cpp v0.4.6** submodule (`78af8326`). It carries the KVMem
hook set (memory factory hook, attention Q/K capture hooks, skip-hole purge,
CUDA stage-in, multimodal batch, MTP, media parser and mtmd helper extensions,
FP32 GDN Record/Fold for ReplaySSM, reasoning-budget initialization), rebased
onto beellama's rewritten `llama_kv_cache`, **plus** the KVarN additions this
fork needs (`get_records` / `move_record_group` on `llama_kv_cache_kvarn`).

`scripts/apply-patches.sh` applies it without creating commits and detects an
already-patched tree. `scripts/windows/build.ps1` applies it automatically
before configuring.

The previous patch set for the old `ggml-org/llama.cpp` base (pin `b81c99b`) is
kept under `patches/legacy-ggml-org/` for reference only. **Do not apply it on
the beellama base** — the cumulative patch above supersedes it (the old series
already conflicted on its own pin).

To check a clean extraction without touching the working submodule:

```bash
mkdir -p /tmp/kvmem-beellama-patch-check
git -C llama.cpp archive 78af8326 | tar -x -C /tmp/kvmem-beellama-patch-check
KVMEM_LLAMA_DIR=/tmp/kvmem-beellama-patch-check scripts/apply-patches.sh
KVMEM_LLAMA_DIR=/tmp/kvmem-beellama-patch-check scripts/apply-patches.sh
```
