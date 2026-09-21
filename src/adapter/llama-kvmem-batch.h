#pragma once

#include "llama-batch.h"
#include "llama-kv-cache.h"

#include "kvmem/kvmem_store.hpp"

#include <cstdint>
#include <vector>

// Map each ubatch token to a cell in its block's GPU slot.
// cell = slot_for(block) * block_tokens + (orig_pos - orig_pos_start)
// Slot number is not the window RoPE coordinate; cell.pos stays the original
// (monotonic) token position. Dim 0 of M-RoPE batches is that sequential pos.
//
// `vslot` is the virtual-slot table (block_id -> slot). Under KVarN the number handed
// to the arena must be monotonic in admission order rather than the physical slot, for
// the reason spelled out at the call site in llama-kvmem-batch.cpp. Pass an empty
// vector for row mode, where the physical slot is used directly.
bool kvmem_fill_slot_info(
        const kvmem::KvMemStore & store,
        uint32_t block_tokens,
        uint32_t kv_size,
        const llama_ubatch & ubatch,
        llama_kv_cache::slot_info & out,
        const std::vector<int32_t> & vslot);
