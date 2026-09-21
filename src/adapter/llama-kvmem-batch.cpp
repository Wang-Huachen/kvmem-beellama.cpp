#include "llama-kvmem-batch.h"

#include "llama-impl.h"

bool kvmem_fill_slot_info(
        const kvmem::KvMemStore & store,
        uint32_t block_tokens,
        uint32_t kv_size,
        const llama_ubatch & ubatch,
        llama_kv_cache::slot_info & out,
        const std::vector<int32_t> & vslot) {
    if (ubatch.n_tokens == 0 || block_tokens == 0 || !ubatch.pos) {
        return false;
    }
    // Logical rows stay unique when image patches share M-RoPE positions.

    // TODO(rebase): beellama's slot_info also carries stage_slots and
    // group_stage_slots for structured caches. KVMem builds the slot layout
    // itself from its own block pool, so both stay empty here. beellama guards
    // every read of them, but this must be revisited if KVMem ever runs on a
    // structured (KVarN) cache, where the stage slots are load-bearing.
    //
    // KVarN status: KVMem is now allowed to run on a KVarN cache, and leaving
    // these empty is still safe. The only reader of host-chosen stage slots sits
    // behind llama_kv_cache_kvarn::uses_compact_read_indices(), which is
    // `!swa && n_stream == 1 && n_seq_max > 1`; KVMem pins n_seq_max to 1, so it
    // is always false. init_batch() asserts exactly that before handing this
    // slot_info to the arena.

    out.s0 = 0;
    out.s1 = 0;
    out.resize(1);
    out.strm[0] = 0;
    out.idxs[0].clear();
    out.idxs[0].reserve(ubatch.n_tokens);

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id && ubatch.n_seq_id[i] > 1) {
            LLAMA_LOG_ERROR("%s: KVMem P1 is single-sequence only\n", __func__);
            return false;
        }
        const llama_pos pos = ubatch.logical_pos ? ubatch.logical_pos[i] : ubatch.pos[i];
        if (pos < 0) {
            LLAMA_LOG_ERROR("%s: negative pos at token %u\n", __func__, i);
            return false;
        }
        const int32_t bid = store.block_id_containing(static_cast<uint32_t>(pos));
        if (bid < 0) {
            LLAMA_LOG_ERROR("%s: no KVMem block for pos %d\n", __func__, (int) pos);
            return false;
        }
        const kvmem::KvMemBlock & blk = store.blocks()[static_cast<uint32_t>(bid)];
        if (blk.gpu_slot < 0) {
            LLAMA_LOG_ERROR("%s: block %u has no GPU slot (pos %d)\n", __func__, blk.block_id, (int) pos);
            return false;
        }
        const uint32_t off = static_cast<uint32_t>(pos) - blk.orig_pos_start;
        if (off >= blk.n_tokens || off >= block_tokens) {
            LLAMA_LOG_ERROR("%s: pos %d out of block %u range\n", __func__, (int) pos, blk.block_id);
            return false;
        }
        // KVarN infers "which group is still being written" from the numeric MAXIMUM
        // of the indices it is handed (kvarn.cu:2149), and the read side uses that to
        // decide between a sealed record and the F16 stage
        // (fattn-mma-kvarn.cuh:77-90). Physical slots do not track write order --
        // eviction can hand the newest block a low slot -- so hand KVarN the virtual
        // number instead, which is monotonic in admission order. Row mode passes an
        // empty table and keeps using the physical slot, so the configuration that
        // works today is untouched.
        int32_t v = -1;
        if (blk.block_id < vslot.size()) {
            v = vslot[blk.block_id];
        }
        if (v < 0) {
            v = blk.gpu_slot;   // row mode, or a block that is not mapped (yet)
        }
        const uint32_t cell = static_cast<uint32_t>(v) * block_tokens + off;
        if (cell >= kv_size) {
            LLAMA_LOG_ERROR("%s: cell %u >= kv_size %u (slot %d, virtual %d)\n",
                    __func__, cell, kv_size, (int) blk.gpu_slot, (int) v);
            return false;
        }
        out.idxs[0].push_back(cell);
    }

    return out.idxs[0].size() == ubatch.n_tokens;
}
