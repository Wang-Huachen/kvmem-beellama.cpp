#pragma once

#include "llama-memory-hybrid.h"
#include "llama-memory-kvmem.h"

#include <memory>

// llama_memory_hybrid whose ATTENTION half IS a llama_memory_kvmem and whose
// recurrent half is the stock llama_memory_recurrent.
//
// Composition, not a wrapper: the attention memory is built here and MOVED INTO
// llama_memory_hybrid, which owns it. The base then calls
// mem_attn->init_kv_batch() -> llama_memory_kvmem::init_kv_batch(), i.e. KVMem's
// slot pool, its virtual slot numbering and (when KVarN is on) its record arena
// are all on the path the hybrid context actually uses.
//
// This replaced an earlier design that inherited the base and borrowed the stock
// llama_kv_cache the base had built. That arrangement had two fatal properties:
// the base's init_kv_batch() never reached KVMem, and the KVarN arena was out of
// reach entirely, so KVMem had to refuse hybrid + KVarN. There must be exactly one
// attention object -- no second, disconnected cache.
//
// Graph still static_cast's to llama_memory_hybrid_context. GDN sees every
// token. Query-replay holes must not roll back GDN. MTP verify reject of a
// short suffix (<= n_rs_seq) uses GPU snapshot planes.
class llama_memory_kvmem_hybrid : public llama_memory_hybrid {
public:
    llama_memory_kvmem_hybrid(
            const llama_model & model,
            const llama_memory_params & params,
            const llama_cparams & cparams);

    // Deliberately NOT overriding init_batch()/clear()/get_can_shift(): the base
    // implementations now do the right thing through the attention memory.
    //   init_batch()  splits the ubatches with the same GDN-rollback rule and then
    //                 calls mem_attn->init_kv_batch(), which is KVMem.
    //   clear()       mem_attn->clear() is llama_memory_kvmem::clear(), which
    //                 resets the KVMem policy itself.
    //   get_can_shift() forwards to the attention memory, which returns false.
    // The only behaviour that is genuinely KVMem-hybrid specific is seq_rm(): the
    // recurrent half must not be rolled back for query-replay holes.
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;

    // The attention half. Non-null and owned by the base class; never free it.
    llama_memory_kvmem * attn_kvmem() { return attn_kvmem_; }

private:
    llama_memory_kvmem * attn_kvmem_ = nullptr;
};
