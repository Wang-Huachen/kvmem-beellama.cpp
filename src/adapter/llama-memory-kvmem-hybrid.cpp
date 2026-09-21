#include "llama-memory-kvmem-hybrid.h"

#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <limits>
#include <cstring>
#include <stdexcept>

static bool use_gdn_replay(const llama_model & model, const llama_cparams & cp) {
    const int mode = llama_kvmem_get_params()->mtp_state;
    if (mode == 0) return false;
    const auto & h = model.hparams;
    bool supported = model.arch == LLM_ARCH_QWEN35 && h.n_layer() == 64 && cp.n_seq_max == 1 &&
        cp.n_rs_seq > 0 && cp.n_rs_seq <= 5 && cp.n_ubatch >= cp.n_rs_seq + 1 && cp.offload_kqv && h.ssm_d_inner == 6144 &&
        h.ssm_d_state == 128 && h.ssm_n_group == 16 && h.ssm_dt_rank == 48 && h.ssm_d_conv == 4;
    ggml_backend_dev_t device = nullptr;
    for (uint32_t il = 0; supported && il < h.n_layer(); ++il) {
        if (!h.is_recr(il)) continue;
        auto * dev = model.dev_layer(il);
        supported = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") == 0 && (!device || device == dev);
        device = dev;
    }
    if (!supported && mode == 2) throw std::runtime_error("GDN replay requires single-sequence CUDA Qwen 27B with MTP 1-5 and all recurrent layers on one GPU");
    // Keep automatic selection on snapshots until the replay regression suite passes.
    return supported && mode == 2;
}

llama_memory_kvmem_hybrid::llama_memory_kvmem_hybrid(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams) :
    llama_memory_hybrid(
            model,
            // Attention half: a KVMem slot pool over storage it builds and owns --
            // the KVarN record arena when KVarN is enabled, a plain row cache
            // otherwise (llama_memory_kvmem's constructor picks). The base class
            // takes ownership from here, so this IS the object the graph and the
            // base both see. There is no borrowed cache and no second wrapper.
            std::make_unique<llama_memory_kvmem>(model, params, cparams),
            // Recurrent half: stock, and identical to what
            // llama_memory_hybrid's own scalar constructor would have built.
            // type_r/type_s/offload/mem_size mirror that constructor's arguments.
            std::make_unique<llama_memory_recurrent>(
                    model,
                    /* type_r */ GGML_TYPE_F32,
                    /* type_s */ GGML_TYPE_F32,
                    /* offload */ cparams.offload_kqv,
                    /* mem_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                    /* n_seq_max */ cparams.n_seq_max,
                    /* n_rs_seq */ cparams.n_rs_seq,
                    [&](int32_t il) {
                        return il < (int32_t) model.hparams.n_layer() && model.hparams.is_recr(il);
                    },
                    /* replay */ use_gdn_replay(model, cparams))) {
    attn_kvmem_ = static_cast<llama_memory_kvmem *>(get_mem_attn());
    GGML_ASSERT(attn_kvmem_ != nullptr && "KVMem hybrid needs a llama_memory_kvmem attention memory");
    // The recurrent half is owned by the base; KVMem only needs to see it for the
    // GDN snapshot/rollback paths it drives from the attention side.
    attn_kvmem_->set_recurrent(get_mem_recr());
    LLAMA_LOG_INFO("%s: KVMem hybrid (attn=slot-pool recr=stock) n_rs_seq=%u kvarn=%d\n",
            __func__, cparams.n_rs_seq, attn_kvmem_->get_kvarn() != nullptr ? 1 : 0);
}

bool llama_memory_kvmem_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const llama_pos p0n = p0 < 0 ? 0 : p0;
    const bool full = seq_id <= 0 && p0n == 0 && p1 < 0;
    llama_memory_recurrent * recr = get_mem_recr();
    if (full) {
        if (!recr->seq_rm(seq_id, p0, p1)) {
            return false;
        }
        return attn_kvmem_ ? attn_kvmem_->seq_rm(seq_id, p0, p1)
                           : get_mem_attn()->seq_rm(seq_id, p0, p1);
    }
    // Query-replay holes must not touch GDN (P4-2 restores it separately).
    // MTP verify reject is a short open suffix (length 1..n_rs_seq): roll
    // GDN back on the GPU snapshot planes instead of a 150 MiB host dump.
    const uint32_t n_rs = recr->n_rs_seq;
    const llama_pos rmax = recr->seq_pos_max(seq_id);
    const llama_pos p1x = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
    if (n_rs > 0 && p0n > 0 && rmax >= 0 && p0n <= rmax && p1x > rmax) {
        const llama_pos rollback = rmax - (p0n - 1);
        if (rollback >= 1 && rollback <= (llama_pos) n_rs) {
            if (!recr->seq_rm(seq_id, p0, p1)) {
                return false;
            }
        }
    }
    return attn_kvmem_ ? attn_kvmem_->seq_rm(seq_id, p0, p1)
                       : get_mem_attn()->seq_rm(seq_id, p0, p1);
}
