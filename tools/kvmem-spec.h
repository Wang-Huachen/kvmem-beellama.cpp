#pragma once

// Shared draft-mtp wiring for llama-kvmem-cli / llama-kvmem-server.
// Does not reimplement llama.cpp's speculative state machine.

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <functional>
#include <string>
#include <vector>

struct kvmem_spec_opts {
    int32_t n_max = 2;
    int32_t n_min = 0;
    float   p_min = 0.0f;
    int32_t n_gpu_layers = 99;
    int32_t n_ctx = 0;
    int32_t n_batch = 512;
    int32_t n_ubatch = 512;
    int32_t n_threads = -1;
    int32_t n_threads_batch = -1;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
    bool    kvmem_enabled = false;
    std::string draft_model; // optional sidecar GGUF; empty = welded nextn
    ggml_type type_k = GGML_TYPE_Q8_0;
    ggml_type type_v = GGML_TYPE_Q8_0;
    ggml_type draft_type = GGML_TYPE_COUNT; // inherit target K/V types unless overridden
    // KVarN record-arena intent (beellama's K/V cache format). DISABLED keeps the
    // plain ggml row cache, i.e. the behaviour before this option existed.
    //
    // Target and MTP draft carry SEPARATE intent on purpose: beellama audits the
    // draft route per architecture (llama-kvarn.cpp llama_kvarn_context_route_for;
    // QWEN35/QWEN35MOE/QWEN4EXP are draft-owned), so one half may be KVarN while
    // the other stays on rows.
    llama_kvarn_params kvarn = {};
    llama_kvarn_params draft_kvarn = {};
};

// Supported cache types: f16, f32, q8_0, q5_0, q4_0.
// q5_0 requires GGML_CUDA_FA_ALL_QUANTS (enabled by this project's build).
ggml_type kvmem_parse_cache_type(const char * s, bool * ok);
// Quantized K/V may independently use q8_0, q5_0 or q4_0; no float/quantized mixing.
// With KVarN on, type_k/type_v are unused (the record arena owns the bytes), so
// callers must skip this check in that case.
bool kvmem_cache_types_ok(ggml_type type_k, ggml_type type_v);

// Parses beellama's KVarN cache names ("kvarn2".."kvarn8" and the canonical
// "kvarn_k4v4_g128" spelling) plus an explicit "off"/"none"/"disabled".
// Returns true when the string denotes a KVarN request (including an explicit
// off), and false when it is not a KVarN name at all, so that the ordinary ggml
// type table can handle it.
//
// NOTE: llama_kvarn_type_from_name() returns LLAMA_KVARN_TYPE_INVALID (-1) for
// unknown names, NOT DISABLED (0); the helper normalises INVALID -> DISABLED.
bool kvmem_parse_kvarn(const char * s, llama_kvarn_params * out);

// Resolves the KVarN K/V pair the way the cache actually supports it.
//
//   bits_k/bits_v   : KVarN bit width on each side (0 = no KVarN named there)
//   plain_k/plain_v : the caller saw an EXPLICIT plain ggml type on that side
//
// ASYMMETRIC WIDTHS ARE SUPPORTED. llama.h spells out all 36 ordered pairs
// (LLAMA_KVARN_K4V2_G128, K6V2, K2V8, ...), so "kvarn4 on K, kvarn2 on V" is a
// first-class cache format and the two sides must be COMBINED, never silently
// collapsed onto one width. beellama does the same (common/arg.cpp:1436-1473,
// common_kvarn_pair_normalize).
//
// A KVarN / plain-type MIX is NOT supported: the record arena stores K and V
// together, so one side cannot stay in ggml rows. beellama warns and upgrades the
// plain side; this helper refuses instead, because silently changing a type the
// user spelled out is exactly the surprise this project's rules forbid.
//
// Returns false and sets `err` to a static message when the pair is unsupported.
bool kvmem_kvarn_pair(int32_t bits_k, int32_t bits_v, bool plain_k, bool plain_v,
                      llama_kvarn_params * out, const char ** err);

struct kvmem_spec_session {
    common_params spec_params;
    common_speculative_init_result_ptr init;
    common_speculative * spec = nullptr;
    llama_context * ctx_dft = nullptr;
    bool use_ckpt_tgt = false;
    bool use_ckpt_dft = false;
    bool use_gdn_replay = false;
    uint32_t n_rs_tgt = 0;
    bool ok = false;
};

bool kvmem_spec_start(kvmem_spec_session & sess,
                      llama_model * model_tgt,
                      llama_context * ctx_tgt,
                      const kvmem_spec_opts & opts);

void kvmem_spec_stop(kvmem_spec_session & sess);

// decode_span / spec_decode_span: client disconnected mid-prefill.
constexpr int KVMEM_DECODE_ABORT = -100;

// Decode [pos0, pos1) with explicit pos/seq_id so process() can catch up draft KV.
// `abort` if set: return KVMEM_DECODE_ABORT when it returns true (checked between batches).
int kvmem_spec_decode_span(llama_context * ctx,
                           common_speculative * spec,
                           const llama_token * toks,
                           int pos0, int pos1, int n_batch,
                           const char * what,
                           const std::function<bool()> & abort = {});

using kvmem_spec_on_token =
        std::function<void(llama_token id, const std::string & piece, bool from_draft)>;

struct kvmem_spec_gen_stats {
    int n_past = 0;
    int n_gen = 0;
    int n_drafted = 0;
    int n_accept = 0;
    int n_restore = 0;
    bool failed = false;
};

// `prompt` is the full prompt; the last token is id_last and must not already
// be in the target KV (same contract as llama.cpp speculative-simple).
kvmem_spec_gen_stats kvmem_spec_generate(
        llama_context * ctx_tgt,
        llama_model * model_tgt,
        kvmem_spec_session & sess,
        const std::vector<llama_token> & prompt,
        int n_predict,
        float temp,
        kvmem_spec_on_token on_token);

// Same as above, but uses a fully-specified sampler (grammar / lazy triggers).
// `abort` if set: stop the generate loop when it returns true (not a failure).
kvmem_spec_gen_stats kvmem_spec_generate(
        llama_context * ctx_tgt,
        llama_model * model_tgt,
        kvmem_spec_session & sess,
        const std::vector<llama_token> & prompt,
        int n_predict,
        common_params_sampling sparams,
        kvmem_spec_on_token on_token,
        const std::function<bool()> & abort = {},
        llama_pos position_offset = 0);
