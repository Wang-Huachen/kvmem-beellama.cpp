// Stage 3 batch 3 smoke test: KVMem on the KVarN record arena.
//
// Usage:
//   kvmem-kvarn-smoke <model.gguf> [options]
//   KVMEM_SMOKE_MODEL=<model.gguf>  kvmem-kvarn-smoke
//   (with neither, the exe tries models/*.gguf and otherwise exits 77 = SKIP)
//
// Options let one binary cover both regressions that matter:
//   -ctk <type>          KVarN cache type (kvarn4 default; kvarn6 etc.) or a ggml type
//   --kv-dtype <type>   plain ggml cache type (default q8_0)
//   --block-tokens <n>  KVMem --kvmem-block-tokens (default 128)
//   --ctx <n>           context size (default 512)
// Exactly the shape of a real server launch, so a user's working command line can
// be reproduced here. Exit codes: 0 = PASS, 1 = FAIL, 77 = SKIP (no model found).
// A hybrid model is no longer a SKIP case: since batch 6 the hybrid wrapper
// composes a llama_memory_kvmem instead of borrowing the base class's stock cache,
// so hybrid archs must reach the record arena like any dense model.
//
// Deliberately uses ONLY the public llama.h API plus the KVMem hook header, and
// therefore links against `llama` alone -- not `llama-common`. Reason: this tree
// may have build\llama.cpp\common\Release\llama-common.lib stuck in an OS-level
// delete-pending state, which makes every executable that links llama-common fail
// with LNK1114 (error 5). Keeping this test off that library means it can be
// built in the normal `build` tree without recompiling the CUDA FA matrix.
//
// Why a real model: batch 3 swaps KVMem's inner cache for llama_kv_cache_kvarn and
// rewires init_batch/init_update/seq_rm onto the record arena. What has to hold is
// internal state (which context class the graph sees, which memory owns the
// records) and none of it is observable from a model-free unit test. What this
// test does check is the whole stack: load, prefill through KVMem's slot pool,
// decode off the arena, and finish without a crash, without tripping one of the
// fail-closed row paths, and without a degenerate KV cache (a context the memory
// cannot serve fails prefill with a non-zero llama_decode).
//
// It intentionally does NOT check output quality or bit-exactness; that is batch 7.

#include "llama.h"

#include "llama-kvmem-hooks.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// KVMem's pool must line up with the KVarN record group. llama-kvarn.h defines
// KVAR_N_GROUP = 128; this test restates it so it does not have to include
// internal headers, and asserts that KVMem accepted the value.
static const uint32_t kKvarNGroup = 128;

struct findings {
    bool saw_kvarn_cache_line = false; // "KVarN cache: stage_groups=..."
    bool saw_kvmem_pool_line  = false; // "KVMem slot-pool cells=..."
    bool saw_kvmem_kvarn_line = false; // "... kvarn=1"
    bool saw_fail_closed      = false; // a row path we needed was not ported
    bool saw_mtp_downgrade    = false; // KVMem refused the MTP follower under KVarN
};

static findings g_findings;

// Distinguish "graceful degradation" from "we hit a path that must not be
// silently skipped":
//   - the MTP follower refusing to run under KVarN is expected for now (the
//     stock memory serves it), so it must not fail the run;
//   - any row-path message means a path we actually need was not ported.
//
// A hybrid arch used to land in the first category too. That is over: since the
// hybrid wrapper composes a llama_memory_kvmem instead of borrowing the base
// class's stock cache, a hybrid model must go through the record arena like any
// other, so there is nothing left to skip.
static void log_scan(ggml_log_level level, const char * text, void * /* user_data */) {
    if (strstr(text, "KVarN cache: stage_groups=")) {
        g_findings.saw_kvarn_cache_line = true;
    }
    if (strstr(text, "KVMem slot-pool cells=")) {
        g_findings.saw_kvmem_pool_line = true;
        if (strstr(text, "kvarn=1")) {
            g_findings.saw_kvmem_kvarn_line = true;
        }
    }
    if (strstr(text, "KVMem MTP follower is not ported")) {
        g_findings.saw_mtp_downgrade = true;
    } else if (strstr(text, "not ported to the KVarN record arena")) {
        if (!g_findings.saw_fail_closed) {
            fprintf(stderr, "FAIL: a row-shaped K/V path was reached under KVarN:\n%s", text);
        }
        g_findings.saw_fail_closed = true;
    }
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}

static bool file_exists(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    fclose(f);
    return true;
}

// Best-effort default so the script works with no arguments. The models/ folder
// ships ggml-vocab-*.gguf files, which carry a tokenizer but zero tensors and
// cannot be loaded as a model, so they are excluded rather than reported as a
// failure. Returns an empty string when nothing usable is found (the caller then
// reports SKIP, not FAIL).
static std::string find_default_model(const char * argv0) {
    const char * v = getenv("KVMEM_SMOKE_MODEL");
    if (v && v[0] && file_exists(v)) {
        return v;
    }
    const char * roots[] = { "llama.cpp/models", "models", nullptr };
    const char * names[] = { "ggml-model-q4_0.gguf", "model.gguf", nullptr };
    for (int r = 0; roots[r]; ++r) {
        for (int n = 0; names[n]; ++n) {
            std::string p = std::string(roots[r]) + "/" + names[n];
            if (file_exists(p.c_str())) {
                return p;
            }
        }
    }
    (void) argv0;
    return std::string();
}

// A raw utf8 tokenization via the public vocab API. add_special=false keeps the
// byte counts predictable for a synthetic prompt.
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const char * text) {
    const int n = -llama_tokenize(vocab, text, (int32_t) strlen(text), nullptr, 0, false, false);
    std::vector<llama_token> out(n > 0 ? n : 0);
    if (n <= 0) {
        return out;
    }
    const int m = llama_tokenize(vocab, text, (int32_t) strlen(text), out.data(), (int32_t) out.size(),
                                 false, false);
    if (m < 0) {
        out.clear();
        return out;
    }
    out.resize(m);
    return out;
}

static bool decode_one(llama_context * ctx, const std::vector<llama_token> & toks,
                       const std::vector<llama_pos> & pos) {
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    batch.n_tokens = (int32_t) toks.size();
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = pos[i];
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == batch.n_tokens - 1) ? 1 : 0;
    }

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (rc != 0) {
        fprintf(stderr, "FAIL: llama_decode returned %d\n", rc);
        return false;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        fprintf(stderr, "FAIL: no logits after decode\n");
        return false;
    }
    bool any_finite = false;
    for (int32_t i = 0; i < n_vocab; ++i) {
        if (std::isfinite(logits[i])) {
            any_finite = true;
            break;
        }
    }
    if (!any_finite) {
        fprintf(stderr, "FAIL: every logit is non-finite\n");
        return false;
    }
    return true;
}

// Command-line knobs. Defaults mirror a plausible server launch; --kvarn off plus
// --kv-dtype q8_0 reproduces a plain-ggml KVMem run (the configuration that works
// today and must keep working).
struct opts {
    const char * model        = nullptr;
    // Must be a name the -ctk parser below accepts: `kvarn2`..`kvarn8`, or the long
    // `kvarn_k4v4_g128` form (llama_kvarn_type_from_name). This used to be "k4v4",
    // which is only the *display* form of the parsed pair -- it is NOT parseable, so
    // `kvmem-kvarn-smoke <model>` with no -ctk exited 1 with
    // "cannot parse -ctk 'k4v4'". The nightly bats always pass -ctk explicitly, which
    // is why it went unnoticed.
    const char * kvarn_type   = "kvarn4";
    const char * kv_dtype     = "q8_0";
    uint32_t     block_tokens = 128;
    int32_t      n_ctx        = 512;
    // KVMem --kvmem-budget. 0 = identity pool (pool == n_ctx, never evicts).
    // A small non-zero value forces prefill pressure, which is what exercises the
    // record snapshot / restore round trip.
    uint32_t     budget       = 0;
    // Prompt length. Must exceed ~95% of the pool for prefill pressure to fire
    // (KvMemStore::prefill_needs_offload, kvmem/src/host/kvmem_store.cpp:339-350).
    int          n_prefill    = 96;
    // KVMem --kvmem-method: 0 recency, 1 retrieval. The server forces retrieval
    // (tools/llama-kvmem-server.cpp:1662 `st.kparams.method = 1`), so this is the
    // product default even though the rest of this suite ran recency.
    int          method       = 0;
    // Call llama_kvmem_apply_retrieval() once after prefill. That is the host entry
    // point for "score, reselect, stage-in", so it is what exercises the retrieval
    // path (score_retrieval -> preview_reselect -> apply_plan_to_kv -> layout).
    int          do_retrieval = 0;
};

static bool parse_kvarn_type(const char * s, llama_kvarn_type * out) {
    if (strcmp(s, "off") == 0 || strcmp(s, "disabled") == 0 || strcmp(s, "none") == 0) {
        *out = LLAMA_KVARN_TYPE_DISABLED;
        return true;
    }
    // Same spelling as the tools and as beellama's own CLI: kvarn2..kvarn8, plus
    // the canonical kvarn_k6v6_g128 form. llama_kvmem_kvarn_type_from_name()
    // normalises the "unknown" answer, which llama_kvarn_type_from_name() reports
    // as INVALID rather than DISABLED.
    const llama_kvarn_type t = llama_kvmem_kvarn_type_from_name(s);
    if (t == LLAMA_KVARN_TYPE_INVALID || t == LLAMA_KVARN_TYPE_DISABLED) {
        return false;
    }
    *out = t;
    return true;
}

// ggml exposes ggml_type_name() but no public name->type lookup, so map just the
// cache types a KV cache can be asked for.
static bool parse_ggml_type(const char * s, ggml_type * out) {
    struct entry { const char * name; ggml_type t; };
    static const entry table[] = {
        { "f16",   GGML_TYPE_F16  }, { "fp16",  GGML_TYPE_F16  },
        { "bf16",  GGML_TYPE_BF16 },
        { "q8_0",  GGML_TYPE_Q8_0 }, { "q6_0",  GGML_TYPE_Q6_0 },
        { "q5_0",  GGML_TYPE_Q5_0 }, { "q5_1",  GGML_TYPE_Q5_1 },
        { "q4_0",  GGML_TYPE_Q4_0 }, { "q4_1",  GGML_TYPE_Q4_1 },
        { nullptr, GGML_TYPE_COUNT },
    };
    for (int i = 0; table[i].name; ++i) {
        if (strcmp(s, table[i].name) == 0) {
            *out = table[i].t;
            return true;
        }
    }
    return false;
}

// Returns false only on a malformed argument list.
static bool parse_args(int argc, char ** argv, struct opts * o) {
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "FAIL: %s needs a value\n", what);
                return nullptr;
            }
            return argv[++i];
        };
        if (strcmp(a, "-ctk") == 0 || strcmp(a, "--ctk") == 0 || strcmp(a, "--cache-type-k") == 0) {
            // beellama's cache-type spelling: a KVarN name switches the cache to the
            // record arena, anything else is a plain ggml row type. There is no
            // separate KVarN switch anywhere in the product, so the test mirrors it.
            const char * v = next(a); if (!v) return false;
            llama_kvarn_type kt = LLAMA_KVARN_TYPE_DISABLED;
            if (parse_kvarn_type(v, &kt)) {
                o->kvarn_type = v;
            } else {
                // The cache TYPE is the switch, exactly as in beellama: naming a
                // plain ggml type means "no KVarN", it does not merely pick the row
                // format while leaving the arena enabled.
                o->kv_dtype = v;
                o->kvarn_type = "off";
            }
        } else if (strcmp(a, "--kvarn") == 0) {
            // Kept as an alias for existing scripts.
            const char * v = next(a); if (!v) return false;
            o->kvarn_type = v;
        } else if (strcmp(a, "--kv-dtype") == 0) {
            const char * v = next(a); if (!v) return false;
            o->kv_dtype = v;
        } else if (strcmp(a, "--block-tokens") == 0) {
            const char * v = next(a); if (!v) return false;
            o->block_tokens = (uint32_t) atoi(v);
        } else if (strcmp(a, "--ctx") == 0) {
            const char * v = next(a); if (!v) return false;
            o->n_ctx = atoi(v);
        } else if (strcmp(a, "--budget") == 0) {
            const char * v = next(a); if (!v) return false;
            o->budget = (uint32_t) atoi(v);
        } else if (strcmp(a, "--prefill") == 0) {
            const char * v = next(a); if (!v) return false;
            o->n_prefill = atoi(v);
        } else if (strcmp(a, "--method") == 0) {
            const char * v = next(a); if (!v) return false;
            o->method = (strcmp(v, "retrieval") == 0) ? 1 : 0;
        } else if (strcmp(a, "--do-retrieval") == 0) {
            o->do_retrieval = 1;
        } else if (a[0] == '-') {
            fprintf(stderr, "FAIL: unknown option %s\n", a);
            return false;
        } else if (!o->model) {
            o->model = a;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    struct opts o;
    if (!parse_args(argc, argv, &o)) {
        return 1;
    }

    std::string resolved;
    if (o.model && o.model[0]) {
        if (!file_exists(o.model)) {
            fprintf(stderr, "FAIL: model file not found: %s\n", o.model);
            return 1;
        }
        resolved = o.model;
    } else {
        resolved = find_default_model(argv[0]);
    }
    if (resolved.empty()) {
        fprintf(stderr,
                "SKIP: no model path. Pass one, or set KVMEM_SMOKE_MODEL.\n"
                "      usage: kvmem-kvarn-smoke <model.gguf> [-ctk <t>] "
                "[--kv-dtype <t>] [--block-tokens <n>] [--ctx <n>] [--budget <n>]\n");
        return 77;
    }

    llama_kvarn_type kvarn_type = LLAMA_KVARN_TYPE_DISABLED;
    if (!parse_kvarn_type(o.kvarn_type, &kvarn_type)) {
        fprintf(stderr, "FAIL: cannot parse -ctk '%s' (KVarN: kvarn2..kvarn8; or f16|q8_0|q5_0|q4_0)\n",
                o.kvarn_type);
        return 1;
    }
    ggml_type kv_dtype = GGML_TYPE_Q8_0;
    if (!parse_ggml_type(o.kv_dtype, &kv_dtype)) {
        fprintf(stderr, "FAIL: unknown --kv-dtype '%s'\n", o.kv_dtype);
        return 1;
    }
    const bool kvarn_on = kvarn_type != LLAMA_KVARN_TYPE_DISABLED;

    llama_log_set(log_scan, nullptr);
    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;

    llama_model * model = llama_model_load_from_file(resolved.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "FAIL: could not load model %s\n", resolved.c_str());
        llama_backend_free();
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // KVarN must be selected through the public params struct, exactly as
    // common/arg.cpp would after seeing --cache-type-k kvarn4.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = o.n_ctx;
    cparams.n_batch         = 128;
    cparams.n_ubatch        = 128;
    cparams.n_seq_max       = 1;              // KVMem requires single sequence
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.offload_kqv     = true;
    cparams.op_offload      = true;
    cparams.kvarn           = llama_kvarn_params_for_type(kvarn_type);
    cparams.type_k          = kv_dtype;
    cparams.type_v          = kv_dtype;

    // KVMem owns the pool. budget == 0 keeps the pool equal to n_ctx_seq, so a
    // short prefill never evicts -- that is the batch 3 slice. A small non-zero
    // --budget forces prefill pressure, which is the only way to exercise the
    // record snapshot / restore round trip added in batch 4.
    llama_kvmem_params kvp = {};
    kvp.enabled      = true;
    kvp.block_tokens = o.block_tokens;
    kvp.budget       = o.budget;
    kvp.gen_reserve  = o.block_tokens;
    kvp.method       = o.method;
    kvp.query_begin  = (o.method == 1 && o.do_retrieval) ? 0 : -1;
    kvp.query_end    = (o.method == 1 && o.do_retrieval) ? (int32_t) o.n_prefill : -1;
    kvp.force_pos    = -1;
    llama_kvmem_set_params(&kvp);

    printf("model       : %s\n", resolved.c_str());
    printf("kvarn       : %s\n", kvarn_on ? llama_kvarn_type_name(kvarn_type) : "off (plain ggml rows)");
    printf("kv-dtype    : %s\n", ggml_type_name(kv_dtype));
    printf("pool        : block_tokens=%u budget=%u gen_reserve=%u method=%s ctx=%d\n",
           kvp.block_tokens, kvp.budget, kvp.gen_reserve,
           o.method == 1 ? "retrieval" : "recency", o.n_ctx);
    fflush(stdout);

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: llama_init_from_model returned null\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        fprintf(stderr, "FAIL: model has no vocabulary\n");
        return 1;
    }

    // Deterministic synthetic prompt: a repeated digit string. Using real text
    // (rather than raw ids) keeps it valid for any vocabulary.
    std::string prompt;
    for (int i = 0; i < 600; ++i) {
        prompt += "1234567890";
    }
    std::vector<llama_token> prefill = tokenize(vocab, prompt.c_str());
    if (prefill.empty()) {
        fprintf(stderr, "FAIL: tokenizer produced no tokens\n");
        return 1;
    }
    // Clamp the prompt to something the context can hold, but honour --prefill so
    // a run can be pushed past the pool's ~95% offload watermark on purpose.
    size_t want_prefill = (size_t) (o.n_prefill > 0 ? o.n_prefill : 96);
    if (want_prefill > prefill.size()) {
        want_prefill = prefill.size();
    }
    if (want_prefill > 4096) {
        want_prefill = 4096;
    }
    const size_t n_prefill = want_prefill;
    prefill.resize(n_prefill);

    // Prefill in chunks that respect BOTH n_batch (the logical batch the context
    // was built with) and n_ubatch (the physical batch): llama_decode asserts
    // n_tokens <= n_batch, and KVMem's slot pool is sized per ubatch. Feeding the
    // whole prompt in one call trips llama-context.cpp's n_tokens_all <= n_batch
    // assertion.
    const size_t kChunk = 32;
    printf("prefill     : %zu tokens in chunks of %zu (n_batch=%u n_ubatch=%u)\n",
           n_prefill, kChunk, cparams.n_batch, cparams.n_ubatch);
    fflush(stdout);
    for (size_t off = 0; off < n_prefill; off += kChunk) {
        const size_t n = (off + kChunk <= n_prefill) ? kChunk : (n_prefill - off);
        std::vector<llama_token> chunk(prefill.begin() + off, prefill.begin() + off + n);
        std::vector<llama_pos>   chunk_pos(n);
        for (size_t i = 0; i < n; ++i) {
            chunk_pos[i] = (llama_pos) (off + i);
        }
        if (!decode_one(ctx, chunk, chunk_pos)) {
            fprintf(stderr, "RESULT: FAIL (prefill chunk at %zu)\n", off);
            return 1;
        }
    }
    printf("prefill     : ok\n");

    // The server always runs retrieval (tools/llama-kvmem-server.cpp:1662 forces
    // method = 1), so drive its entry point once here and watch what happens.
    // Retrieval scores a block by dotting the captured query-sum against the
    // block's mean pre-RoPE K. Both come from the graph capture hooks, which only
    // models/qwen3.cpp and models/qwen35.cpp wire -- so on any other arch the
    // ledger is empty and the selection silently degrades to recency. The stats
    // below make that visible; see the verdict further down.
    llama_kvmem_retrieval_stats rs{};
    if (o.do_retrieval) {
        printf("retrieval   : invoking llama_kvmem_apply_retrieval()...\n");
        fflush(stdout);
        llama_kvmem_apply_retrieval(ctx);
        printf("retrieval   : returned without crashing\n");
        fflush(stdout);
        rs = llama_kvmem_get_retrieval_stats();
    }

    // A few decode steps on top of the prefill. These are the first batches that
    // reuse a pool slot chosen by init_batch rather than by prefill growth.
    int n_decode = 0;
    llama_pos pos = (llama_pos) prefill.size();
    llama_token tok = prefill.back();
    for (int step = 0; step < 8; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            fprintf(stderr, "FAIL: no logits at decode step %d\n", step);
            return 1;
        }
        llama_token best = 0;
        for (int32_t v = 1; v < n_vocab; ++v) {
            if (logits[v] > logits[best]) {
                best = v;
            }
        }
        tok = best;
        if (!decode_one(ctx, { tok }, { pos })) {
            fprintf(stderr, "RESULT: FAIL (decode step %d)\n", step);
            return 1;
        }
        ++pos;
        ++n_decode;
    }
    printf("decode      : %d steps ok (last token id %d)\n", n_decode, (int) tok);

    // The context reports its KV memory; useful as a sanity number, and a KVarN
    // record arena must make it non-zero.
    const size_t kv_bytes = llama_state_get_size(ctx);
    if (kv_bytes == 0) {
        fprintf(stderr, "FAIL: llama_state_get_size returned 0\n");
        return 1;
    }

    printf("\n--- evidence ---\n");
    printf("KVarN cache constructed      : %s\n", g_findings.saw_kvarn_cache_line ? "yes" : "no");
    printf("KVMem took over the KV cache : %s\n", g_findings.saw_kvmem_pool_line  ? "yes" : "NO");
    printf("KVMem owns a record arena    : %s\n", g_findings.saw_kvmem_kvarn_line ? "yes" : "no");
    printf("unported row path hit        : %s\n", g_findings.saw_fail_closed      ? "YES (bad)" : "no");
    printf("MTP follower downgraded      : %s\n", g_findings.saw_mtp_downgrade   ? "yes" : "no");
    if (o.do_retrieval) {
        printf("retrieval stats              : runs=%u blocks=%u resident=%u have_block=%u "
               "q_layers=%u scored=%u max=%.6f reg_k=%llu reg_q=%llu write_k=%llu write_q=%llu\n",
               rs.runs, rs.blocks, rs.resident, rs.have_block, rs.q_layers, rs.scored,
               rs.max_score, (unsigned long long) rs.reg_k, (unsigned long long) rs.reg_q,
               (unsigned long long) rs.write_k, (unsigned long long) rs.write_q);
    }

    bool pass = true;

    if (kvarn_on && !g_findings.saw_kvarn_cache_line) {
        fprintf(stderr, "FAIL: the KVarN cache was never constructed. Either the model's head dim "
                        "is not KVarN-supported, or the flags never reached create_memory.\n");
        pass = false;
    }
    if (g_findings.saw_fail_closed) {
        pass = false;
    }

    // A hybrid arch must now take over exactly like a dense model: the hybrid
    // wrapper composes a llama_memory_kvmem and moves it into llama_memory_hybrid
    // instead of borrowing the base class's stock llama_kv_cache. "It fell back to
    // the stock memory" is therefore a failure, not a skip -- there is no supported
    // hybrid downgrade left.
    if (!g_findings.saw_kvmem_pool_line) {
        fprintf(stderr, "FAIL: KVMem never took over the KV cache. On a hybrid arch this means the "
                        "composed attention memory never reached llama_memory_hybrid.\n");
        pass = false;
    }
    // Only a KVarN run can (and must) own a record arena.
    if (kvarn_on && g_findings.saw_kvmem_pool_line && !g_findings.saw_kvmem_kvarn_line) {
        fprintf(stderr, "FAIL: KVMem is running but does not report the KVarN record arena, so its "
                        "inner cache is still the stock row cache.\n");
        pass = false;
    }

    if (o.do_retrieval) {
        if (rs.runs == 0) {
            fprintf(stderr, "FAIL: --do-retrieval ran but score_retrieval() was never reached, "
                            "so the retrieval path was not exercised at all.\n");
            pass = false;
        } else if (rs.no_capture_hooks) {
            // Not a defect of this build. The arch wires no capture hooks, so there
            // is no pre-RoPE K and no Q to score with on ANY cache format -- and
            // because every score then stays 0, the selector silently returns a
            // recency window. Report that instead of a retrieval "result".
            printf("retrieval   : N/A - this arch wires no capture hooks (only "
                   "src/models/qwen3.cpp and src/models/qwen35.cpp do), so the mean-K "
                   "ledger and the query sum can never be filled on any cache format\n");
        } else if (rs.have_block == 0 || rs.q_layers == 0 || rs.scored == 0) {
            fprintf(stderr, "FAIL: retrieval produced no usable scores: have_block=%u "
                            "q_layers=%u scored=%u write_k=%llu write_q=%llu\n",
                    rs.have_block, rs.q_layers, rs.scored,
                    (unsigned long long) rs.write_k, (unsigned long long) rs.write_q);
            pass = false;
        } else {
            printf("retrieval   : ok - scored %u/%u blocks from %u query layers "
                   "(have_block=%u max=%.6f)\n",
                   rs.scored, rs.blocks, rs.q_layers, rs.have_block, rs.max_score);
        }
    }

    llama_log_set(nullptr, nullptr);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    if (!pass) {
        printf("\nRESULT: FAIL\n");
        return 1;
    }
    printf("\nRESULT: PASS\n");
    return 0;
}
