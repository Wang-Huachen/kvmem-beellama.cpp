// kvarn-move-selftest.cpp
//
// Model-free, backend-level self test for the "whole crate move" primitive that
// a KVMem + KVarN integration needs: copy one KVarN record group's bytes from
// physical slot A to physical slot B and read it back.
//
// What is exercised, all through the ggml C API only (no llama_model, no GGUF,
// no llama_kv_cache_kvarn instance):
//
//   1. llama_kvarn_make_layout(head_dim, group, k_bits, v_bits)  -> tile layout
//   2. ggml_kvarn_store       : seal a 128-token tile from an F16 stage into the
//                               I8 record arena (one record per head)
//   3. ggml_backend_tensor_copy on a level-3 record-group view: the A -> B byte
//                               move, no re-quantization
//   4. ggml_kvarn_materialize : read the tile back to F16 and compare
//
// IMPORTANT API FACT: ggml_kvarn_view cannot be used here. It is a graph proxy
// with no compute kernel on any backend (CPU: empty case in
// ggml-cpu/ggml-cpu.c; CUDA: absent from the op dispatch; declared a
// zero-allocation proxy in ggml-alloc.c) and its result is only ever consumed
// as src[1]/src[2] of GGML_OP_FLASH_ATTN_EXT. ggml_kvarn_materialize is the
// supported op that turns records back into a real F16 value, so that is what
// this test reads back with.
//
// Graph construction order (same discipline as llama.cpp/tests/test-backend-ops.cpp):
// every input tensor AND every op node (with the result tensor the KVarN op
// constructors create themselves) is built first, then a single
// ggml_backend_alloc_ctx_tensors() call gives all of them a buffer, and only
// then is any data written or any graph computed. Writing to a tensor before its
// buffer exists trips GGML_ASSERT(buf != NULL) in ggml-backend.cpp:338.
//
// Record group addressing used by the store op (read from
// ggml/src/ggml-cpu/ops.cpp:ggml_compute_forward_kvarn_store and
// ggml/src/ggml-cuda/kvarn.cu:kvarn_store_kernel_hishmem):
//   encoded index -> cell = uint32_t(payload)
//   group_global  = cell / 128          pos = cell % 128
//   stream        = group_global / groups_per_stream
//   group         = group_global - stream * groups_per_stream
//   record byte   = head * nb[1] + (stream * groups_per_stream + group) * nb[2]
// With n_stream == 1 (KVMem forces unified / n_seq_max == 1) that is simply
// group = cell / 128, and one record group is exactly nb[2] bytes of the arena.
//
// ASCII only. Build: cmake --build <build> --config Release --target kvarn-move-selftest

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "llama-kvarn.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t GROUP         = 128; // KVAR_N_GROUP: tokens per KVarN tile
constexpr int64_t HEAD_DIM      = 128; // tile edge
constexpr int64_t N_HEAD_KV     = 8;
constexpr int64_t N_STREAM      = 1;
constexpr int32_t STAGE_GROUPS  = 3;   // production default: sink + tail
constexpr int32_t TAIL_GROUPS   = STAGE_GROUPS - 1;
constexpr int32_t SINKHORN_ITERS = 16;

// Group 1 = A (source), group 2 = B (destination of the byte move),
// group 3 = C (negative control, sealed from different stage data).
constexpr int64_t GROUP_A  = 1;
constexpr int64_t GROUP_B  = 2;
constexpr int64_t GROUP_C  = 3;
constexpr int64_t N_TOKENS = 3 * GROUP;   // group 0 (sink, never recorded) + A + C
constexpr int64_t N_GROUPS = 8;           // arena depth in record groups

// ---------------------------------------------------------------------------

struct bits_case {
    int bits = 0;
    size_t record_size = 0;

    // inputs
    ggml_tensor * current = nullptr; // F32 [128, n_head_kv, n_tokens]
    ggml_tensor * indices = nullptr; // I64 [n_tokens]
    ggml_tensor * stage   = nullptr; // F16 [128, n_head_kv, 128*stage_groups*n_stream]
    ggml_tensor * records = nullptr; // I8  [record_size, n_head_kv, n_groups]
    ggml_tensor * idx_a   = nullptr; // I64 [128] -> cells of group A
    ggml_tensor * idx_b   = nullptr; // I64 [128] -> cells of group B
    ggml_tensor * idx_c   = nullptr; // I64 [128] -> cells of group C

    // op nodes; the KVarN constructors create their own result tensors, so these
    // must exist before ggml_backend_alloc_ctx_tensors() runs.
    ggml_tensor * store = nullptr;
    ggml_tensor * mat_a = nullptr;   // F16 [128, n_head_kv, 128, n_stream]
    ggml_tensor * mat_b = nullptr;
    ggml_tensor * mat_c = nullptr;

    // byte-move views
    ggml_tensor * copy_src = nullptr;
    ggml_tensor * copy_dst = nullptr;
};

std::vector<uint8_t> render_record_group(const std::vector<uint8_t> & arena, const ggml_tensor * records, int64_t g) {
    const size_t row    = size_t(records->ne[0]);
    const size_t heads  = size_t(records->ne[1]);
    const size_t stride = size_t(records->nb[2]);
    return std::vector<uint8_t>(arena.begin() + size_t(g) * stride,
                                arena.begin() + size_t(g) * stride + heads * row);
}

void report_bytes(const char * label, const std::vector<uint8_t> & bytes) {
    uint32_t h = 2166136261u;
    for (uint8_t b : bytes) {
        h = (h ^ b) * 16777619u;
    }
    std::printf("  %-24s : %6zu bytes, FNV-1a %08x\n", label, bytes.size(), (unsigned) h);
}

// Deterministic F16 payload. Group C uses a much smaller divisor so the
// negative control is guaranteed to differ from A.
ggml_fp16_t selftest_value(int64_t group, int64_t token, int64_t dim) {
    const double x = double((token * 131 + dim * 7 + group * 17) % 2048) - 1024.0;
    return ggml_fp32_to_fp16(float(x / (group == GROUP_C ? 3.0 : 1024.0)));
}

// ---------------------------------------------------------------------------

void fill_current(ggml_tensor * current) {
    std::vector<float> host(size_t(ggml_nelements(current)));
    const int64_t ne0 = current->ne[0];
    const int64_t ne1 = current->ne[1];
    const int64_t nt  = current->ne[2];
    for (int64_t t = 0; t < nt; ++t) {
        const int64_t group = t / GROUP;
        for (int64_t h = 0; h < ne1; ++h) {
            for (int64_t d = 0; d < ne0; ++d) {
                host[size_t((t * ne1 + h) * ne0 + d)] = ggml_fp16_to_fp32(selftest_value(group, t % GROUP, d));
            }
        }
    }
    ggml_backend_tensor_set(current, host.data(), 0, ggml_nbytes(current));
}

void fill_indices(ggml_tensor * indices) {
    std::vector<int64_t> host(size_t(indices->ne[0]));
    for (size_t i = 0; i < host.size(); ++i) {
        // A plain logical cell index. llama_kvarn_encode_store_cell(cell, 0)
        // produces exactly this value, so the kernel derives the F16 stage slot
        // from the group instead of using a host-selected physical slot.
        host[i] = int64_t(i);
    }
    ggml_backend_tensor_set(indices, host.data(), 0, ggml_nbytes(indices));
}

void fill_group_cells(ggml_tensor * idx, int64_t group) {
    std::vector<int64_t> host(size_t(idx->ne[0]));
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = group * GROUP + int64_t(i);
    }
    ggml_backend_tensor_set(idx, host.data(), 0, ggml_nbytes(idx));
}

// ---------------------------------------------------------------------------

ggml_tensor * build_store(ggml_context * ctx, bits_case & c) {
    ggml_tensor * res = ggml_kvarn_store(ctx, c.current, c.indices, c.stage, c.records,
            c.bits, SINKHORN_ITERS, /*value=*/false, STAGE_GROUPS);
    // op_params[3] = tokens_per_stream_hint. Keeping it at 0 disables the CUDA
    // workspace/direct split paths in ggml_cuda_op_kvarn_store, so the store
    // runs on the generic kernel (kvarn_store_kernel_hishmem) - the route an
    // external, cache-less caller gets.
    res->op_params[3] = 0;
    res->op_params[4] = 0;          // SWA off
    res->op_params[8] = TAIL_GROUPS;
    res->op_params[9] = 1;          // eager_records: close each tile when it fills
    return res;
}

ggml_tensor * build_materialize(ggml_context * ctx, bits_case & c, ggml_tensor * idx) {
    ggml_tensor * res = ggml_kvarn_materialize(ctx, c.records, c.stage, idx,
            (int) GROUP, /*stream_start=*/0, (int) N_STREAM, c.bits,
            /*value=*/false, STAGE_GROUPS);
    res->op_params[4]  = 0;         // emit_rotated = false (apply the inverse WHT)
    res->op_params[6]  = 0;         // SWA off
    res->op_params[8]  = TAIL_GROUPS;
    res->op_params[9]  = 1;         // eager_records must match the store
    // With read_indirect off the materialize kernel derives the cell from
    // blockIdx.x (encoded = blockIdx.x, which is always group 0 - the sink), so
    // no read window could ever reach a recorded group. read_indirect on makes
    // the op take the cell from *indices, which is the only way to point the
    // read-back at a specific record group.
    res->op_params[10] = 1;         // read_indirect: cell = indices[blockIdx.x]
    return res;
}

ggml_status run_graph(ggml_context * ctx, ggml_backend_t backend, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    return ggml_backend_graph_compute(backend, gf);
}

// ---------------------------------------------------------------------------

bool run_one_bits(ggml_backend_t backend, ggml_context * ctx, bits_case & c) {
    const llama_kvarn_tile_layout layout = llama_kvarn_make_layout((int) HEAD_DIM, (int) GROUP, c.bits, c.bits);

    std::printf("\n");
    std::printf("  bits               : %d\n", c.bits);
    std::printf("  record_size (K)    : %zu bytes  == llama_kvarn_packed_bytes(128*128,%d) + 3*128*2\n",
            c.record_size, c.bits);
    std::printf("  layout.tile_bytes  : %zu bytes  (K+V full crate: the production 1:1 arena row)\n", layout.tile_bytes);
    std::printf("  arena              : [%zu, %lld, %lld] I8\n",
            c.record_size, (long long) N_HEAD_KV, (long long) N_GROUPS);
    std::printf("  stage              : [%lld, %lld, %lld] F16 (stage_groups=%d, tail_groups=%d)\n",
            (long long) GROUP, (long long) N_HEAD_KV, (long long) (GROUP * STAGE_GROUPS * N_STREAM),
            (int) STAGE_GROUPS, (int) TAIL_GROUPS);

    // ---- 1. seal group A and the negative-control group C ------------------
    {
        const ggml_status st = run_graph(ctx, backend, c.store);
        if (st != GGML_STATUS_SUCCESS) {
            std::printf("  STORE              : FAILED (status %d)\n", (int) st);
            return false;
        }
    }

    // ---- 2. the "whole crate move": A -> B, raw bytes, no re-quantization ---
    ggml_backend_tensor_copy(c.copy_src, c.copy_dst);

    // ---- 3. read the arena back and compare record groups bit for bit -------
    std::vector<uint8_t> arena(size_t(ggml_nbytes(c.records)));
    ggml_backend_tensor_get(c.records, arena.data(), 0, arena.size());

    const std::vector<uint8_t> rec_a = render_record_group(arena, c.records, GROUP_A);
    const std::vector<uint8_t> rec_b = render_record_group(arena, c.records, GROUP_B);
    const std::vector<uint8_t> rec_c = render_record_group(arena, c.records, GROUP_C);

    const bool bytes_ab_equal = (rec_a == rec_b);
    const bool bytes_ac_equal = (rec_a == rec_c);

    report_bytes("record(A) bytes", rec_a);
    report_bytes("record(B) bytes", rec_b);
    report_bytes("record(C) bytes", rec_c);
    std::printf("  A->B byte copy     : %zu bytes\n", size_t(c.records->nb[2]));
    std::printf("  record(A)==record(B): %s\n", bytes_ab_equal ? "YES" : "NO");
    std::printf("  record(A)==record(C): %s (negative control, expect NO)\n", bytes_ac_equal ? "YES" : "NO");

    // ---- 4. read A, B and C back, each through its own materialize node ----
    std::vector<uint8_t> mat_a(size_t(ggml_nbytes(c.mat_a)));
    std::vector<uint8_t> mat_b(mat_a.size());
    std::vector<uint8_t> mat_c(mat_a.size());

    if (run_graph(ctx, backend, c.mat_a) != GGML_STATUS_SUCCESS) {
        std::printf("  MATERIALIZE(A)     : FAILED\n");
        return false;
    }
    ggml_backend_tensor_get(c.mat_a, mat_a.data(), 0, mat_a.size());

    if (run_graph(ctx, backend, c.mat_b) != GGML_STATUS_SUCCESS) {
        std::printf("  MATERIALIZE(B)     : FAILED\n");
        return false;
    }
    ggml_backend_tensor_get(c.mat_b, mat_b.data(), 0, mat_b.size());

    if (run_graph(ctx, backend, c.mat_c) != GGML_STATUS_SUCCESS) {
        std::printf("  MATERIALIZE(C)     : FAILED\n");
        return false;
    }
    ggml_backend_tensor_get(c.mat_c, mat_c.data(), 0, mat_c.size());

    const bool view_ab_equal = (mat_a == mat_b);
    const bool view_ac_equal = (mat_a == mat_c);

    report_bytes("materialize(A) bytes", mat_a);
    report_bytes("materialize(B) bytes", mat_b);
    report_bytes("materialize(C) bytes", mat_c);
    std::printf("  view(A) == view(B) : %s\n", view_ab_equal ? "YES" : "NO");
    std::printf("  view(A) == view(C) : %s (negative control, expect NO)\n", view_ac_equal ? "YES" : "NO");

    if (!view_ab_equal) {
        size_t first = mat_a.size();
        for (size_t i = 0; i < mat_a.size(); ++i) {
            if (mat_a[i] != mat_b[i]) { first = i; break; }
        }
        std::printf("  first differing byte (materialize A vs B): %zu\n", first);
    }

    const bool ok = bytes_ab_equal && view_ab_equal && !bytes_ac_equal && !view_ac_equal;
    if (!ok) {
        std::printf("  NOTE: a failed negative control means this run cannot tell the groups apart.\n");
    }
    std::printf("  RESULT             : %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main() {
    std::printf("kvarn-move-selftest\n");

    ggml_backend_load_all();

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    const char * kind = "GPU";
    if (dev == nullptr) {
        dev  = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        kind = "CPU (no GPU device available)";
    }
    if (dev == nullptr) {
        std::printf("  RESULT             : FAIL (no ggml backend device available)\n");
        return 1;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        std::printf("  RESULT             : FAIL (backend init failed for %s)\n", ggml_backend_dev_name(dev));
        return 1;
    }

    std::printf("  backend            : %s (%s) [%s]\n",
            ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), kind);

    const int bits_list[] = { 4, 5, 6, 8 };
    const size_t n_cases = sizeof(bits_list) / sizeof(bits_list[0]);
    std::vector<bits_case> cases(n_cases);

    // no_alloc must be true: ggml_backend_alloc_ctx_tensors() takes ownership of
    // assigning every tensor's data pointer to one backend buffer, and it asserts
    // ggml_get_no_alloc(ctx) in ggml-alloc.c.
    ggml_init_params params;
    params.mem_size   = 256u * 1024u * 1024u;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::printf("  RESULT             : FAIL (ggml_init failed)\n");
        ggml_backend_free(backend);
        return 1;
    }

    // Phase 1: build every input tensor and every op node (each KVarN op creates
    // its own result tensor) for every bit width. Nothing is written, nothing is
    // computed, and no tensor has a buffer yet.
    for (size_t i = 0; i < n_cases; ++i) {
        bits_case & c = cases[i];
        c.bits = bits_list[i];
        c.record_size = llama_kvarn_packed_bytes((int) (GROUP * GROUP), c.bits) +
                3 * size_t(GROUP) * sizeof(uint16_t);

        c.current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, GROUP, N_HEAD_KV, N_TOKENS);
        c.indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, N_TOKENS);
        c.stage   = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, GROUP, N_HEAD_KV, GROUP * STAGE_GROUPS * N_STREAM);
        c.records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, (int64_t) c.record_size, N_HEAD_KV, N_GROUPS);
        c.idx_a   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, GROUP);
        c.idx_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, GROUP);
        c.idx_c   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, GROUP);

        c.store = build_store(ctx, c);
        c.mat_a = build_materialize(ctx, c, c.idx_a);
        c.mat_b = build_materialize(ctx, c, c.idx_b);
        c.mat_c = build_materialize(ctx, c, c.idx_c);

        c.copy_src = ggml_view_3d(ctx, c.records, (int64_t) c.record_size, N_HEAD_KV, 1,
                c.records->nb[1], c.records->nb[2], size_t(GROUP_A) * c.records->nb[2]);
        c.copy_dst = ggml_view_3d(ctx, c.records, (int64_t) c.record_size, N_HEAD_KV, 1,
                c.records->nb[1], c.records->nb[2], size_t(GROUP_B) * c.records->nb[2]);
    }

    // Phase 2: one device buffer for every tensor built above.
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        std::printf("  RESULT             : FAIL (backend buffer allocation failed)\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // Phase 3: only now that every tensor owns a buffer, upload the inputs.
    for (size_t i = 0; i < n_cases; ++i) {
        bits_case & c = cases[i];
        fill_current(c.current);
        fill_indices(c.indices);
        fill_group_cells(c.idx_a, GROUP_A);
        fill_group_cells(c.idx_b, GROUP_B);
        fill_group_cells(c.idx_c, GROUP_C);
    }

    // Phase 4: run. Each materialize node reads its own pre-filled index tensor,
    // so the three read-backs cannot interfere with each other.
    bool all_pass = true;
    for (size_t i = 0; i < n_cases; ++i) {
        if (!run_one_bits(backend, ctx, cases[i])) {
            all_pass = false;
        }
    }

    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    std::printf("\n  OVERALL RESULT     : %s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
