#pragma once

#include "llama-kv-cache.h"
#include "llama-kv-cache-kvarn.h"
#include "llama-memory.h"
#include "llama-kvmem-hooks.h"

#include "kvmem/kvmem_runtime.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/rope.hpp"

#include <condition_variable>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

struct llama_model;
struct llama_cparams;
struct llama_memory_params;
struct ggml_tensor;
class llama_memory_recurrent;
class llama_memory_kvmem_mtp;

// Bounded block-slot pool over a llama_kv_cache_kvarn.
//
// GPU attention cache size = min(n_ctx, budget + gen_reserve). Each logical
// KVMem block occupies one slot of `block_tokens` cells. Reselect never packs
// cells into [0, W); resident blocks keep their slot.
//
// Two pointers, one owner:
//   kvarn_   the record arena + native KVarN attention (owns the records)
//   kv_      == kvarn_->get_metadata_cache(): cells/pos/logical bookkeeping only
// `kv_` keeps the plain llama_kv_cache type because every pure-metadata call
// site (get_cells / seq_rm_logical / get_size / memory_breakdown / ...) keeps
// working unchanged through it. Everything that touches records, state or the
// eviction contract must go through `kvarn_`.
//
// DESIGN RED LINE: block_tokens_ is pinned to KVAR_N_GROUP, so one KVMem block
// is exactly one KVarN record group. Anything else would require deleting part
// of a sealed record, which the format has no primitive for.
//
// init_batch returns a llama_kv_cache_kvarn_context wrapping the plain
// llama_kv_cache_context, which is what lights up KVarN direct attention
// (llama-graph.cpp dynamic_casts on the kvarN context class).
class llama_memory_kvmem : public llama_memory_i {
public:
    // KVMem builds (and owns) its own attention store from params/cparams: a KVarN
    // record arena when KVarN is enabled, a plain row cache otherwise. There is
    // deliberately no way to hand it an already-built cache. The old hybrid mode
    // borrowed the llama_memory_hybrid base class's stock llama_kv_cache, which is
    // exactly why that path could never reach the record arena; the hybrid wrapper
    // now composes instead and moves THIS object into the base class, so the base's
    // init_kv_batch() lands on the one and only attention object. See
    // llama-memory-kvmem-hybrid.h.
    llama_memory_kvmem(
            const llama_model & model,
            const llama_memory_params & params,
            const llama_cparams & cparams);

    ~llama_memory_kvmem() override;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    // The record arena owns the eviction contract, so these four must not fall
    // through to llama_memory_i's permissive defaults:
    //   get_seq_rm_capability() default would advertise arbitrary-range removal
    //     and unbounded suffix rollback that KVarN rejects.
    //   init_kv_batch() default returns nullptr, and the inherited meaning ("ask
    //     the inner cache to place these ubatches") would bypass KVMem's slot pool
    //     entirely -- including assign_virtual_slots(), without which KVarN's
    //     live_group inference breaks and a partial group reads zeros. The hybrid
    //     base class calls this method, so it runs the same preparation init_batch
    //     does and only differs in that it does not split the ubatches itself.
    //   get_kv_n_stream()/get_kv_size() defaults return 0, which would flip
    //     `unified` in the hybrid wrappers and desynchronise kv_size reports.
    llama_memory_context_ptr init_kv_batch(const std::vector<llama_ubatch> & ubatches) override;
    uint32_t get_kv_n_stream() const override;
    uint32_t get_kv_size() const override;
    seq_rm_capability get_seq_rm_capability() const override;

    // The precision-tail queries MUST be forwarded to whatever cache actually
    // owns the tail. Before batch 3 KVMem wrapped a real llama_kv_cache and the
    // base default was never consulted; now KVMem is the outer object, so
    // inheriting plain llama_memory_i defaults would (a) report GGML_TYPE_COUNT
    // for the tail type and make llama_context throw
    //   "KV tail cache did not report its resolved storage type"
    // whenever a tail is requested, and (b) claim zero tail coverage.
    ggml_type get_kv_tail_type() const override;
    uint32_t get_kv_tail_group_count() const override;
    bool get_kv_tail_coverage(uint32_t group_index, llama_seq_id seq_id,
                              llama_kv_tail_coverage_info & out) const override;
    void reset_kv_tail_planner_timing() override;
    uint64_t get_kv_tail_planner_timing_ns() const override;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    // cell-level removal is delegated to the inner cache; seq_rm_logical and
    // seq_rm_cell both honour llama_kv_cell_ext::logical_pos
    bool seq_rm_cell(llama_seq_id seq_id, uint32_t cell_idx) override;
    // must NOT delegate: the inner cache matches the raw pos, KVMem matches
    // the logical pos (same convention as seq_rm_logical)
    int cells_at_pos(llama_seq_id seq_id, llama_pos pos, uint32_t * cell_indices, int n_max) override;

    // TODO(rebase): llama_memory_i::get_seq_rm_capability() is overridden above
    // and now delegates to the KVarN arena, which clamps the suffix rollback to
    // KVAR_N_GROUP. What is still not modelled is KVMem's own extra restriction:
    // the hybrid seq_rm deliberately does not roll GDN back for query-replay
    // holes. Tighten this further once the hybrid path is ported.

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    // Forwarded because llama_memory_hybrid::kv_memory_stats() asks its attention
    // half; without this the composed hybrid would report the llama_memory_i
    // default (all zeros) where the stock row cache used to report real numbers.
    llama_kv_memory_stats kv_memory_stats() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    // The metadata half of the inner cache. Callers that used to treat get_kv()
    // as "the cache that stores K/V" must now understand it as cells only: it
    // has zero layers under KVarN (the arena supplies filtered layer ids), so
    // every row-shaped access through it silently no-ops.
    llama_kv_cache * get_kv() { return kv_; }
    // The record arena + native attention owner. Not const: every consumer may
    // have to move records or query live tensors.
    llama_kv_cache_kvarn * get_kvarn() { return kvarn_; }
    kvmem::KvMemRuntime & runtime() { return *runtime_; }
    const kvmem::KvMemRuntime & runtime() const { return *runtime_; }

    uint32_t kv_size() const { return kv_size_; }
    uint32_t block_tokens() const { return block_tokens_; }
    uint32_t n_slots() const { return n_slots_; }

    // Slot-pool prepare used by both the dense KVMem memory and the hybrid
    // wrapper (attn half). Fills per-ubatch slot_info and the capture pos queue.
    bool prepare_ubatches(
            const std::vector<llama_ubatch> & ubatches,
            uint32_t n_new_tokens,
            llama_kv_cache::slot_info_vec_t & sinfos);
    void reset_policy();

    // "Does layer il have a KV cache?" -- as opposed to "does the row cache
    // expose rows for layer il?".
    //
    // The two questions coincide only while KVMem wraps a plain llama_kv_cache.
    // Under KVarN `kv_` is the arena's METADATA cache, which is built with a layer
    // filter that rejects every id (llama-kv-cache-kvarn.cpp:1353), so
    // kvmem_cache_has_layer(kv_, il) is false for every layer. Call sites that use
    // the check purely as "is this an attention layer?" -- the decode-time running
    // mean and the query-state serialisation, neither of which reads a row -- must
    // therefore ask the arena when one is present. Row readers must keep using
    // kvmem_cache_has_layer() so they stay fail-closed.
    bool kvmem_layer_has_kv(int32_t il) const;

    // Retrieval health snapshot; see llama_kvmem_retrieval_stats in
    // llama-kvmem-hooks.h for why this exists.
    llama_kvmem_retrieval_stats retrieval_stats() const;

    int32_t alloc_slot();
    // Virtual slot numbers. KVarN infers "which group is still being written" from the
    // NUMERIC MAXIMUM of the indices it is handed (kvarn.cu:2149 keeps the largest
    // group as `live_group`), and the read side uses that to decide whether a group
    // comes from its sealed record or from the F16 stage
    // (fattn-mma-kvarn.cuh:77-90). KVMem's physical slots do NOT track write order --
    // eviction can hand the newest block a low slot -- which makes KVarN read an
    // incomplete group from an unsealed record and silently attend to zeros.
    //
    // So hand KVarN a virtual number instead: monotonic in admission order, recycled
    // within the same pool size. The newest block always carries the highest virtual
    // number, which is exactly the invariant KVarN assumes. Physical placement stays
    // KVMem's business, and KVarN needs no change at all.
    uint32_t virtual_slot_of(uint32_t block_id) const;
    void assign_virtual_slots();
    void free_slot(int32_t slot);
    int32_t peek_free_slot() const;
    // Map original pos to (gpu_slot, offset in block). For a token the target
    // has not appended yet (MTP draft), predict the slot the next alloc would
    // take without popping the free list. Returns false if no mapping exists.
    bool slot_for_orig_pos(llama_pos pos, int32_t * slot, uint32_t * off) const;
    // Same lookup, but returning the number the cache cells are ACTUALLY addressed
    // by: the virtual slot (admission order) when a KVarN arena is present, the
    // physical slot otherwise (virtual_slot_of() falls back to gpu_slot, so row
    // mode is unchanged). Anything that fills or reads cells must use this; only
    // slot allocation and record placement may use the physical number.
    bool slot_for_orig_pos_virtual(llama_pos pos, int32_t * slot, uint32_t * off) const;
    // First cell of a block's rows, in that same numbering.
    uint32_t block_cell_base(uint32_t block_id) const;
    const kvmem::KvMemStore & store() const { return runtime_->store(); }

    void note_ubatch_pos(const std::vector<llama_pos> & pos);
    void reset_query_acc();
    void register_capture(struct ggml_tensor * t, int il, char which);
    void capture_on_new_graph();
    bool capture_can_reuse(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) const;
    void harvest_pending(struct ggml_backend_sched * sched);
    void harvest_flush();
    void harvest_perf_print_sum();
    void harvest_capture(struct ggml_tensor * t, int il, char which);
    void apply_retrieval();
    void set_turn_spans(const llama_kvmem_turn_spans & spans);
    bool query_contains(llama_pos row) const;
    bool query_overlaps(uint32_t n, const llama_pos * rows) const;
    llama_kvmem_attention_view attention_view(bool canonical = true) const;
    bool can_append(uint32_t end, uint32_t generation_rows, bool all_history, std::string & reason) const;
    llama_kvmem_selection preview_retrieval();
    bool selection_fits(const llama_kvmem_selection & selection, uint32_t end, uint32_t generation_rows) const;
    bool commit_unchanged(const llama_kvmem_attention_view & view, const llama_kvmem_selection & selection);
    void apply_selection(const llama_kvmem_selection & selection);
    bool commit_resident(bool canonical = true);
    bool get_query(llama_kvmem_query_state & state);
    bool set_query(const llama_kvmem_query_state & state);
    void freeze_query(bool frozen) { query_frozen_ = frozen; }
    // The follower also invalidates a pending proof when old draft KV changes.
    void note_attention_change() { ++attention_epoch_; }
    bool query_replay_fits(uint32_t query_begin, uint32_t prompt_end) const;
    void dump_kv_compare(int32_t block_id, bool writeback_test = false);
    void trace_working_set(const char * tag) const;
    void set_replay(bool replay);
    bool replay() const { return replay_; }
    bool want_prefill_capture() const {
        return prefill_capture_ && !retrieval_pinned_ && !replay_;
    }
    // Recapture Q for retrieval even while replaying a cached query span.
    bool want_q_capture() const {
        return method_ == 1 && !retrieval_pinned_ && !query_frozen_ &&
            (explicit_spans_ ? !turn_spans_.query.empty() : query_begin_ >= 0);
    }
    bool want_decode_mean() const {
        return retrieval_pinned_ && method_ == 1 && !replay_;
    }
    void end_prefill_capture() { prefill_capture_ = false; }
    // Next request continues this sequence: flush decode mean, unpin
    // retrieval, harvest mean-K for the suffix. Does not wipe prefix KV.
    void begin_cached_turn(bool reset_query = true);
    // Same-query skip: keep the retrieved GPU window. Prefill still harvests
    // the new tail; recency pressure must not evict selected history.
    void keep_selected_window() { keep_selected_ = true; }
    // After skip/reselect prefill: pin so decode mean-K uses gen_reserve.
    void pin_working_set() {
        retrieval_pinned_ = true;
        keep_selected_ = true;
    }
    size_t free_slot_count() const { return free_slots_.size(); }
    void truncate_cached(uint32_t n_past);
    void occupy_in(llama_kv_cache * cache, uint32_t block_id);
    llama_pos model_pos(uint32_t logical_pos) const;
    bool remove_logical(llama_context * ctx, llama_pos begin, llama_pos end);
    uint32_t store_n_tokens() const {
        return runtime_ ? runtime_->store().total_tokens() : 0;
    }
    llama_pos recr_pos_max() const;
    void decode_mean_commit(uint32_t n_keep);
    void decode_mean_discard();
    void decode_mean_flush();
    void set_recurrent(llama_memory_recurrent * recr);
    bool has_recurrent() const { return recr_ != nullptr; }
    bool gdn_replay_enabled() const;
    bool gdn_replay_begin(llama_pos start, uint32_t width);
    bool gdn_replay_commit(llama_context * ctx, uint32_t n_keep);
    void set_query_span(int32_t begin, int32_t end) {
        harvest_flush();
        explicit_spans_ = false;
        query_begin_ = begin;
        query_end_ = end;
    }
    void set_force_pos(int32_t pos) { force_pos_ = pos; }
    void set_mtp_follower(llama_memory_kvmem_mtp * mtp) { mtp_ = mtp; }
    llama_memory_kvmem_mtp * mtp_follower() { return mtp_; }

    const kvmem::RopeConfig & rope() const { return rope_; }
    ggml_type type_k() const { return type_k_; }
    ggml_type type_v() const { return type_v_; }
    bool v_trans() const { return v_trans_; }
    uint32_t n_embd_k() const { return n_embd_k_; }
    uint32_t n_embd_v() const { return n_embd_v_; }

    kvmem::RawKvStore & raw() { return *raw_; }

private:
    friend struct kvmem_transfer_test_access;
    struct SlotBackend : public kvmem::KvMemBackend {
        llama_memory_kvmem * owner = nullptr;
        int32_t alloc_gpu_slot() override { return owner->alloc_slot(); }
        void free_gpu_slot(int32_t slot) override { owner->free_slot(slot); }
        void copy_block_to_host(uint32_t block_id, int32_t gpu_slot,
                                void * host, uint64_t bytes) override {
            owner->copy_gpu_block_to_host(block_id, gpu_slot, host, bytes);
        }
        void copy_block_from_host(uint32_t block_id, int32_t gpu_slot,
                                  const void * host, uint64_t bytes) override {
            owner->copy_gpu_block_from_host(block_id, gpu_slot, host, bytes);
        }
    };

    uint32_t resident_tokens() const;
    bool prepare_working_set(uint32_t n_new_tokens);
    void apply_plan_to_kv(const kvmem::KvMemPlan & plan);
    // Place GPU-resident blocks into slots 0..N-1 in orig_pos order.
    // Resident KV is copied slot-to-slot; cold blocks memcpy packed GPU K/V.
    bool layout_gpu_slots_by_orig_pos();
    // KVarN reads a partially filled group only from the F16 stage, and only while
    // that group is the live one (the highest occupied slot). Eviction can leave the
    // partial block on a lower slot, so repack by orig_pos when that happens.
    bool ensure_partial_block_is_live();
    bool gpu_kv_already_resident(uint32_t block_id) const;
    bool gpu_kv_complete(uint32_t block_id, const llama_kv_cache * cache) const;
    std::vector<uint32_t> retrieval_mandatory() const;
    void occupy_block_cells(uint32_t block_id);
    void reset_slots();
    void trace_plan(const char * tag, const kvmem::KvMemPlan & plan) const;
    void write_block_to_gpu(uint32_t block_id);
    // --- KVarN record round-trip (see stage3-batch4-design.md) ---------------
    // Under KVarN a block's GPU residency IS its record group, and a record's
    // bytes depend only on the token values, the head slice and the bit width --
    // never on which slot holds it. So a block leaves the GPU by copying its
    // record bytes to host and comes back by copying them in again. That is what
    // makes the store operator unnecessary here.
    uint64_t record_bytes_per_block() const;
    // True when this block's records can be safely copied out of the arena. Two
    // blocks cannot:
    //   * the block sitting in slot 0. Slot 0 is KVarN's permanent F16 "sink":
    //     the store kernel only seals a group at pos==127 and explicitly skips
    //     group 0 (ggml/src/ggml-cuda/kvarn.cu:1031-1037, :1114-1123), and the
    //     read side always takes group 0 from the F16 stage. Its nb[2] bytes are
    //     another group's leftovers, so snapshotting them is meaningless.
    //   * a partially filled block (n_tokens < block_tokens). Its group is never
    //     sealed, so it too lives only in the F16 stage. See snapshot_records().
    bool block_records_snapshotable(uint32_t block_id) const;
    void snapshot_records(uint32_t block_id);
    void snapshot_records_above(uint32_t token_pos);   // KVMEM-FIX(2026-09-26)
    bool restore_records(uint32_t block_id);
    void harvest_gpu_v(uint32_t block_id);
    void harvest_gpu_v_commit();
    void harvest_gpu_v_flush_slab();
    void harvest_write_batch();
    // After a prefill graph, enqueue packed K/V D2H for GPU-resident
    // full blocks. Does not wait; apply_plan / retrieval commit.
    void harvest_full_blocks_async();
    void decode_mean_ingest(struct ggml_backend_sched * sched);
    void decode_mean_reset();
    void decode_mean_add_range(uint32_t tok0, uint32_t n_add);
    void decode_mean_zero_acc();
    bool decode_mean_add_host_layer(struct ggml_tensor * t, int il, uint32_t tok0, uint32_t n_add);
    void decode_mean_print_sum();
    struct HarvestVJob {
        uint32_t pos0 = 0;
        uint32_t n = 0;
        uint32_t il = 0;
        bool is_k = false;
        const uint8_t * gpu_src = nullptr;
        size_t nbytes = 0;
        size_t pin_off = 0;
        ggml_tensor * vt = nullptr;
        size_t tensor_off = 0;
    };
    struct HarvestVBatch {
        int slot = -1;
        std::vector<HarvestVJob> jobs;
    };
    std::vector<HarvestVJob> harvest_v_jobs_;
    HarvestVBatch harvest_v_pending_;
    // 1 while a block's packed D2H is queued or in flight (until commit).
    std::vector<uint8_t> harvest_gpu_queued_;
    void copy_gpu_block_to_host(uint32_t block_id, int32_t gpu_slot,
                                void * host, uint64_t bytes);
    void copy_gpu_block_from_host(uint32_t block_id, int32_t gpu_slot,
                                  const void * host, uint64_t bytes);
    void score_retrieval();
    bool read_gpu_block(uint32_t block_id, uint32_t il, bool is_k, std::vector<float> & out) const;
    static void kv_stats(const char * tag, const float * a, const float * b, size_t n);
    static void tensor_to_f32_token_major(const struct ggml_tensor * t, std::vector<float> & out);
    static void bytes_to_f32_token_major(const uint8_t * data, ggml_type type,
                                         int64_t d, int64_t h, int64_t n,
                                         size_t nb0, size_t nb1, size_t nb2,
                                         std::vector<float> & out);
    static void bytes_to_f16_token_major(const uint8_t * data, ggml_type type,
                                         int64_t d, int64_t h, int64_t n,
                                         size_t nb0, size_t nb1, size_t nb2,
                                         std::vector<uint16_t> & out);
    void harvest_from_host(int il, char which, const uint8_t * host,
                           ggml_type type, int64_t d, int64_t h, int64_t n,
                           size_t nb0, size_t nb1, size_t nb2);
    bool d2h_init();
    void d2h_free();
    void d2h_commit(int slot);
    bool d2h_submit(struct ggml_backend * be);
    bool harvest_perf_on() const { return perf_.enabled; }
    void harvest_perf_emit_graph_line();
    void retr_perf_print();
    void harvest_worker_start();
    void harvest_worker_stop();
    void harvest_loop();
    void harvest_wait_slot(int slot);
    bool harvest_worker_on() const;

    struct HarvestPerf {
        bool enabled = false;
        bool sum_printed = false;
        bool graph_line_printed = false;
        uint32_t n_ubatch = 0;
        uint32_t n_tok = 0;
        int64_t harvest_entry_us = 0;
        int64_t sync_us = 0;
        int64_t d2d_us = 0;
        int64_t snap_wait_us = 0;
        int64_t d2h_submit_us = 0;
        int64_t commit_us = 0;
        int64_t d2h_wait_us = 0;
        int64_t pack_us = 0;
        int64_t nvme_us = 0;
        uint64_t nvme_bytes = 0;
        uint64_t nvme_syscalls = 0;
        int64_t last_sync_us = 0;
        int64_t last_d2d_us = 0;
        int64_t last_snap_wait_us = 0;
        int64_t last_d2h_submit_us = 0;
        int64_t last_commit_us = 0;
        int64_t last_d2h_wait_us = 0;
        int64_t last_pack_us = 0;
        int64_t last_nvme_us = 0;
        uint64_t last_nvme_bytes = 0;
        uint64_t last_nvme_syscalls = 0;
        uint32_t n_pressure = 0;
        uint32_t n_pressure_out = 0;
    };

    struct RetrPerf {
        bool enabled = false;
        int64_t total_us = 0;
        int64_t flush_us = 0;
        int64_t score_us = 0;
        int64_t plan_us = 0;
        int64_t stage_out_us = 0;
        int64_t stage_out_gpu_us = 0;
        int64_t stage_out_host_us = 0;
        int64_t admit_us = 0;
        int64_t seq_rm_us = 0;
        int64_t occupy_us = 0;
        int64_t layout_d2h_us = 0;
        int64_t layout_h2d_us = 0;
        int64_t copy_us = 0;
        int64_t rope_us = 0;
        int64_t hadamard_us = 0;
        int64_t set_us = 0;
        int64_t mtp_us = 0;
        int64_t dump_us = 0;
        uint32_t n_move = 0;
        uint32_t n_raw = 0;
        uint32_t n_skip = 0;
        uint32_t n_stage_in = 0;
        int laid_out = 0;
    };

    const llama_model & model_;
    uint32_t block_tokens_ = 128;
    uint32_t kv_size_ = 0;
    uint32_t n_slots_ = 0;
    // Virtual slot per block, and the next number to hand out. Only used when a KVarN
    // arena is present; row mode keeps using blk.gpu_slot directly, so nothing here
    // affects the configuration that works today.
    std::vector<int32_t> vslot_;              // block_id -> virtual slot, -1 if not resident
    uint32_t vslot_next_ = 0;
    // Monotonic admission counter, used only to order blocks by "who entered last".
    std::vector<uint64_t> vorder_;            // block_id -> admission sequence
    uint64_t vorder_next_ = 0;
    bool trace_ = false;
    // Optional reserved scratch slot for layout_gpu_slots_by_orig_pos(): reordering
    // slots is a set of cycles, so a cycle needs somewhere to park a record. -1 means
    // none is reserved, and the layout borrows a slot that no block targets instead.
    int32_t spare_slot_ = -1;
    // Exactly one of these two is set, depending on whether KVarN is enabled:
    //   kvarn_owned_ : KVarN on  -> the record arena is the storage
    //   kv_owned_    : KVarN off -> a plain row cache, as before batch 3
    // Owns the record arena. The metadata cache lives inside it.
    std::unique_ptr<llama_kv_cache_kvarn> kvarn_owned_;
    // Owns the plain row cache (KVarN disabled).
    std::unique_ptr<llama_kv_cache> kv_owned_;
    // Non-owning view of kvarn_owned_. This is the switch every KVarN-only branch
    // keys on: null means "row cache", which is what makes a disabled KVarN fall
    // back to the pre-batch-3 behaviour everywhere.
    llama_kv_cache_kvarn * kvarn_ = nullptr;
    // Plain cells/pos bookkeeping; under KVarN it is the metadata half of the arena
    // (kvarn_->get_metadata_cache()) and owns no rows at all. In row mode it is the
    // whole store and lives in kv_owned_.
    llama_kv_cache * kv_ = nullptr;
    // Last llama_context handed to init_update(). move_record_group() does not
    // synchronise, so the caller must llama_synchronize() on this before moving
    // records. Captured on every init_update, which process_ubatch() calls
    // before the init_batch loop.
    llama_context * sync_ctx_ = nullptr;
    llama_memory_recurrent * recr_ = nullptr;
    struct GdnReplay;
    std::unique_ptr<GdnReplay> gdn_replay_;
    llama_memory_kvmem_mtp * mtp_ = nullptr;
    SlotBackend backend_;
    std::unique_ptr<kvmem::KvMemRuntime> runtime_;
    std::unique_ptr<kvmem::RawKvStore> raw_;
    std::vector<int32_t> free_slots_;
    // block_id -> its record bytes for every layer, packed in slot order. Absent
    // (or the wrong length) means "no authoritative host copy"; see
    // restore_records().
    std::unordered_map<uint32_t, std::vector<uint8_t>> rec_host_;
    // Blocks whose records live only in the F16 stage and therefore must keep
    // their GPU slot: the sink-slot block and the partial tail block. Clearing
    // one of their cells without keeping the slot would make attention read
    // zeros or another group's bytes.
    std::vector<uint8_t> rec_pinned_;
    struct RowPosition {
        std::array<llama_pos, 4> pos{};
        llama_token token = LLAMA_TOKEN_NULL;
        bool spatial = false;
    };
    std::vector<RowPosition> row_positions_;

    kvmem::RopeConfig rope_{};
    uint32_t n_layer_ = 0;
    uint32_t n_embd_k_ = 0;
    uint32_t n_embd_v_ = 0;
    uint32_t n_head_ = 0;
    uint32_t n_head_kv_ = 0;
    uint32_t n_embd_head_ = 0;
    ggml_type type_k_ = GGML_TYPE_F16;
    ggml_type type_v_ = GGML_TYPE_F16;
    bool v_trans_ = false;
    bool replay_ = false;
    bool retrieval_pinned_ = false;
    bool keep_selected_ = false;
    bool prefill_capture_ = true;
    int32_t method_ = 0;
    int32_t query_begin_ = -1;
    int32_t query_end_ = -1;
    int32_t force_pos_ = -1;
    llama_kvmem_turn_spans turn_spans_;
    bool explicit_spans_ = false;
    bool query_frozen_ = false;
    uint64_t attention_epoch_ = 0;

    std::vector<std::vector<llama_pos>> pos_queue_;
    std::vector<llama_pos> cur_pos_;
    struct CaptureNode {
        ggml_tensor * t = nullptr;
        int il = 0;
        char which = 0;
    };
    std::vector<CaptureNode> pending_capture_;
    std::vector<CaptureNode> decode_mean_pending_k_;
    std::vector<llama_pos> decode_mean_pending_pos_;
    uint32_t decode_mean_n_ = 0;
    uint32_t decode_mean_block_ = ~0u;
    uint32_t decode_mean_pos0_ = 0;
    std::vector<std::vector<float>> decode_mean_host_;
    std::vector<uint8_t> decode_mean_src_; // 0 none, 1 gpu, 2 host
    struct DecodeMeanStats {
        uint32_t n_tok = 0;
        uint32_t n_flush = 0;
        uint32_t n_gpu = 0;
        uint32_t n_host = 0;
        uint32_t n_miss = 0;
        uint32_t last_block = ~0u;
        uint32_t last_n = 0;
        uint32_t last_layers = 0;
        float last_rms = 0.0f;
        bool printed = false;
    };
    DecodeMeanStats decode_mean_stats_;
    bool graph_has_q_ = false;
    bool graph_has_k_ = false;
    bool graph_has_record_ = false;
    struct CaptureD2hPipe;
    std::unique_ptr<CaptureD2hPipe> d2h_;
    struct HarvestWorker {
        std::mutex mu;
        std::condition_variable cv;
        std::vector<int> q;
        std::thread th;
        bool stop = false;
    };
    std::unique_ptr<HarvestWorker> harvest_w_;
    HarvestPerf perf_;
    RetrPerf retr_;
    std::vector<std::vector<float>> q_sum_;
    std::vector<uint32_t> q_count_;

    // ---- retrieval health (see score_retrieval) --------------------------------
    // The mean-K ledger and the query sum are written from the graph capture hooks
    // and from nothing else. Counting both ends of that pipe is the only way to
    // tell "this arch never captures" from "capture ran but produced nothing",
    // which otherwise look identical from the outside: an empty ledger, every
    // block scoring 0, and a selector that quietly falls back to recency.
    uint64_t retr_reg_k_  = 0;  // kvmem_capture_k registrations
    uint64_t retr_reg_q_  = 0;  // kvmem_capture_q registrations
    uint64_t retr_write_k_ = 0; // successful mean-K ledger writes
    uint64_t retr_write_q_ = 0; // successful query-sum accumulations
    uint32_t retr_runs_ = 0;    // score_retrieval() invocations
    uint32_t retr_blocks_ = 0;
    uint32_t retr_resident_ = 0;
    uint32_t retr_have_ = 0;
    uint32_t retr_q_layers_ = 0;
    uint32_t retr_scored_ = 0;
    float    retr_max_score_ = 0.0f;
    // One degraded-mode report per context, not per turn.
    bool     retr_warned_ = false;
};

// GPU attn-cache cell count for a KVMem slot pool (budget + gen_reserve,
// clamped to n_ctx on identity). Hybrid uses this as llama_memory_hybrid's
// attn kv_size so the two halves agree.
uint32_t llama_kvmem_pool_cells(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams);
