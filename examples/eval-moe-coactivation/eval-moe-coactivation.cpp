// llama-eval-moe-coactivation
//
// Runs an MoE LLM over a slice of the MMLU dataset and records, per subject,
// pairwise expert co-activation statistics both within a single MoE layer
// (intra-layer) and across pairs of MoE layers (inter-layer, position-aligned
// on the same token). Output is a single JSON file suitable for post-hoc
// analysis in Python (e.g. conditional probability, PMI, lift heatmaps).
//
// Mechanism (mirrors examples/eval-moe-mmlu/eval-moe-mmlu.cpp): every MoE
// forward pass in llama.cpp materialises a small int32 tensor named
// `ffn_moe_topk-<il>` (logical shape [n_expert_used, n_tokens]). It is already
// exposed through ggml_backend_sched_eval_callback. We install a custom
// callback that filters by that tensor-name regex, copies the int32 data via
// ggml_backend_tensor_get, and stores it into a per-decode slice. After all
// decodes for a question complete (prefill + up to --gen-tokens generation
// steps), we walk the slices once and tally intra- and inter-layer pair counts
// into a per-subject accumulator. The slice vector is then cleared so we never
// hold more than one question's slices in memory.
//
// Differences from eval-moe-mmlu:
//   * Supports autoregressive generation (--gen-tokens N, default 4). Routing
//     during the model's answer-writing phase is captured too.
//   * Emits a 3D [L, E, E] intra-layer pair co-occurrence matrix per subject
//     and a 4D [L, L, E, E] inter-layer pair co-occurrence matrix (both raw
//     counts only; derived metrics are computed in Python).
//   * Tracks tokens_prefill and tokens_generated separately.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ------------------------------------------------------------------- tunables

// Number of test questions to run per MMLU subject. Change here and rebuild.
static constexpr int QUERIES_PER_SUBJECT = 50;

// Number of few-shot exemplars drawn from each subject's dev split. The OLMoE
// authors evaluate MMLU 5-shot in their published numbers.
static constexpr int N_SHOT = 5;

// Number of autoregressive generation tokens after prefill. MMLU answers are
// 1-2 tokens for the letter (A/B/C/D), plus a trailing newline and chat-
// template EOS; 4 tokens covers that with a small safety margin. Set to 0 to
// disable generation entirely (prompt-only mode, useful for control
// experiments and back-compat with the eval-moe-mmlu harness).
static constexpr int GEN_TOKENS        = 4;
// Default for params.embedding. See comment at params.embedding below.
// Overridable via --embeddings / --no-embeddings CLI flag.
static bool          g_embeddings_mode = false;

// Default inter-layer output mode. 0 = full upper-triangular [L, L, E, E]
// (back-compat); K>0 = only k=1..K off-diagonals, output as [L, K, E, E]
// (memory-friendly for large archs like DeepSeek-V3). Override via the
// --inter-k-lag CLI flag.
static constexpr int INTER_K_LAG_DEFAULT = 0;

// Default minimum pair count to emit in sparse mode (0 = emit everything).
// Set >0 to switch the inter-layer output to COO encoding that only lists
// cells with count > N. Useful for very large E (DeepSeek-V3 256 experts).
static constexpr int SPARSE_MIN_COUNT_DEFAULT = 0;

// ------------------------------------------------------------------- globals

// Per-subject pair counts. All matrices are int64 to avoid overflow on long
// runs (each cell is incremented up to n_expert_used * n_expert_used per
// token, so a 2850-question × 500-token run easily reaches ~1e9).
struct subject_pair_counts {
    // [n_layer][n_expert]   marginal firing counts (sum of intra pair counts
    //                       over j equals this for each (L, e_i)).
    std::vector<std::vector<int64_t>>                           marginal_expert_counts;
    // [n_layer][n_expert][n_expert]  co-occurrence counts within a layer.
    // intra_pair_counts[L][e_i][e_j] = # tokens where both e_i and e_j fired
    //                                  in the topk of layer L.
    std::vector<std::vector<std::vector<int64_t>>>              intra_pair_counts;
    // [n_layer][n_layer][n_expert][n_expert]  co-occurrence counts across
    // layer pairs (position-aligned: same token position t in both layers).
    // inter_pair_counts[L1][L2][e_i][e_j] = # tokens t such that e_i fired in
    //                                       layer L1 at position t AND e_j
    //                                       fired in layer L2 at position t.
    // Stored for L1 <= L2 only; symmetric read in Python.
    std::vector<std::vector<std::vector<std::vector<int64_t>>>> inter_pair_counts;

    // [n_layer][inter_k_lag][n_expert][n_expert]  alternative storage used
    // when inter_k_lag > 0. inter_klag_counts[L1][k-1][e_i][e_j] = # tokens t
    // such that e_i fired in L1 at position t AND e_j fired in layer
    // (L1 + k) at position t. Only filled when the tool is run with
    // --inter-k-lag K > 0; the inner [inter_k_lag] axis is dense (entries
    // where L1 + k >= n_layer are emitted as zeros and may be ignored by
    // downstream consumers).
    std::vector<std::vector<std::vector<std::vector<int64_t>>>> inter_klag_counts;

    int64_t tokens_prefill   = 0;
    int64_t tokens_generated = 0;
};

// One decode call (prefill OR a single generation step) contributes one slice
// of topk tensors. We keep the full set of slices for the current question
// until all decodes complete, then walk them once to tally pair counts and
// clear.
struct decode_slice {
    int                               start_offset;  // global token position of the first token in this slice
    int                               n_tokens;      // prefill_tokens.size() or 1 for a gen step
    std::vector<std::vector<int32_t>> topk;          // [n_layer][stride1 * n_tokens], stride1 = n_expert
};

struct moe_accumulator {
    int n_layer    = 0;
    int n_expert   = 0;  // total experts per MoE layer (= stride1)
    int n_expert_k = 0;  // top-k (experts chosen per token)

    std::string arch;
    std::string current_subject;

    // Per-subject pair counts.
    std::map<std::string, subject_pair_counts> subjects;

    // Buffer of slices for the question currently being processed.
    std::vector<decode_slice> current_question_slices;
    int                       current_question_n_prefill = 0;
    int                       current_question_n_gen     = 0;

    // Pointer (set in callback) to the slice currently being filled. The
    // callback is invoked per-MoE-layer per-decode, and slices map 1:1 to
    // llama_decode calls. We initialise the slice BEFORE llama_decode so the
    // callback has a destination.
    decode_slice * current_slice = nullptr;

    // Aggregate pair counts (filled at end-of-main by summing subjects).
    std::vector<std::vector<int64_t>>                           agg_marginal_expert_counts;
    std::vector<std::vector<std::vector<int64_t>>>              agg_intra_pair_counts;
    std::vector<std::vector<std::vector<std::vector<int64_t>>>> agg_inter_pair_counts;
    // Same as agg_inter_pair_counts but indexed by (L1, k) for memory-friendly
    // k-lag mode. Only populated when inter_k_lag > 0.
    std::vector<std::vector<std::vector<std::vector<int64_t>>>> agg_inter_klag_counts;
    int64_t                                                     agg_tokens_prefill   = 0;
    int64_t                                                     agg_tokens_generated = 0;

    // Phase 3 config flags (set from CLI).
    int  inter_k_lag       = INTER_K_LAG_DEFAULT;       // 0 = full upper triangular; K>0 = k-lag
    bool per_subject_inter = false;                     // emit inter-layer per subject
    int  sparse_min_count  = SPARSE_MIN_COUNT_DEFAULT;  // 0 = dense; >0 = COO sparse

    int seen_k = -1;
};

static moe_accumulator g_acc;

// ------------------------------------------------------------------- callback

static bool moe_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;

    // Filter: only ffn_moe_topk-<layer_index>
    static const std::regex re("^ffn_moe_topk-([0-9]+)$");
    std::cmatch             m;
    if (!std::regex_match(t->name, m, re)) {
        return false;
    }

    if (ask) {
        return true;
    }

    if (g_acc.current_slice == nullptr) {
        // Defensive: no slice is currently being filled. Skip.
        return true;
    }

    int il = std::stoi(m[1].str());
    if (il < 0 || il >= g_acc.n_layer) {
        LOG_WRN("%s: layer index %d out of expected range [0,%d) - skipping\n", __func__, il, g_acc.n_layer);
        return true;
    }
    if (t->type != GGML_TYPE_I32) {
        LOG_WRN("%s: tensor %s has unexpected type %s (expected I32) - skipping\n", __func__, t->name,
                ggml_type_name(t->type));
        return true;
    }

    // Shape: ne[0] = top-k, ne[1] = n_tokens
    const int k    = int(t->ne[0]);
    const int ntok = int(t->ne[1]);
    if (k <= 0 || ntok <= 0) {
        return true;
    }
    if (g_acc.seen_k < 0) {
        g_acc.seen_k = k;
        LOG_INF("%s: first ffn_moe_topk tensor: ne=[%lld,%lld,%lld] type=%s k=%d\n", __func__, (long long) t->ne[0],
                (long long) t->ne[1], (long long) t->ne[2], ggml_type_name(t->type), k);
        if (k != g_acc.n_expert_k) {
            LOG_WRN("%s: top-k from tensor (%d) does not match metadata (%d)\n", __func__, k, g_acc.n_expert_k);
        }
    }

    // `ffn_moe_topk-<il>` is a ggml_view produced by ggml_argsort_top_k from
    // a [n_expert, n_tokens] tensor. The view reports ne=[k, n_tokens] but
    // inherits nb[1] = n_expert * sizeof(int32) from the underlying argsort
    // result, so it is non-contiguous. The top-k expert IDs for token `tok`
    // live at buf[tok * n_expert + j] (NOT tok * k + j). ggml_backend_tensor_get
    // reads the full n_bytes range including the inter-token gaps, so we read
    // n_expert * ntok ints total.
    const size_t stride0 = t->nb[0] / sizeof(int32_t);  // 1
    const size_t stride1 = t->nb[1] / sizeof(int32_t);  // n_expert
    GGML_ASSERT(stride0 == 1 && "expected contiguous ints along axis 0");
    if (stride1 != (size_t) g_acc.n_expert) {
        LOG_WRN("%s: stride1 (%zu) does not match n_expert (%d) at layer %d\n", __func__, stride1, g_acc.n_expert, il);
        return true;
    }
    if (ntok != g_acc.current_slice->n_tokens) {
        LOG_WRN("%s: tensor n_tokens (%d) does not match slice n_tokens (%d) at layer %d\n", __func__, ntok,
                g_acc.current_slice->n_tokens, il);
        return true;
    }

    const size_t         buf_elems = (size_t) ntok * stride1;
    std::vector<int32_t> buf(buf_elems);
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    auto & slice_layer = g_acc.current_slice->topk[il];
    GGML_ASSERT(slice_layer.size() == buf_elems);
    std::memcpy(slice_layer.data(), buf.data(), buf_elems * sizeof(int32_t));

    return true;
}

// ------------------------------------------------------------------- pair-count tally

// Walk the slices for the current question once, computing per-subject
// intra-layer and inter-layer pair counts. The slice vector is cleared here
// so memory does not accumulate across questions.
static void tally_pair_counts_for_question() {
    auto &    counts     = g_acc.subjects[g_acc.current_subject];
    const int n_layer    = g_acc.n_layer;
    const int n_expert   = g_acc.n_expert;
    const int n_expert_k = g_acc.n_expert_k;
    const int n_gen      = g_acc.current_question_n_gen;
    const int n_prefill  = g_acc.current_question_n_prefill;

    // Defensive: n_expert_k may be larger than the actual k if metadata
    // disagrees with the first captured tensor. Use the captured k.
    const int topk_k = g_acc.seen_k > 0 ? g_acc.seen_k : n_expert_k;

    // Build a per-layer view over the full token sequence (prefill + gen).
    // For each layer L, `seq[L]` points to the contiguous topk buffer for the
    // first token at position 0; subsequent tokens are at stride1 (= n_expert)
    // apart.
    //
    // The slices are already laid out so that the i-th token of slice s starts
    // at position (s.start_offset + i) and the topk[j] for that token is at
    // slice.topk[L][i * stride1 + j]. So the global token at position p in
    // layer L is: (slice s containing p).topk[L][(p - s.start_offset) * stride1 + j].

    const int n_total = n_prefill + n_gen;

    auto get_token_topk = [&](int layer, int global_pos, int j) -> int32_t {
        // Find the slice containing global_pos.
        for (const auto & sl : g_acc.current_question_slices) {
            if (global_pos >= sl.start_offset && global_pos < sl.start_offset + sl.n_tokens) {
                const int local = global_pos - sl.start_offset;
                return sl.topk[layer][(size_t) local * n_expert + j];
            }
        }
        GGML_ASSERT(false && "token position not in any slice");
        return -1;
    };

    // Intra-layer pair counts: for each layer L, for each token t, walk all
    // k*k pairs from the topk.
    for (int L = 0; L < n_layer; ++L) {
        auto & layer_intra = counts.intra_pair_counts[L];
        for (int t = 0; t < n_total; ++t) {
            for (int j1 = 0; j1 < topk_k; ++j1) {
                const int32_t e1 = get_token_topk(L, t, j1);
                if (e1 < 0 || e1 >= n_expert) {
                    continue;
                }
                for (int j2 = 0; j2 < topk_k; ++j2) {
                    const int32_t e2 = get_token_topk(L, t, j2);
                    if (e2 < 0 || e2 >= n_expert) {
                        continue;
                    }
                    layer_intra[e1][e2] += 1;
                }
            }
        }
    }

    // Inter-layer pair counts: branch on inter_k_lag.
    if (g_acc.inter_k_lag > 0) {
        // k-lag mode: for each layer L1 and lag k in 1..inter_k_lag, count
        // co-occurrences with L2 = L1 + k. Stored as [L1][k-1].
        const int K_lag = g_acc.inter_k_lag;
        for (int L1 = 0; L1 < n_layer; ++L1) {
            for (int k = 1; k <= K_lag && (L1 + k) < n_layer; ++k) {
                const int L2          = L1 + k;
                auto &    layer_inter = counts.inter_klag_counts[L1][k - 1];
                for (int t = 0; t < n_total; ++t) {
                    for (int j1 = 0; j1 < topk_k; ++j1) {
                        const int32_t e1 = get_token_topk(L1, t, j1);
                        if (e1 < 0 || e1 >= n_expert) {
                            continue;
                        }
                        for (int j2 = 0; j2 < topk_k; ++j2) {
                            const int32_t e2 = get_token_topk(L2, t, j2);
                            if (e2 < 0 || e2 >= n_expert) {
                                continue;
                            }
                            layer_inter[e1][e2] += 1;
                        }
                    }
                }
            }
        }
    } else {
        // Full upper-triangular mode: for each pair (L1, L2) with L1 <= L2,
        // for each token t, walk topk[L1][t] x topk[L2][t].
        for (int L1 = 0; L1 < n_layer; ++L1) {
            for (int L2 = L1; L2 < n_layer; ++L2) {
                auto & layer_inter = counts.inter_pair_counts[L1][L2];
                for (int t = 0; t < n_total; ++t) {
                    for (int j1 = 0; j1 < topk_k; ++j1) {
                        const int32_t e1 = get_token_topk(L1, t, j1);
                        if (e1 < 0 || e1 >= n_expert) {
                            continue;
                        }
                        for (int j2 = 0; j2 < topk_k; ++j2) {
                            const int32_t e2 = get_token_topk(L2, t, j2);
                            if (e2 < 0 || e2 >= n_expert) {
                                continue;
                            }
                            layer_inter[e1][e2] += 1;
                        }
                    }
                }
            }
        }
    }

    counts.tokens_prefill += n_prefill;
    counts.tokens_generated += n_gen;

    g_acc.current_question_slices.clear();
    g_acc.current_question_n_prefill = 0;
    g_acc.current_question_n_gen     = 0;
    g_acc.current_slice              = nullptr;
}

// ------------------------------------------------------------------- JSON I/O

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char esc[8];
                    std::snprintf(esc, sizeof(esc), "\\u%04x", c);
                    out += esc;
                } else {
                    out += char(c);
                }
        }
    }
    return out;
}

static void write_json_int_array(FILE * f, const std::vector<int64_t> & v) {
    std::fputc('[', f);
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) {
            std::fputc(',', f);
        }
        std::fprintf(f, "%lld", static_cast<long long>(v[i]));
    }
    std::fputc(']', f);
}

static void write_json_2d_int_array(FILE * f, const std::vector<std::vector<int64_t>> & m) {
    std::fputc('[', f);
    for (size_t i = 0; i < m.size(); ++i) {
        if (i > 0) {
            std::fputc(',', f);
        }
        std::fputc('\n', f);
        std::fputc(' ', f);
        write_json_int_array(f, m[i]);
    }
    std::fputc(']', f);
}

static void write_json_3d_int_array(FILE * f, const std::vector<std::vector<std::vector<int64_t>>> & m) {
    std::fputc('[', f);
    for (size_t i = 0; i < m.size(); ++i) {
        if (i > 0) {
            std::fputc(',', f);
        }
        std::fputc('\n', f);
        std::fputc(' ', f);
        write_json_2d_int_array(f, m[i]);
    }
    std::fputc(']', f);
}

static void write_json_4d_int_array_upper_triangular(
    FILE *                                                              f,
    const std::vector<std::vector<std::vector<std::vector<int64_t>>>> & m) {
    // m has shape [L, L, E, E]. We emit a [L, L, E, E] object so consumers can
    // access by [L1][L2][e_i][e_j]. Note: we stored only L1 <= L2; the lower
    // triangle is left as zeros. Symmetry is a downstream analysis concern.
    std::fputc('[', f);
    for (size_t L1 = 0; L1 < m.size(); ++L1) {
        if (L1 > 0) {
            std::fputc(',', f);
        }
        std::fputc('\n', f);
        std::fputc(' ', f);
        write_json_3d_int_array(f, m[L1]);
    }
    std::fputc(']', f);
}

// Emit a [L, K, E, E] array (k-lag format). m[L1][k-1] corresponds to the
// pair (L1, L1 + k). Entries where L1 + k >= n_layer are emitted as zeros.
// We pre-allocate the inner dimension based on m's actual size (K).
static void write_json_4d_int_array_klag(FILE *                                                              f,
                                         const std::vector<std::vector<std::vector<std::vector<int64_t>>>> & m) {
    // m has shape [n_layer, K, E, E]
    std::fputc('[', f);
    for (size_t L1 = 0; L1 < m.size(); ++L1) {
        if (L1 > 0) {
            std::fputc(',', f);
        }
        std::fputc('\n', f);
        std::fputc(' ', f);
        write_json_3d_int_array(f, m[L1]);
    }
    std::fputc(']', f);
}

// Emit a [L, L, E, E] array as COO (coordinate, value) triples. Only cells
// with value > min_count are emitted. Output schema:
//   {"shape": [L, L, E, E], "indices": [[L1, L2, e_i, e_j], ...], "values": [v, ...]}
// Indices in the upper triangle only (L1 <= L2) when k-lag mode is not used;
// for k-lag mode, indices span [L, K, E, E] (k = L2 - L1).
static void write_json_sparse_coo(FILE *                                                              f,
                                  const std::vector<std::vector<std::vector<std::vector<int64_t>>>> & m,
                                  int                                                                 min_count,
                                  bool                                                                is_klag_format) {
    const size_t L  = m.size();
    const size_t D1 = is_klag_format ? (L > 0 ? m[0].size() : 0) : L;
    const size_t E  = (L > 0 && D1 > 0) ? m[0][0].size() : 0;

    // First pass: count
    size_t nnz = 0;
    if (is_klag_format) {
        for (size_t L1 = 0; L1 < L; ++L1) {
            for (size_t k = 0; k < D1; ++k) {
                for (size_t ei = 0; ei < E; ++ei) {
                    for (size_t ej = 0; ej < E; ++ej) {
                        if (m[L1][k][ei][ej] > min_count) {
                            nnz++;
                        }
                    }
                }
            }
        }
    } else {
        for (size_t L1 = 0; L1 < L; ++L1) {
            for (size_t L2 = L1; L2 < L; ++L2) {
                for (size_t ei = 0; ei < E; ++ei) {
                    for (size_t ej = 0; ej < E; ++ej) {
                        if (m[L1][L2][ei][ej] > min_count) {
                            nnz++;
                        }
                    }
                }
            }
        }
    }

    std::fprintf(f, "{\"shape\":[");
    if (is_klag_format) {
        std::fprintf(f, "%zu,%zu,%zu,%zu", L, D1, E, E);
    } else {
        std::fprintf(f, "%zu,%zu,%zu,%zu", L, L, E, E);
    }
    std::fprintf(f, "],\"min_count\":%d,\"nnz\":%zu,\"indices\":[", min_count, nnz);
    bool first = true;
    if (is_klag_format) {
        for (size_t L1 = 0; L1 < L; ++L1) {
            for (size_t k = 0; k < D1; ++k) {
                for (size_t ei = 0; ei < E; ++ei) {
                    for (size_t ej = 0; ej < E; ++ej) {
                        if (m[L1][k][ei][ej] > min_count) {
                            if (!first) {
                                std::fputc(',', f);
                            }
                            first = false;
                            std::fprintf(f, "[%zu,%zu,%zu,%zu]", L1, k, ei, ej);
                        }
                    }
                }
            }
        }
    } else {
        for (size_t L1 = 0; L1 < L; ++L1) {
            for (size_t L2 = L1; L2 < L; ++L2) {
                for (size_t ei = 0; ei < E; ++ei) {
                    for (size_t ej = 0; ej < E; ++ej) {
                        if (m[L1][L2][ei][ej] > min_count) {
                            if (!first) {
                                std::fputc(',', f);
                            }
                            first = false;
                            std::fprintf(f, "[%zu,%zu,%zu,%zu]", L1, L2, ei, ej);
                        }
                    }
                }
            }
        }
    }
    std::fprintf(f, "],\"values\":[");
    first = true;
    if (is_klag_format) {
        for (size_t L1 = 0; L1 < L; ++L1) {
            for (size_t k = 0; k < D1; ++k) {
                for (size_t ei = 0; ei < E; ++ei) {
                    for (size_t ej = 0; ej < E; ++ej) {
                        if (m[L1][k][ei][ej] > min_count) {
                            if (!first) {
                                std::fputc(',', f);
                            }
                            first = false;
                            std::fprintf(f, "%lld", (long long) m[L1][k][ei][ej]);
                        }
                    }
                }
            }
        }
    } else {
        for (size_t L1 = 0; L1 < L; ++L1) {
            for (size_t L2 = L1; L2 < L; ++L2) {
                for (size_t ei = 0; ei < E; ++ei) {
                    for (size_t ej = 0; ej < E; ++ej) {
                        if (m[L1][L2][ei][ej] > min_count) {
                            if (!first) {
                                std::fputc(',', f);
                            }
                            first = false;
                            std::fprintf(f, "%lld", (long long) m[L1][L2][ei][ej]);
                        }
                    }
                }
            }
        }
    }
    std::fprintf(f, "]}");
}

// ------------------------------------------------------------------- MMLU I/O

struct mmlu_row {
    std::string              split;
    std::string              subject;
    std::string              question;
    std::vector<std::string> choices;  // 4 entries, letter-prefixed
    int                      answer = 0;
};

static std::vector<mmlu_row> load_mmlu_jsonl(const std::string & path) {
    std::vector<mmlu_row> rows;
    std::ifstream         in(path);
    if (!in) {
        LOG_ERR("cannot open mmlu jsonl: %s\n", path.c_str());
        return rows;
    }

    auto skip_ws = [](const std::string & line, size_t p) -> size_t {
        while (p < line.size() && std::isspace((unsigned char) line[p])) {
            ++p;
        }
        return p;
    };

    auto extract_string_field = [&](const std::string & line, const std::string & key,
                                    size_t start = 0) -> std::string {
        std::string needle = "\"" + key + "\"";
        size_t      p      = line.find(needle, start);
        if (p == std::string::npos) {
            return {};
        }
        p += needle.size();
        p = skip_ws(line, p);
        if (p >= line.size() || line[p] != ':') {
            return {};
        }
        ++p;
        p = skip_ws(line, p);
        if (p >= line.size() || line[p] != '"') {
            return {};
        }
        ++p;
        std::string out;
        while (p < line.size()) {
            char c = line[p++];
            if (c == '\\' && p < line.size()) {
                char n = line[p++];
                switch (n) {
                    case '"':
                        out += '"';
                        break;
                    case '\\':
                        out += '\\';
                        break;
                    case 'n':
                        out += '\n';
                        break;
                    case 't':
                        out += '\t';
                        break;
                    case 'r':
                        out += '\r';
                        break;
                    default:
                        out += n;
                }
                continue;
            }
            if (c == '"') {
                break;
            }
            out += c;
        }
        return out;
    };

    auto extract_int_field = [&](const std::string & line, const std::string & key, size_t start = 0) -> int {
        std::string needle = "\"" + key + "\"";
        size_t      p      = line.find(needle, start);
        if (p == std::string::npos) {
            return -1;
        }
        p += needle.size();
        p = skip_ws(line, p);
        if (p >= line.size() || line[p] != ':') {
            return -1;
        }
        ++p;
        p        = skip_ws(line, p);
        int  v   = 0;
        bool any = false;
        bool neg = false;
        if (p < line.size() && line[p] == '-') {
            neg = true;
            ++p;
        }
        while (p < line.size() && std::isdigit((unsigned char) line[p])) {
            v = v * 10 + (line[p] - '0');
            ++p;
            any = true;
        }
        return any ? (neg ? -v : v) : -1;
    };

    auto extract_choices = [&](const std::string & line, size_t start) -> std::vector<std::string> {
        std::vector<std::string> out;
        std::string              needle = "\"choices\"";
        size_t                   p      = line.find(needle, start);
        if (p == std::string::npos) {
            return out;
        }
        p += needle.size();
        p = skip_ws(line, p);
        if (p >= line.size() || line[p] != ':') {
            return out;
        }
        ++p;
        p = skip_ws(line, p);
        if (p >= line.size() || line[p] != '[') {
            return out;
        }
        ++p;
        while (p < line.size() && out.size() < 4) {
            p = skip_ws(line, p);
            if (p < line.size() && line[p] == ',') {
                ++p;
                p = skip_ws(line, p);
            }
            if (p >= line.size() || line[p] != '"') {
                break;
            }
            ++p;
            std::string s;
            while (p < line.size() && line[p] != '"') {
                if (line[p] == '\\' && p + 1 < line.size()) {
                    s += line[p];
                    s += line[p + 1];
                    p += 2;
                } else {
                    s += line[p++];
                }
            }
            if (p < line.size() && line[p] == '"') {
                ++p;
            }
            out.push_back(std::move(s));
        }
        return out;
    };

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] != '{') {
            continue;
        }
        mmlu_row r;
        r.split    = extract_string_field(line, "split");
        r.subject  = extract_string_field(line, "subject");
        r.question = extract_string_field(line, "question");
        r.choices  = extract_choices(line, 0);
        r.answer   = extract_int_field(line, "answer");
        if (r.split.empty() || r.choices.size() != 4 || r.answer < 0) {
            LOG_WRN("skipping malformed mmlu row: %.80s...\n", line.c_str());
            continue;
        }
        rows.push_back(std::move(r));
    }
    return rows;
}

static std::vector<std::string> load_subjects(const std::string & path) {
    std::vector<std::string> out;
    std::ifstream            in(path);
    if (!in) {
        LOG_ERR("cannot open subjects list: %s\n", path.c_str());
        return out;
    }
    std::string s;
    while (std::getline(in, s)) {
        if (!s.empty()) {
            out.push_back(s);
        }
    }
    return out;
}

// ------------------------------------------------------------------- prompt

static std::string build_question_text(const mmlu_row & q) {
    std::ostringstream os;
    os << q.question << "\n\n";
    for (const auto & c : q.choices) {
        os << c << "\n";
    }
    os << "Answer:";
    return os.str();
}

static std::string build_fewshot_user_text(const std::vector<mmlu_row> & dev_rows, const mmlu_row & test_row) {
    std::ostringstream os;
    for (int i = 0; i < (int) dev_rows.size(); ++i) {
        const auto & d = dev_rows[i];
        os << "Question: " << d.question << "\n";
        for (const auto & c : d.choices) {
            os << c << "\n";
        }
        os << "Answer: " << char('A' + d.answer) << "\n\n";
    }
    os << "Question: " << test_row.question << "\n";
    for (const auto & c : test_row.choices) {
        os << c << "\n";
    }
    os << "Answer:";
    return os.str();
}

// ------------------------------------------------------------------- main

static void print_usage(int argc, char ** argv) {
    (void) argc;
    fprintf(
        stderr,
        "usage: %s [standard llama.cpp args] [options]\n"
        "\n"
        "Required:\n"
        "  -m, --model <path>          local GGUF path (or use -hf .../...)\n"
        "\n"
        "MMLU inputs (defaults shown):\n"
        "      --mmlu <path>           path to mmlu.jsonl     (default: build/moe-mmlu/mmlu.jsonl)\n"
        "      --subjects <path>       path to subjects.txt   (default: build/moe-mmlu/subjects.txt)\n"
        "\n"
        "Run control:\n"
        "      --questions-per-subject <N>   override QUERIES_PER_SUBJECT (default %d)\n"
        "      --n-shots <N>                override N_SHOT             (default %d)\n"
        "      --gen-tokens <N>             autoregressive decode cap   (default %d, 0 = prompt-only)\n"
        "\n"
        "Output control (Phase 3, large-arch memory):\n"
        "      --inter-k-lag <K>             restrict inter-layer to k=1..K only (default 0 = full upper triangular).\n"
        "                                    Output is then [L, K, E, E] instead of [L, L, E, E].\n"
        "      --per-subject-inter           also emit inter-layer pair counts per subject (default: aggregate only)\n"
        "      --sparse-min-count <N>        emit inter-layer as COO {indices, values, shape} keeping only cells > N\n"
        "\n"
        "Output:\n"
        "  -o, --output <path>         output JSON (default: build/moe-coactivation/coactivation.json)\n"
        "\n"
        "Standard llama.cpp flags (from common_params_parse) are also accepted: -ngl, -c, --seed, etc.\n",
        argv[0], QUERIES_PER_SUBJECT, N_SHOT, GEN_TOKENS);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // ----- custom args: extract from argv BEFORE common_params_parse so that
    // unknown flags don't trip its strict parser.
    int         questions_per_subject = QUERIES_PER_SUBJECT;
    int         n_shot                = N_SHOT;
    int         gen_tokens            = GEN_TOKENS;
    int         inter_k_lag           = INTER_K_LAG_DEFAULT;
    bool        per_subject_inter     = false;
    int         sparse_min_count      = SPARSE_MIN_COUNT_DEFAULT;
    std::string mmlu_path             = "build/moe-mmlu/mmlu.jsonl";
    std::string subjects_path         = "build/moe-mmlu/subjects.txt";
    std::string output_path           = "build/moe-coactivation/coactivation.json";

    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) {
                LOG_ERR("%s requires an argument (%s)\n", a.c_str(), what);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--mmlu") {
            mmlu_path = next("path");
        } else if (a == "--subjects") {
            subjects_path = next("path");
        } else if (a == "--questions-per-subject") {
            questions_per_subject = std::stoi(next("N"));
        } else if (a == "--n-shots") {
            n_shot = std::stoi(next("N"));
        } else if (a == "--gen-tokens") {
            gen_tokens = std::stoi(next("N"));
        } else if (a == "--inter-k-lag") {
            inter_k_lag = std::stoi(next("K"));
        } else if (a == "--per-subject-inter") {
            per_subject_inter = true;
        } else if (a == "--sparse-min-count") {
            sparse_min_count = std::stoi(next("N"));
        } else if (a == "-o" || a == "--output") {
            output_path = next("path");
        } else if (a == "--embeddings") {
            g_embeddings_mode = true;
        } else if (a == "--no-embeddings") {
            g_embeddings_mode = false;
        }
    }

    std::vector<char *> filtered;
    filtered.reserve(argc);
    filtered.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mmlu" || a == "--subjects" || a == "--questions-per-subject" || a == "--n-shots" ||
            a == "--gen-tokens" || a == "--inter-k-lag" || a == "--sparse-min-count") {
            ++i;
            continue;
        }
        if (a == "-o" || a == "--output") {
            ++i;
            continue;
        }
        // --per-subject-inter takes no value; drop it from the filtered argv
        if (a == "--per-subject-inter") {
            continue;
        }
        filtered.push_back(argv[i]);
    }

    common_params params;
    // Default n_batch=2048 is too small for some 5-shot MMLU prompts; bump it.
    params.n_batch  = 4096;
    params.n_ubatch = 4096;

    if (!common_params_parse((int) filtered.size(), filtered.data(), params, LLAMA_EXAMPLE_COMMON)) {
        print_usage(argc, argv);
        return 1;
    }

    // -------- load inputs
    LOG_INF("loading mmlu jsonl: %s\n", mmlu_path.c_str());
    auto mmlu_rows = load_mmlu_jsonl(mmlu_path);
    if (mmlu_rows.empty()) {
        LOG_ERR("no mmlu rows loaded - run download_mmlu.py first\n");
        return 1;
    }
    LOG_INF("  loaded %zu rows\n", mmlu_rows.size());

    LOG_INF("loading subjects list: %s\n", subjects_path.c_str());
    auto subjects = load_subjects(subjects_path);
    if (subjects.empty()) {
        LOG_ERR("no subjects loaded\n");
        return 1;
    }
    LOG_INF("  %zu subjects\n", subjects.size());

    std::map<std::string, std::vector<const mmlu_row *>> by_subject_dev;
    std::map<std::string, std::vector<const mmlu_row *>> by_subject_test;
    for (const auto & r : mmlu_rows) {
        if (r.split == "dev") {
            by_subject_dev[r.subject].push_back(&r);
        }
        if (r.split == "test") {
            by_subject_test[r.subject].push_back(&r);
        }
    }

    // -------- init llama
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = moe_eval_callback;
    params.cb_eval_user_data = &g_acc;
    params.warmup            = false;

    // Force `cparams.embeddings = true` -> `output_all = true` in llama_decode
    //
    // NB: the cb_eval hook fires for every ffn_moe_topk-<il> tensor in the
    // compute graph regardless of params.embedding. Setting embedding=true
    // is therefore NOT required to capture routing counts - it only affects
    // whether the OUTPUT tensor is materialised. On long multilingual
    // prefills (e.g. Greek/INCLUDE questions), embedding=true triggers a
    // CUDA "illegal memory access" in the MoE routing kernel (observed on
    // A100 PCIe + A100 NVLink, both llama.cpp master and e5df8bfb8).
    // Default embedding=false (set --embeddings to opt back in to the old
    // behaviour for A/B testing).
    params.embedding = g_embeddings_mode;

    auto   init_result = common_init_from_params(params);
    auto * model       = init_result->model();
    auto * ctx         = init_result->context();
    if (!model || !ctx) {
        LOG_ERR("failed to load model/context\n");
        return 1;
    }

    g_acc.n_layer = llama_model_n_layer(model);
    LOG_INF("model n_layer = %d\n", g_acc.n_layer);

    auto read_meta_int = [&](const char * key, int fallback) -> int {
        char    buf[64];
        int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
        if (n > 0) {
            try {
                return std::stoi(std::string(buf, n));
            } catch (...) { /* fall through */
            }
        }
        return fallback;
    };

    char        arch_buf[64] = { 0 };
    int32_t     arch_n       = llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf));
    std::string arch;
    if (arch_n > 0 && arch_n < (int32_t) sizeof(arch_buf)) {
        arch.assign(arch_buf, arch_n);
    } else {
        LOG_WRN("could not read general.architecture (returned %d) - falling back to olmoe key naming\n", arch_n);
        arch = "olmoe";
    }
    LOG_INF("model architecture (from GGUF) = %s\n", arch.c_str());
    g_acc.arch = arch;

    int n_expert_meta   = read_meta_int((arch + ".expert_count").c_str(), -1);
    int n_expert_k_meta = read_meta_int((arch + ".expert_used_count").c_str(), -1);

    if (n_expert_meta <= 0) {
        n_expert_meta = 64;
    }
    if (n_expert_k_meta <= 0) {
        n_expert_k_meta = 8;
    }
    g_acc.n_expert   = n_expert_meta;
    g_acc.n_expert_k = n_expert_k_meta;
    LOG_INF("model n_expert = %d, n_expert_used = %d\n", g_acc.n_expert, g_acc.n_expert_k);

    // Phase 3 config validation and propagation.
    if (inter_k_lag < 0) {
        LOG_ERR("--inter-k-lag must be >= 0 (got %d)\n", inter_k_lag);
        return 1;
    }
    if (inter_k_lag > g_acc.n_layer) {
        LOG_WRN("--inter-k-lag %d exceeds n_layer %d - clamping to %d\n", inter_k_lag, g_acc.n_layer, g_acc.n_layer);
        inter_k_lag = g_acc.n_layer;
    }
    g_acc.inter_k_lag       = inter_k_lag;
    g_acc.per_subject_inter = per_subject_inter;
    g_acc.sparse_min_count  = sparse_min_count;
    LOG_INF("output config: inter_k_lag=%d (0 = full), per_subject_inter=%s, sparse_min_count=%d\n", g_acc.inter_k_lag,
            g_acc.per_subject_inter ? "true" : "false", g_acc.sparse_min_count);

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    // Sampler (greedy). Mirrors examples/eval-moe-popqa/eval-moe-popqa.cpp.
    auto sparams         = llama_sampler_chain_default_params();
    sparams.no_perf      = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const llama_token   eos_tok = llama_vocab_eos(vocab);

    // -------- chat template
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) {
        LOG_ERR("model has no chat template - cannot build few-shot prompts\n");
        return 1;
    }

    // -------- pre-allocate per-subject pair-count structures
    // When inter_k_lag > 0 we use inter_klag_counts (shape [L, K, E, E]) and
    // skip inter_pair_counts to save memory. When K=0 we use inter_pair_counts
    // (shape [L, L, E, E] upper triangular) for back-compat.
    for (const auto & subj : subjects) {
        subject_pair_counts spc;
        spc.marginal_expert_counts.assign(g_acc.n_layer, std::vector<int64_t>(g_acc.n_expert, 0));
        spc.intra_pair_counts.assign(
            g_acc.n_layer, std::vector<std::vector<int64_t>>(g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0)));
        if (g_acc.inter_k_lag > 0) {
            spc.inter_klag_counts.assign(
                g_acc.n_layer, std::vector<std::vector<std::vector<int64_t>>>(
                                   g_acc.inter_k_lag, std::vector<std::vector<int64_t>>(
                                                          g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0))));
        } else {
            spc.inter_pair_counts.assign(
                g_acc.n_layer, std::vector<std::vector<std::vector<int64_t>>>(
                                   g_acc.n_layer, std::vector<std::vector<int64_t>>(
                                                      g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0))));
        }
        g_acc.subjects[subj] = std::move(spc);
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    // -------- main loop
    int        total_questions        = 0;
    int64_t    total_tokens_prefill   = 0;
    int64_t    total_tokens_generated = 0;
    const auto t_run_start            = std::chrono::steady_clock::now();

    for (size_t si = 0; si < subjects.size(); ++si) {
        const std::string & subj = subjects[si];
        g_acc.current_subject    = subj;

        const auto & dev_it  = by_subject_dev.find(subj);
        const auto & test_it = by_subject_test.find(subj);
        if (test_it == by_subject_test.end() || test_it->second.empty()) {
            LOG_WRN("subject %s has no test rows - skipping\n", subj.c_str());
            continue;
        }
        if (n_shot > 0 && (dev_it == by_subject_dev.end() || dev_it->second.empty())) {
            LOG_WRN("subject %s has no dev rows - falling back to 0-shot\n", subj.c_str());
        }

        std::vector<mmlu_row> dev_rows;
        if (dev_it != by_subject_dev.end()) {
            int take = std::min<int>(n_shot, (int) dev_it->second.size());
            for (int j = 0; j < take; ++j) {
                dev_rows.push_back(*dev_it->second[j]);
            }
        }

        const auto & test_rows = test_it->second;
        const int    n_q       = std::min<int>(questions_per_subject, (int) test_rows.size());

        LOG_INF("[%zu/%zu] %s - %d questions\n", si + 1, subjects.size(), subj.c_str(), n_q);

        for (int qi = 0; qi < n_q; ++qi) {
            const mmlu_row & q = *test_rows[qi];

            // Build chat messages.
            std::vector<llama_chat_message> msgs;
            std::string                     sys_text =
                "The following are multiple choice questions (with answers).\n"
                "Answer with only the letter A, B, C, or D.";
            msgs.push_back({ "system", strdup(sys_text.c_str()) });

            std::string user_text = n_shot > 0 ? build_fewshot_user_text(dev_rows, q) : build_question_text(q);
            msgs.push_back({ "user", strdup(user_text.c_str()) });
            msgs.push_back({ "assistant", strdup("") });

            std::vector<char> fmt(8192);
            int               n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(),
                                                            /*add_ass*/ true, fmt.data(), (int) fmt.size());
            if (n < 0) {
                LOG_ERR("chat template application failed (returned %d)\n", n);
                for (auto & m : msgs) {
                    free(const_cast<char *>(m.content));
                }
                continue;
            }
            if (n >= (int) fmt.size()) {
                fmt.resize(n + 1);
                n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, fmt.data(), (int) fmt.size());
            }
            std::string prompt(fmt.data(), n);

            std::vector<llama_token> toks = common_tokenize(ctx, prompt, add_bos, true);
            if (toks.empty()) {
                LOG_WRN("  empty tokenization for %s q%d - skipping\n", subj.c_str(), qi);
                for (auto & m : msgs) {
                    free(const_cast<char *>(m.content));
                }
                continue;
            }

            const int n_batch_max = llama_n_batch(ctx);
            if ((int) toks.size() > n_batch_max) {
                LOG_WRN("  skipping %s q%d - tokenized prompt (%zu) exceeds n_batch (%d)\n", subj.c_str(), qi,
                        toks.size(), n_batch_max);
                for (auto & m : msgs) {
                    free(const_cast<char *>(m.content));
                }
                continue;
            }

            // -------- prefill decode
            {
                decode_slice slice;
                slice.start_offset = 0;
                slice.n_tokens     = (int) toks.size();
                slice.topk.assign(g_acc.n_layer, std::vector<int32_t>((size_t) g_acc.n_expert * toks.size(), 0));
                g_acc.current_question_slices.push_back(std::move(slice));
                g_acc.current_slice              = &g_acc.current_question_slices.back();
                g_acc.current_question_n_prefill = (int) toks.size();
            }

            llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("  llama_decode (prefill) failed for %s q%d\n", subj.c_str(), qi);
                for (auto & m : msgs) {
                    free(const_cast<char *>(m.content));
                }
                // Discard the slices for this question.
                g_acc.current_question_slices.clear();
                g_acc.current_question_n_prefill = 0;
                g_acc.current_question_n_gen     = 0;
                g_acc.current_slice              = nullptr;
                continue;
            }

            // -------- autoregressive decode (skipped when --gen-tokens 0)
            int n_gen = 0;
            if (gen_tokens > 0) {
                for (int step = 0; step < gen_tokens; ++step) {
                    llama_token id = llama_sampler_sample(smpl, ctx, -1);
                    if (id == eos_tok || llama_vocab_is_eog(vocab, id)) {
                        break;
                    }

                    decode_slice slice;
                    slice.start_offset = (int) toks.size() + step;
                    slice.n_tokens     = 1;
                    slice.topk.assign(g_acc.n_layer, std::vector<int32_t>((size_t) g_acc.n_expert, 0));
                    g_acc.current_question_slices.push_back(std::move(slice));
                    g_acc.current_slice = &g_acc.current_question_slices.back();

                    llama_batch next_batch = llama_batch_get_one(&id, 1);
                    if (llama_decode(ctx, next_batch) != 0) {
                        LOG_ERR("  llama_decode (gen step %d) failed for %s q%d\n", step, subj.c_str(), qi);
                        break;
                    }
                    n_gen++;
                }
            }
            g_acc.current_question_n_gen = n_gen;

            // -------- tally pair counts and flush slices for this question
            tally_pair_counts_for_question();

            total_questions++;
            total_tokens_prefill += (int64_t) toks.size();
            total_tokens_generated += n_gen;

            for (auto & m : msgs) {
                free(const_cast<char *>(m.content));
            }

            // Clear KV cache between questions.
            if (llama_get_memory(ctx) != nullptr) {
                llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
            }
        }
    }

    const auto   t_run_end = std::chrono::steady_clock::now();
    const double secs      = std::chrono::duration<double>(t_run_end - t_run_start).count();

    LOG_INF("done: %d questions in %.1f s (%.1f q/s) - prefill=%lld, generated=%lld\n", total_questions, secs,
            total_questions / std::max(1.0, secs), (long long) total_tokens_prefill,
            (long long) total_tokens_generated);

    // -------- compute marginals from intra pair counts (sanity-checked) and aggregate
    // We rebuild marginal_expert_counts as the sum over j of intra_pair_counts (which
    // is the count of token-and-j pairs that fired each expert, i.e. marginal with
    // top-k multiplicity). For per-expert marginal (count of tokens where e fired),
    // divide by k. Here we report the k-multiplied form so the JSON matches what
    // eval-moe-mmlu emits (also k-multiplied).
    for (auto & kv : g_acc.subjects) {
        auto & spc = kv.second;
        for (int L = 0; L < g_acc.n_layer; ++L) {
            for (int e1 = 0; e1 < g_acc.n_expert; ++e1) {
                int64_t s = 0;
                for (int e2 = 0; e2 < g_acc.n_expert; ++e2) {
                    s += spc.intra_pair_counts[L][e1][e2];
                }
                spc.marginal_expert_counts[L][e1] = s;
            }
        }
    }

    // Aggregate across subjects. Branch on inter_k_lag.
    g_acc.agg_marginal_expert_counts.assign(g_acc.n_layer, std::vector<int64_t>(g_acc.n_expert, 0));
    g_acc.agg_intra_pair_counts.assign(
        g_acc.n_layer, std::vector<std::vector<int64_t>>(g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0)));
    g_acc.agg_tokens_prefill   = 0;
    g_acc.agg_tokens_generated = 0;

    if (g_acc.inter_k_lag > 0) {
        // k-lag aggregate
        g_acc.agg_inter_klag_counts.assign(
            g_acc.n_layer, std::vector<std::vector<std::vector<int64_t>>>(
                               g_acc.inter_k_lag, std::vector<std::vector<int64_t>>(
                                                      g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0))));
        for (const auto & kv : g_acc.subjects) {
            const auto & spc = kv.second;
            g_acc.agg_tokens_prefill += spc.tokens_prefill;
            g_acc.agg_tokens_generated += spc.tokens_generated;
            for (int L = 0; L < g_acc.n_layer; ++L) {
                for (int e = 0; e < g_acc.n_expert; ++e) {
                    g_acc.agg_marginal_expert_counts[L][e] += spc.marginal_expert_counts[L][e];
                    for (int e2 = 0; e2 < g_acc.n_expert; ++e2) {
                        g_acc.agg_intra_pair_counts[L][e][e2] += spc.intra_pair_counts[L][e][e2];
                        for (int k = 1; k <= g_acc.inter_k_lag && (L + k) < g_acc.n_layer; ++k) {
                            g_acc.agg_inter_klag_counts[L][k - 1][e][e2] += spc.inter_klag_counts[L][k - 1][e][e2];
                        }
                    }
                }
            }
        }
    } else {
        // Full upper-triangular aggregate (back-compat)
        g_acc.agg_inter_pair_counts.assign(
            g_acc.n_layer, std::vector<std::vector<std::vector<int64_t>>>(
                               g_acc.n_layer, std::vector<std::vector<int64_t>>(
                                                  g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0))));
        for (const auto & kv : g_acc.subjects) {
            const auto & spc = kv.second;
            g_acc.agg_tokens_prefill += spc.tokens_prefill;
            g_acc.agg_tokens_generated += spc.tokens_generated;
            for (int L = 0; L < g_acc.n_layer; ++L) {
                for (int e = 0; e < g_acc.n_expert; ++e) {
                    g_acc.agg_marginal_expert_counts[L][e] += spc.marginal_expert_counts[L][e];
                    for (int e2 = 0; e2 < g_acc.n_expert; ++e2) {
                        g_acc.agg_intra_pair_counts[L][e][e2] += spc.intra_pair_counts[L][e][e2];
                        for (int L2 = L; L2 < g_acc.n_layer; ++L2) {
                            g_acc.agg_inter_pair_counts[L][L2][e][e2] += spc.inter_pair_counts[L][L2][e][e2];
                        }
                    }
                }
            }
        }
    }

    // -------- write JSON
    {
        std::string dir = output_path.substr(0, output_path.find_last_of('/'));
        if (!dir.empty()) {
            std::string cmd = "mkdir -p '" + dir + "'";
            if (std::system(cmd.c_str()) != 0) {
                LOG_WRN("could not create output directory %s\n", dir.c_str());
            }
        }
    }

    FILE * fout = std::fopen(output_path.c_str(), "w");
    if (!fout) {
        LOG_ERR("cannot open output for writing: %s\n", output_path.c_str());
        return 1;
    }

    int subjects_with_data = 0;
    for (const auto & kv : g_acc.subjects) {
        if (kv.second.tokens_prefill > 0 || kv.second.tokens_generated > 0) {
            subjects_with_data++;
        }
    }

    std::fprintf(fout, "{\n");
    std::fprintf(fout, "  \"model\": \"%s\",\n", json_escape(params.model.get_name()).c_str());
    std::fprintf(fout, "  \"model_arch\": {\n");
    std::fprintf(fout, "    \"name\": \"%s\",\n", json_escape(g_acc.arch).c_str());
    std::fprintf(fout, "    \"n_layer\": %d,\n", g_acc.n_layer);
    std::fprintf(fout, "    \"n_expert\": %d,\n", g_acc.n_expert);
    std::fprintf(fout, "    \"n_expert_used\": %d\n", g_acc.n_expert_k);
    std::fprintf(fout, "  },\n");
    std::fprintf(fout, "  \"config\": {\n");
    std::fprintf(fout, "    \"questions_per_subject\": %d,\n", questions_per_subject);
    std::fprintf(fout, "    \"n_shot\": %d,\n", n_shot);
    std::fprintf(fout, "    \"gen_tokens\": %d,\n", gen_tokens);
    std::fprintf(fout, "    \"inter_k_lag\": %d,\n", g_acc.inter_k_lag);
    std::fprintf(fout, "    \"per_subject_inter\": %s,\n", g_acc.per_subject_inter ? "true" : "false");
    std::fprintf(fout, "    \"sparse_min_count\": %d,\n", g_acc.sparse_min_count);
    std::fprintf(fout, "    \"few_shot_pool\": \"cais/mmlu dev split\",\n");
    std::fprintf(fout, "    \"prompt_format\": \"few_shot_chat\"\n");
    std::fprintf(fout, "  },\n");
    std::fprintf(fout, "  \"totals\": {\n");
    std::fprintf(fout, "    \"subjects_run\": %d,\n", subjects_with_data);
    std::fprintf(fout, "    \"questions_total\": %d,\n", total_questions);
    std::fprintf(fout, "    \"tokens_prefill_total\": %lld,\n", (long long) total_tokens_prefill);
    std::fprintf(fout, "    \"tokens_generated_total\": %lld,\n", (long long) total_tokens_generated);
    std::fprintf(fout, "    \"tokens_total\": %lld\n", (long long) (total_tokens_prefill + total_tokens_generated));
    std::fprintf(fout, "  },\n");
    std::fprintf(fout, "  \"aggregate\": {\n");
    std::fprintf(fout, "    \"tokens_prefill\": %lld,\n", (long long) g_acc.agg_tokens_prefill);
    std::fprintf(fout, "    \"tokens_generated\": %lld,\n", (long long) g_acc.agg_tokens_generated);
    std::fprintf(fout, "    \"tokens_total\": %lld,\n",
                 (long long) (g_acc.agg_tokens_prefill + g_acc.agg_tokens_generated));
    std::fprintf(fout, "    \"marginal_expert_counts\": ");
    write_json_2d_int_array(fout, g_acc.agg_marginal_expert_counts);
    std::fprintf(fout, ",\n    \"intra_pair_counts\": ");
    write_json_3d_int_array(fout, g_acc.agg_intra_pair_counts);
    if (g_acc.inter_k_lag > 0) {
        // k-lag mode: emit [L, K, E, E] (or COO if sparse)
        if (g_acc.sparse_min_count > 0) {
            std::fprintf(fout, ",\n    \"inter_klag_counts_sparse\": ");
            write_json_sparse_coo(fout, g_acc.agg_inter_klag_counts, g_acc.sparse_min_count, /*is_klag_format=*/true);
        } else {
            std::fprintf(fout, ",\n    \"inter_klag_counts\": ");
            write_json_4d_int_array_klag(fout, g_acc.agg_inter_klag_counts);
        }
    } else {
        // full upper-triangular mode (back-compat)
        if (g_acc.sparse_min_count > 0) {
            std::fprintf(fout, ",\n    \"inter_pair_counts_sparse\": ");
            write_json_sparse_coo(fout, g_acc.agg_inter_pair_counts, g_acc.sparse_min_count, /*is_klag_format=*/false);
        } else {
            std::fprintf(fout, ",\n    \"inter_pair_counts\": ");
            write_json_4d_int_array_upper_triangular(fout, g_acc.agg_inter_pair_counts);
        }
    }
    std::fprintf(fout, "\n  },\n");
    std::fprintf(fout, "  \"subjects\": {\n");

    bool first_subj = true;
    for (const auto & kv : g_acc.subjects) {
        const std::string & subj_name = kv.first;
        const auto &        spc       = kv.second;
        if (spc.tokens_prefill == 0 && spc.tokens_generated == 0) {
            continue;
        }

        if (!first_subj) {
            std::fprintf(fout, ",\n");
        }
        first_subj = false;

        std::fprintf(fout, "    \"%s\": {\n", json_escape(subj_name).c_str());
        std::fprintf(fout, "      \"questions\": %d,\n",
                     (int) std::min<int64_t>(questions_per_subject, (int64_t) by_subject_test[subj_name].size()));
        std::fprintf(fout, "      \"tokens_prefill\": %lld,\n", (long long) spc.tokens_prefill);
        std::fprintf(fout, "      \"tokens_generated\": %lld,\n", (long long) spc.tokens_generated);
        std::fprintf(fout, "      \"tokens_total\": %lld,\n", (long long) (spc.tokens_prefill + spc.tokens_generated));
        std::fprintf(fout, "      \"marginal_expert_counts\": ");
        write_json_2d_int_array(fout, spc.marginal_expert_counts);
        std::fprintf(fout, ",\n      \"intra_pair_counts\": ");
        write_json_3d_int_array(fout, spc.intra_pair_counts);
        // Per-subject inter-layer output: only when --per-subject-inter is set
        if (g_acc.per_subject_inter) {
            if (g_acc.inter_k_lag > 0) {
                if (g_acc.sparse_min_count > 0) {
                    std::fprintf(fout, ",\n      \"inter_klag_counts_sparse\": ");
                    write_json_sparse_coo(fout, spc.inter_klag_counts, g_acc.sparse_min_count, /*is_klag_format=*/true);
                } else {
                    std::fprintf(fout, ",\n      \"inter_klag_counts\": ");
                    write_json_4d_int_array_klag(fout, spc.inter_klag_counts);
                }
            } else {
                if (g_acc.sparse_min_count > 0) {
                    std::fprintf(fout, ",\n      \"inter_pair_counts_sparse\": ");
                    write_json_sparse_coo(fout, spc.inter_pair_counts, g_acc.sparse_min_count,
                                          /*is_klag_format=*/false);
                } else {
                    std::fprintf(fout, ",\n      \"inter_pair_counts\": ");
                    write_json_4d_int_array_upper_triangular(fout, spc.inter_pair_counts);
                }
            }
        }
        std::fprintf(fout, "\n    }");
    }

    std::fprintf(fout, "\n  }\n");
    std::fprintf(fout, "}\n");
    std::fclose(fout);

    LOG_INF("wrote %s\n", output_path.c_str());

    llama_sampler_free(smpl);
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
