// llama-eval-moe-humaneval
//
// Runs an MoE LLM (e.g. allenai/OLMoE-1B-7B-0125-Instruct-GGUF) over a slice
// of the openai/openai_humaneval dataset and records, per problem
// (identified by upstream `task_id`), how often each expert cell was
// activated at each MoE layer. Output is a single JSON file suitable for
// post-hoc analysis in Python.
//
// Like eval-moe-bigbench, this tool autoregressively decodes up to
// --gen-tokens after prefill. The generated text is persisted verbatim per
// problem in the output JSON so downstream tooling can decide what to do
// with it (run pass@1 scoring, compare to canonical_solution, etc.).
//
// Crucially, **this tool never executes the generated Python and never
// scores it for correctness** -- the `match_metric` field of the output
// JSON is the literal string "none". Routing counts include both prefill
// and generation tokens in a single combined [n_layer][n_expert] matrix
// per problem, so the result is directly comparable to eval-moe-popqa and
// eval-moe-bigbench.
//
// Mechanism: every MoE forward pass in llama.cpp materialises a small
// int32 tensor named "ffn_moe_topk-<il>" (shape [n_expert_used, n_tokens]).
// It is already exposed through ggml_backend_sched_eval_callback. We
// install a custom callback that filters by that tensor-name regex,
// copies the int32 data via ggml_backend_tensor_get, and tallies counts
// into a per-task_id matrix.
//
// It is arch-agnostic: any MoE model whose GGUF exposes
// <arch>.expert_count and <arch>.expert_used_count works (e.g. OLMoE,
// Mixtral, DeepSeek-MoE, Qwen2/3 MoE).

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <sys/stat.h>
#include <sys/types.h>

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

// Number of HumanEval problems to run per task_id. HumanEval only has 164
// distinct task_ids, so this is effectively an upper bound; the binary
// silently clamps to whatever is available.
static constexpr int QUESTIONS_PER_TASK = 164;

// Number of generation tokens after prefill. Set to 0 to disable
// generation entirely (prompt-only mode, useful for control experiments).
// The canonical HumanEval paper uses pass@1 with greedy decoding and a
// 200-token cap per problem; we default to 256 to leave headroom for the
// model to emit a final newline (which is decoded and counted, matching
// the eval-moe-bigbench semantics).
static constexpr int GEN_TOKENS = 256;

// Default sampling: greedy, deterministic. Matches the canonical HumanEval
// paper protocol (Chen et al., 2021, "Evaluating Large Language Models
// Trained on Code", arXiv:2107.03374).
static constexpr int DEFAULT_SEED = 42;

// ------------------------------------------------------------------- globals

struct moe_accumulator {
    int n_layer    = 0;
    int n_expert   = 0;  // total experts per MoE layer
    int n_expert_k = 0;  // top-k (experts chosen per token)

    // model architecture name from general.architecture
    std::string arch;

    // current task_id (problem) under evaluation
    std::string current_task_id;

    // per task_id: task_id -> vector<vector<int64_t>> of size [n_layer][n_expert]
    std::map<std::string, std::vector<std::vector<int64_t>>> counts;
    // per task_id: task_id -> number of prefill tokens (prompt encoding)
    std::map<std::string, int64_t>                           tokens_prefill;
    // per task_id: task_id -> number of generated tokens (autoregressive decode)
    std::map<std::string, int64_t>                           tokens_generated;
    // per task_id: task_id -> number of problems attempted (denominator)
    std::map<std::string, int>                               n_questions;
    // per task_id: task_id -> did the generated completion hit --gen-tokens
    // without ever emitting EOS/EOG?
    std::map<std::string, bool>                              completion_truncated;
    // per task_id: task_id -> full generated completion (utf-8 text)
    std::map<std::string, std::string>                       completion_text;
    // per task_id: task_id -> function entry_point (carried through from JSONL)
    std::map<std::string, std::string>                       entry_point;

    // per task_id: task_id -> vector<vector<vector<int64_t>>> [n_layer][n_expert][n_expert]
    // intra_pair_counts[task][L][e_i][e_j] = # tokens where both e_i and e_j fired
    //                                         in the top-k of layer L (k * k slots).
    std::map<std::string, std::vector<std::vector<std::vector<int64_t>>>> intra_counts;
    // per task_id: task_id -> vector<vector<vector<int64_t>>> [n_layer - 1][n_expert][n_expert]
    // adj_pair_counts[task][L][e_i][e_j] = # tokens t such that e_i fired in layer L at t
    //                                       AND e_j fired in layer L + 1 at t (k * k slots).
    // Layer index n_layer - 1 is omitted (no L + 1 exists).
    std::map<std::string, std::vector<std::vector<std::vector<int64_t>>>> adj_counts;

    // Slice buffer: current_topk[il] holds the raw top-k expert IDs (stride1 = n_expert,
    // not k) for the current decode. Populated by moe_eval_callback; consumed by
    // tally_pairs_for_question() right after llama_decode() returns.
    // current_topk[il].size() = stride1 * n_tokens for the current decode.
    std::vector<std::vector<int32_t>> current_topk;
    int                               current_n_tokens = 0;

    // expose shape on first capture so we can validate later tensors
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

    // Allocate host buffer and copy tensor data.
    // Note: `ffn_moe_topk-<il>` is a ggml_view produced by ggml_argsort_top_k
    // from a [n_expert, n_tokens] tensor. The view reports ne=[k, n_tokens]
    // but inherits nb[1] = n_expert * sizeof(int32) from the underlying
    // argsort result, so it is non-contiguous. The top-k expert IDs for
    // token `tok` live at buf[tok * n_expert + j] (NOT tok * k + j).
    // ggml_backend_tensor_get reads the full n_bytes range including the
    // inter-token gaps, so we read n_expert * ntok ints total.
    const size_t stride0 = t->nb[0] / sizeof(int32_t);  // 1
    const size_t stride1 = t->nb[1] / sizeof(int32_t);  // n_expert
    GGML_ASSERT(stride0 == 1 && "expected contiguous ints along axis 0");
    const size_t         buf_elems = (size_t) ntok * stride1;
    std::vector<int32_t> buf(buf_elems);
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    auto & layer_counts = g_acc.counts[g_acc.current_task_id][il];
    for (int tok = 0; tok < ntok; ++tok) {
        const int32_t * row = buf.data() + (size_t) tok * stride1;
        for (int j = 0; j < k; ++j) {
            const int32_t eid = row[j];
            if (eid >= 0 && eid < g_acc.n_expert) {
                layer_counts[eid] += 1;
            } else {
                LOG_WRN("%s: expert id %d out of range [0,%d) at layer %d, token %d\n", __func__, int(eid),
                        g_acc.n_expert, il, tok);
            }
        }
    }

    // Also copy the raw top-k into the per-decode slice buffer so
    // tally_pairs_for_question() can compute intra- and adjacent-layer pair
    // counts position-aligned across layers. This duplicates the tensor copy
    // but avoids changing the marginal-tally hot path's stride math.
    // current_topk[il] has stride1 = n_expert elements per token (interleaved
    // gaps), matching the ggml_view layout.
    if ((int) g_acc.current_topk.size() != g_acc.n_layer) {
        g_acc.current_topk.assign(g_acc.n_layer, std::vector<int32_t>());
    }
    if (g_acc.current_topk[il].size() != buf_elems) {
        g_acc.current_topk[il].resize(buf_elems);
    }
    std::memcpy(g_acc.current_topk[il].data(), buf.data(), buf_elems * sizeof(int32_t));
    if (g_acc.current_n_tokens < ntok) {
        g_acc.current_n_tokens = ntok;
    }

    return true;
}

// Walks the per-decode slice buffer (filled by moe_eval_callback) and tallies
// intra-layer [L, E, E] and adjacent-layer [L - 1, E, E] pair counts for the
// current task_id. Each token contributes k * k increments per layer pair
// (one for each (j1, j2) top-k slot pair). Must be called after every
// llama_decode() and before llama_memory_clear().
static void tally_pairs_for_question(moe_accumulator & acc) {
    const int    n_layer  = acc.n_layer;
    const int    n_expert = acc.n_expert;
    const int    ntok     = acc.current_n_tokens;
    const size_t stride1  = (size_t) n_expert;  // slice layout: n_expert per token

    if (ntok <= 0 || n_layer <= 0) {
        return;
    }
    if ((int) acc.current_topk.size() != n_layer) {
        LOG_WRN("%s: current_topk size %zu != n_layer %d - skipping\n", __func__, acc.current_topk.size(), n_layer);
        return;
    }

    auto get = [&](int L, int tok, int j) -> int32_t {
        const auto & layer = acc.current_topk[L];
        return layer[(size_t) tok * stride1 + (size_t) j];
    };

    const std::string & subj  = acc.current_task_id;
    auto &              intra = acc.intra_counts[subj];
    auto &              adj   = acc.adj_counts[subj];

    // Intra-layer: for each layer L, for each token t, walk all (j1, j2) top-k
    // slot pairs from the same layer's slice.
    for (int L = 0; L < n_layer; ++L) {
        if ((int) intra.size() <= L) {
            continue;  // task_id was skipped (no rows)
        }
        auto & layer_intra = intra[L];
        for (int tok = 0; tok < ntok; ++tok) {
            for (int j1 = 0; j1 < acc.n_expert_k; ++j1) {
                const int32_t e1 = get(L, tok, j1);
                if (e1 < 0 || e1 >= n_expert) {
                    continue;
                }
                for (int j2 = 0; j2 < acc.n_expert_k; ++j2) {
                    const int32_t e2 = get(L, tok, j2);
                    if (e2 < 0 || e2 >= n_expert) {
                        continue;
                    }
                    layer_intra[e1][e2] += 1;
                }
            }
        }
    }

    // Adjacent-layer: position-aligned across (L, L + 1). Skip if n_layer < 2.
    if (n_layer >= 2 && (int) adj.size() >= n_layer - 1) {
        for (int L = 0; L < n_layer - 1; ++L) {
            auto & layer_adj = adj[L];
            for (int tok = 0; tok < ntok; ++tok) {
                for (int j1 = 0; j1 < acc.n_expert_k; ++j1) {
                    const int32_t e1 = get(L, tok, j1);
                    if (e1 < 0 || e1 >= n_expert) {
                        continue;
                    }
                    for (int j2 = 0; j2 < acc.n_expert_k; ++j2) {
                        const int32_t e2 = get(L + 1, tok, j2);
                        if (e2 < 0 || e2 >= n_expert) {
                            continue;
                        }
                        layer_adj[e1][e2] += 1;
                    }
                }
            }
        }
    }

    // Release the per-decode slice buffer immediately to bound memory at
    // ~n_layer * stride1 * ntok * 4 bytes (~30-60 KB for OLMoE prefill).
    acc.current_n_tokens = 0;
    for (auto & layer : acc.current_topk) {
        std::vector<int32_t>().swap(layer);
    }
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

// ------------------------------------------------------------------- HumanEval I/O

struct humaneval_row {
    std::string task_id;             // upstream id, e.g. "HumanEval/0" or "test/0"
    std::string prompt;              // function header + docstring
    std::string canonical_solution;  // kept in JSONL for downstream tooling; unused at runtime
    std::string test;                // kept in JSONL for downstream tooling; unused at runtime
    std::string entry_point;         // function name (e.g. "has_close_elements")
};

// Minimal ad-hoc JSON line parser. Each row is one object on its own line.
// Fields are extracted by name; unknown fields are ignored.
static std::vector<humaneval_row> load_humaneval_jsonl(const std::string & path) {
    std::vector<humaneval_row> rows;
    std::ifstream              in(path);
    if (!in) {
        LOG_ERR("cannot open humaneval jsonl: %s\n", path.c_str());
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

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] != '{') {
            continue;
        }
        humaneval_row r;
        r.task_id            = extract_string_field(line, "task_id");
        r.prompt             = extract_string_field(line, "prompt");
        r.canonical_solution = extract_string_field(line, "canonical_solution");
        r.test               = extract_string_field(line, "test");
        r.entry_point        = extract_string_field(line, "entry_point");
        if (r.task_id.empty() || r.prompt.empty() || r.entry_point.empty()) {
            LOG_WRN("skipping malformed humaneval row: %.80s...\n", line.c_str());
            continue;
        }
        rows.push_back(std::move(r));
    }
    return rows;
}

static std::vector<std::string> load_tasks(const std::string & path) {
    std::vector<std::string> out;
    std::ifstream            in(path);
    if (!in) {
        LOG_ERR("cannot open tasks list: %s\n", path.c_str());
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

// HumanEval's `prompt` is already the model input: a Python function header
// followed by a docstring. We feed it verbatim (no chat template, no
// Q: ... A: wrapper, no manual indentation) -- this matches the canonical
// HumanEval paper protocol and keeps the routing signal focused on the
// problem itself.
static std::string build_humaneval_prompt(const humaneval_row & r) {
    return r.prompt;
}

// ------------------------------------------------------------------- main

static void print_usage(int argc, char ** argv) {
    (void) argc;
    fprintf(stderr,
            "usage: %s [standard llama.cpp args] [options]\n"
            "\n"
            "Required:\n"
            "  -m, --model <path>          local GGUF path (or use -hf .../...)\n"
            "\n"
            "HumanEval inputs (defaults shown):\n"
            "      --humaneval <path>      path to humaneval.jsonl (default: build/moe-humaneval/humaneval.jsonl)\n"
            "      --tasks <path>          path to tasks.txt       (default: build/moe-humaneval/tasks.txt)\n"
            "\n"
            "Run control:\n"
            "      --questions-per-task <N>   override QUESTIONS_PER_TASK (default %d)\n"
            "      --gen-tokens <N>           autoregressive decode cap  (default %d, 0 = prompt-only)\n"
            "\n"
            "Output:\n"
            "  -o, --output <path>         output JSON (default: build/moe-humaneval/expert_counts.json)\n"
            "\n"
            "Note: this tool never executes the generated Python and never scores it\n"
            "for correctness -- the `match_metric` field of the output JSON is the\n"
            "literal string 'none'. The full generated completion is persisted per\n"
            "problem in the output for downstream pass@1 / canonical-solution\n"
            "comparison tooling.\n"
            "\n"
            "Standard llama.cpp flags (from common_params_parse) are also accepted: -ngl, -c, --seed, etc.\n",
            argv[0], QUESTIONS_PER_TASK, GEN_TOKENS);
}

// Best-effort mkdir -p (parents created if missing). Used before opening
// the output file so --output build/foo/bar/expert_counts.json works
// without forcing the user to mkdir manually.
static int mkdir_p(const std::string & dir) {
    if (dir.empty()) {
        return 0;
    }
    std::string accum;
    for (char c : dir) {
        accum.push_back(c);
        if (c == '/') {
            if (::mkdir(accum.c_str(), 0755) != 0 && errno != EEXIST) {
                return -1;
            }
        }
    }
    if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // ----- custom args: extract from argv BEFORE common_params_parse so that
    // unknown flags don't trip its strict parser.
    int         questions_per_task = QUESTIONS_PER_TASK;
    int         gen_tokens         = GEN_TOKENS;
    std::string humaneval_path     = "build/moe-humaneval/humaneval.jsonl";
    std::string tasks_path         = "build/moe-humaneval/tasks.txt";
    std::string output_path        = "build/moe-humaneval/expert_counts.json";

    // First pass: extract custom args.
    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) {
                LOG_ERR("%s requires an argument (%s)\n", a.c_str(), what);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--humaneval") {
            humaneval_path = next("path");
        } else if (a == "--tasks") {
            tasks_path = next("path");
        } else if (a == "--questions-per-task") {
            questions_per_task = std::stoi(next("N"));
        } else if (a == "--gen-tokens") {
            gen_tokens = std::stoi(next("N"));
        } else if (a == "-o" || a == "--output") {
            output_path = next("path");
        }
    }

    // Second pass: build a filtered argv without our custom args.
    std::vector<char *> filtered;
    filtered.reserve(argc);
    filtered.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--humaneval" || a == "--tasks" || a == "--questions-per-task" || a == "--gen-tokens") {
            ++i;  // skip value too
            continue;
        }
        if (a == "-o" || a == "--output") {
            ++i;  // skip value too
            continue;
        }
        filtered.push_back(argv[i]);
    }

    common_params params;
    // Default n_batch=2048 can be too small for some prompts; bump it.
    params.n_batch  = 4096;
    params.n_ubatch = 4096;

    if (!common_params_parse((int) filtered.size(), filtered.data(), params, LLAMA_EXAMPLE_COMMON)) {
        print_usage(argc, argv);
        return 1;
    }

    // Seed the sampler deterministically. LLAMA_EXAMPLE_COMMON does not
    // necessarily set params.sampling.seed; default to DEFAULT_SEED unless
    // overridden.
    if (params.sampling.seed == LLAMA_DEFAULT_SEED) {
        params.sampling.seed = DEFAULT_SEED;
    }

    // -------- load inputs
    LOG_INF("loading humaneval jsonl: %s\n", humaneval_path.c_str());
    auto humaneval_rows = load_humaneval_jsonl(humaneval_path);
    if (humaneval_rows.empty()) {
        LOG_ERR("no humaneval rows loaded - run download_humaneval.py first\n");
        return 1;
    }
    LOG_INF("  loaded %zu rows\n", humaneval_rows.size());

    LOG_INF("loading tasks list: %s\n", tasks_path.c_str());
    auto tasks = load_tasks(tasks_path);
    if (tasks.empty()) {
        LOG_ERR("no tasks loaded\n");
        return 1;
    }
    LOG_INF("  %zu tasks\n", tasks.size());

    // index rows by task_id
    std::map<std::string, const humaneval_row *> by_task_id;
    for (const auto & r : humaneval_rows) {
        // If multiple rows share a task_id, the first wins (deterministic).
        if (by_task_id.find(r.task_id) == by_task_id.end()) {
            by_task_id[r.task_id] = &r;
        }
    }

    // -------- init llama
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // Wire the eval callback BEFORE context creation (params.cb_eval is
    // consumed when common_init_from_params builds the context). With
    // params.embedding=true the cparams.embeddings flag forces
    // output_all=true on every llama_decode, so the prefill routes all
    // tokens through every MoE layer (without this only the last position
    // flows through MoE because inp_out_ids has size 1).
    params.cb_eval           = moe_eval_callback;
    params.cb_eval_user_data = &g_acc;
    params.warmup            = false;
    params.embedding         = true;

    auto   init_result = common_init_from_params(params);
    auto * model       = init_result->model();
    auto * ctx         = init_result->context();
    if (!model || !ctx) {
        LOG_ERR("failed to load model/context\n");
        return 1;
    }

    // Discover MoE hparams. The public API doesn't expose n_expert /
    // n_expert_used, so we read them from GGUF metadata (arch-specific
    // keys), with sensible OLMoE-shaped fallbacks.
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
        LOG_WRN(
            "could not read general.architecture (returned %d) - "
            "falling back to olmoe key naming\n",
            arch_n);
        arch = "olmoe";
    }
    LOG_INF("model architecture (from GGUF) = %s\n", arch.c_str());
    g_acc.arch = arch;

    int n_expert_meta   = -1;
    int n_expert_k_meta = -1;
    n_expert_meta       = read_meta_int((arch + ".expert_count").c_str(), -1);
    n_expert_k_meta     = read_meta_int((arch + ".expert_used_count").c_str(), -1);

    if (n_expert_meta <= 0) {
        n_expert_meta = 64;
    }
    if (n_expert_k_meta <= 0) {
        n_expert_k_meta = 8;
    }

    g_acc.n_expert   = n_expert_meta;
    g_acc.n_expert_k = n_expert_k_meta;
    LOG_INF("model n_expert = %d, n_expert_used = %d\n", g_acc.n_expert, g_acc.n_expert_k);

    // Print system info
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    // Build the sampler once (greedy, deterministic). Greedy matches the
    // canonical HumanEval pass@1 protocol and gives a reproducible JSON
    // across runs with the same seed.
    auto sparams         = llama_sampler_chain_default_params();
    sparams.no_perf      = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const llama_token   eos_tok = llama_vocab_eos(vocab);

    // -------- pre-allocate per-task count matrices + counters
    for (const auto & task_id : tasks) {
        g_acc.counts[task_id].assign(g_acc.n_layer, std::vector<int64_t>(g_acc.n_expert, 0));
        g_acc.tokens_prefill[task_id]       = 0;
        g_acc.tokens_generated[task_id]     = 0;
        g_acc.n_questions[task_id]          = 0;
        g_acc.completion_truncated[task_id] = false;
        g_acc.completion_text[task_id]      = "";
        g_acc.entry_point[task_id]          = "";
        g_acc.intra_counts[task_id].assign(
            g_acc.n_layer, std::vector<std::vector<int64_t>>(g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0)));
        if (g_acc.n_layer >= 2) {
            g_acc.adj_counts[task_id].assign(
                g_acc.n_layer - 1,
                std::vector<std::vector<int64_t>>(g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0)));
        }
    }

    // -------- main loop: for each task_id, for each problem attempt
    int        total_questions        = 0;
    int        completions_truncated  = 0;
    int64_t    total_tokens_prefill   = 0;
    int64_t    total_tokens_generated = 0;
    const auto t_run_start            = std::chrono::steady_clock::now();

    for (size_t ti = 0; ti < tasks.size(); ++ti) {
        const std::string & task_id = tasks[ti];
        g_acc.current_task_id       = task_id;

        const auto & task_it = by_task_id.find(task_id);
        if (task_it == by_task_id.end() || task_it->second == nullptr) {
            LOG_WRN("task_id %s has no rows - skipping\n", task_id.c_str());
            continue;
        }
        const humaneval_row & r   = *task_it->second;
        const int             n_q = std::min<int>(questions_per_task, 1);  // HumanEval has at most 1 row per task_id

        // Carry entry_point through to the output even if we skip the decode.
        g_acc.entry_point[task_id] = r.entry_point;

        LOG_INF("[%zu/%zu] %s (%s) - %d problem\n", ti + 1, tasks.size(), task_id.c_str(), r.entry_point.c_str(), n_q);

        for (int qi = 0; qi < n_q; ++qi) {
            // Build prompt + tokenize. We deliberately bypass the chat
            // template here and feed the HumanEval prompt verbatim --
            // HumanEval is a code-completion task; any wrapper would
            // contaminate the routing signal.
            const std::string        prompt  = build_humaneval_prompt(r);
            const bool               add_bos = llama_vocab_get_add_bos(vocab);
            std::vector<llama_token> toks    = common_tokenize(ctx, prompt, add_bos, true);
            if (toks.empty()) {
                LOG_WRN("  empty tokenization for %s - skipping\n", task_id.c_str());
                continue;
            }

            // Safety: skip prompts that exceed the model's batch limit. The
            // GGML_ASSERT in llama-context.cpp is a hard crash, not a
            // recoverable error, so we must guard here. Long HumanEval
            // prompts (with extensive docstrings) can approach this limit
            // for small context windows.
            const int n_batch_max = llama_n_batch(ctx);
            if ((int) toks.size() > n_batch_max) {
                LOG_WRN("  skipping %s - tokenized prompt (%zu) exceeds n_batch (%d)\n", task_id.c_str(), toks.size(),
                        n_batch_max);
                continue;
            }

            // -------- prefill
            llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("  llama_decode (prefill) failed for %s\n", task_id.c_str());
                continue;
            }

            // Walk the slice buffer to tally intra/adjacent pair counts
            // for the prefill pass.
            tally_pairs_for_question(g_acc);

            g_acc.tokens_prefill[task_id] += (int64_t) toks.size();
            total_tokens_prefill += (int64_t) toks.size();
            g_acc.n_questions[task_id] += 1;

            // -------- autoregressive decode (skipped when --gen-tokens 0)
            //
            // Stop conditions (fixed-budget semantics):
            //   1. EOS / EOG sampled -- generated completion is finalized
            //   2. step == gen_tokens -- generated completion is truncated
            //
            // We deliberately do NOT apply a newline / dedent / blank-line
            // heuristic: the user asked for "decode a given number of
            // tokens" so we always decode exactly `gen_tokens` non-EOG
            // tokens (or stop early on EOG). This keeps the generated-
            // token count trivially comparable across problems and across
            // archs.
            std::string completion;
            int         n_gen          = 0;
            bool        stopped_on_eog = false;
            for (int step = 0; step < gen_tokens; ++step) {
                llama_token id = llama_sampler_sample(smpl, ctx, -1);
                if (id == eos_tok || llama_vocab_is_eog(vocab, id)) {
                    stopped_on_eog = true;
                    break;
                }

                const std::string piece = common_token_to_piece(ctx, id, /*special=*/true);
                completion += piece;
                n_gen++;

                // Feed sampled token back through the model so the
                // eval callback captures routing for this generation
                // step too.
                llama_batch next_batch = llama_batch_get_one(&id, 1);
                if (llama_decode(ctx, next_batch) != 0) {
                    LOG_ERR("  llama_decode (gen step %d) failed for %s\n", step, task_id.c_str());
                    break;
                }
                // Walk the slice buffer to tally intra/adjacent pair
                // counts for this single generated token.
                tally_pairs_for_question(g_acc);
            }

            g_acc.tokens_generated[task_id] += n_gen;
            total_tokens_generated += n_gen;

            // A problem is "truncated" iff we exited the decode loop at the
            // --gen-tokens bound without ever emitting EOG. We record this
            // both per-task and in the totals for downstream tooling.
            const bool truncated                = !stopped_on_eog && (n_gen >= gen_tokens) && (gen_tokens > 0);
            g_acc.completion_truncated[task_id] = truncated;
            g_acc.completion_text[task_id]      = completion;
            if (truncated) {
                completions_truncated++;
            }

            total_questions++;

            // -------- logging
            char preview[200];
            std::strncpy(preview, completion.c_str(), sizeof(preview) - 1);
            preview[sizeof(preview) - 1] = '\0';
            for (size_t i = 0; i < strlen(preview); ++i) {
                if (preview[i] == '\n') {
                    preview[i] = ' ';
                }
            }
            LOG_INF("  [%d/%d] task_id=%s entry=%s gen=%d eog=%s trunc=%s | %s\n", qi + 1, n_q, task_id.c_str(),
                    r.entry_point.c_str(), n_gen, stopped_on_eog ? "yes" : "no ", truncated ? "yes" : "no ", preview);

            // Clear the KV cache for this sequence so it doesn't
            // accumulate across the many problems (default n_ctx = 4096
            // would otherwise overflow after a handful of long HumanEval
            // problems).
            if (llama_get_memory(ctx) != nullptr) {
                llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
            }
        }
    }

    llama_sampler_free(smpl);

    const auto   t_run_end = std::chrono::steady_clock::now();
    const double secs      = std::chrono::duration<double>(t_run_end - t_run_start).count();

    LOG_INF("done: %d problems in %.1f s (%.1f q/s) - truncated=%d - prefill=%lld gen=%lld\n", total_questions, secs,
            total_questions / std::max(1.0, secs), completions_truncated, (long long) total_tokens_prefill,
            (long long) total_tokens_generated);

    // -------- write JSON
    {
        std::string dir;
        const auto  pos = output_path.find_last_of('/');
        if (pos != std::string::npos) {
            dir = output_path.substr(0, pos);
        }
        if (!dir.empty()) {
            if (mkdir_p(dir) != 0) {
                LOG_WRN("could not create output directory %s\n", dir.c_str());
            }
        }
    }

    FILE * fout = std::fopen(output_path.c_str(), "w");
    if (!fout) {
        LOG_ERR("cannot open output for writing: %s\n", output_path.c_str());
        return 1;
    }

    int tasks_with_data = 0;
    for (const auto & kv : g_acc.counts) {
        if (!kv.second.empty()) {
            tasks_with_data++;
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
    std::fprintf(fout, "    \"questions_per_task\": %d,\n", questions_per_task);
    std::fprintf(fout, "    \"gen_tokens\": %d,\n", gen_tokens);
    std::fprintf(fout, "    \"prompt_format\": \"%s\",\n", "humaneval_zero_shot_continuation");
    std::fprintf(fout, "    \"match_metric\": \"%s\"\n", "none");
    std::fprintf(fout, "  },\n");
    std::fprintf(fout, "  \"dataset\": \"%s\",\n", json_escape("openai/openai_humaneval").c_str());
    std::fprintf(fout, "  \"split\": \"%s\",\n", json_escape("test").c_str());
    std::fprintf(fout, "  \"totals\": {\n");
    std::fprintf(fout, "    \"tasks_run\": %d,\n", tasks_with_data);
    std::fprintf(fout, "    \"questions_total\": %d,\n", total_questions);
    std::fprintf(fout, "    \"tokens_total_prefill\": %lld,\n", (long long) total_tokens_prefill);
    std::fprintf(fout, "    \"tokens_total_generated\": %lld,\n", (long long) total_tokens_generated);
    std::fprintf(fout, "    \"completions_truncated\": %d\n", completions_truncated);
    std::fprintf(fout, "  },\n");

    // ---- aggregate: sum per-task_id matrices into a dataset-wide view.
    // Schema mirrors the per-task keys but uses `marginal_expert_counts`
    // for clarity (per-task keeps `layer_expert_counts` for backward
    // compatibility with heatmap_from_cpp.py).
    std::vector<std::vector<int64_t>> marginal_total(g_acc.n_layer, std::vector<int64_t>(g_acc.n_expert, 0));
    std::vector<std::vector<std::vector<int64_t>>> intra_total;
    std::vector<std::vector<std::vector<int64_t>>> adj_total;
    if (g_acc.n_layer > 0) {
        intra_total.assign(g_acc.n_layer,
                           std::vector<std::vector<int64_t>>(g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0)));
        if (g_acc.n_layer >= 2) {
            adj_total.assign(g_acc.n_layer - 1, std::vector<std::vector<int64_t>>(
                                                    g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0)));
        }
    }
    for (const auto & kv : g_acc.counts) {
        const auto & task_id = kv.first;
        const auto & mat     = kv.second;
        if (mat.empty()) {
            continue;
        }
        for (int L = 0; L < g_acc.n_layer; ++L) {
            for (int e = 0; e < g_acc.n_expert; ++e) {
                marginal_total[L][e] += mat[L][e];
            }
        }
        const auto & intra_it = g_acc.intra_counts.find(task_id);
        if (intra_it != g_acc.intra_counts.end() && (int) intra_it->second.size() == g_acc.n_layer) {
            for (int L = 0; L < g_acc.n_layer; ++L) {
                for (int e1 = 0; e1 < g_acc.n_expert; ++e1) {
                    for (int e2 = 0; e2 < g_acc.n_expert; ++e2) {
                        intra_total[L][e1][e2] += intra_it->second[L][e1][e2];
                    }
                }
            }
        }
        const auto & adj_it = g_acc.adj_counts.find(task_id);
        if (adj_it != g_acc.adj_counts.end() && (int) adj_it->second.size() == g_acc.n_layer - 1) {
            for (int L = 0; L < g_acc.n_layer - 1; ++L) {
                for (int e1 = 0; e1 < g_acc.n_expert; ++e1) {
                    for (int e2 = 0; e2 < g_acc.n_expert; ++e2) {
                        adj_total[L][e1][e2] += adj_it->second[L][e1][e2];
                    }
                }
            }
        }
    }

    std::fprintf(fout, "  \"aggregate\": {\n");
    std::fprintf(fout, "    \"marginal_expert_counts\": ");
    write_json_2d_int_array(fout, marginal_total);
    std::fprintf(fout, ",\n    \"intra_pair_counts\": ");
    write_json_3d_int_array(fout, intra_total);
    std::fprintf(fout, ",\n    \"adjacent_pair_counts\": ");
    write_json_3d_int_array(fout, adj_total);
    std::fprintf(fout, "\n  },\n");

    std::fprintf(fout, "  \"tasks\": {\n");

    bool first_task = true;
    for (const auto & kv : g_acc.counts) {
        const std::string & task_id      = kv.first;
        const auto &        layer_counts = kv.second;
        if (layer_counts.empty()) {
            continue;
        }

        if (!first_task) {
            std::fprintf(fout, ",\n");
        }
        first_task = false;

        const int n_q = g_acc.n_questions[task_id];

        std::fprintf(fout, "    \"%s\": {\n", json_escape(task_id).c_str());
        std::fprintf(fout, "      \"entry_point\": \"%s\",\n", json_escape(g_acc.entry_point[task_id]).c_str());
        std::fprintf(fout, "      \"questions\": %d,\n", n_q);
        std::fprintf(fout, "      \"n_tokens_prefill\": %lld,\n", (long long) g_acc.tokens_prefill[task_id]);
        std::fprintf(fout, "      \"n_tokens_generated\": %lld,\n", (long long) g_acc.tokens_generated[task_id]);
        std::fprintf(fout, "      \"completion_truncated\": %s,\n",
                     g_acc.completion_truncated[task_id] ? "true" : "false");
        std::fprintf(fout, "      \"layer_expert_counts\": ");
        write_json_2d_int_array(fout, layer_counts);

        const auto & intra_it = g_acc.intra_counts.find(task_id);
        if (intra_it != g_acc.intra_counts.end() && (int) intra_it->second.size() == g_acc.n_layer) {
            std::fprintf(fout, ",\n      \"intra_pair_counts\": ");
            write_json_3d_int_array(fout, intra_it->second);
        }
        const auto & adj_it = g_acc.adj_counts.find(task_id);
        if (adj_it != g_acc.adj_counts.end() && (int) adj_it->second.size() == g_acc.n_layer - 1) {
            std::fprintf(fout, ",\n      \"adjacent_pair_counts\": ");
            write_json_3d_int_array(fout, adj_it->second);
        }
        std::fprintf(fout, ",\n");
        std::fprintf(fout, "      \"completion\": \"%s\"\n", json_escape(g_acc.completion_text[task_id]).c_str());
        std::fprintf(fout, "    }");
    }

    std::fprintf(fout, "\n  }\n");
    std::fprintf(fout, "}\n");
    std::fclose(fout);

    LOG_INF("wrote %s\n", output_path.c_str());

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
