// llama-eval-moe-mmlu
//
// Runs an MoE LLM (e.g. allenai/OLMoE-1B-7B-0125-Instruct-GGUF) over a slice
// of the MMLU dataset and records, per subject, how often each expert cell was
// activated at each MoE layer. Output is a single JSON file suitable for
// post-hoc analysis in Python.
//
// Mechanism: every MoE forward pass in llama.cpp materialises a small int32
// tensor named "ffn_moe_topk-<il>" (shape [n_expert_used, n_tokens]). It is
// already exposed through ggml_backend_sched_eval_callback. We install a custom
// callback that filters by that tensor-name regex, copies the int32 data via
// ggml_backend_tensor_get, and tallies counts into a per-subject matrix.

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

// ------------------------------------------------------------------- globals

struct moe_accumulator {
    int n_layer    = 0;
    int n_expert   = 0;  // total experts per MoE layer
    int n_expert_k = 0;  // top-k (experts chosen per token)

    // model architecture name from general.architecture
    std::string arch;

    // current subject under evaluation
    std::string current_subject;

    // per subject: subject -> vector<vector<int64_t>> of size [n_layer][n_expert]
    std::map<std::string, std::vector<std::vector<int64_t>>>              counts;
    // per subject: subject -> vector<vector<vector<int64_t>>> [n_layer][n_expert][n_expert]
    // intra_pair_counts[subj][L][e_i][e_j] = # tokens where both e_i and e_j fired
    //                                       in the top-k of layer L (k * k slots).
    std::map<std::string, std::vector<std::vector<std::vector<int64_t>>>> intra_counts;
    // per subject: subject -> vector<vector<vector<int64_t>>> [n_layer - 1][n_expert][n_expert]
    // adj_pair_counts[subj][L][e_i][e_j] = # tokens t such that e_i fired in layer L at t
    //                                     AND e_j fired in layer L + 1 at t (k * k slots).
    // Layer index n_layer - 1 is omitted (no L + 1 exists).
    std::map<std::string, std::vector<std::vector<std::vector<int64_t>>>> adj_counts;
    // per subject: subject -> number of forward-passed tokens
    std::map<std::string, int64_t>                                        tokens;

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

    auto & layer_counts = g_acc.counts[g_acc.current_subject][il];
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
// current subject. Each token contributes k * k increments per layer pair
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

    const std::string & subj  = acc.current_subject;
    auto &              intra = acc.intra_counts[subj];
    auto &              adj   = acc.adj_counts[subj];

    // Intra-layer: for each layer L, for each token t, walk all (j1, j2) top-k
    // slot pairs from the same layer's slice.
    for (int L = 0; L < n_layer; ++L) {
        if ((int) intra.size() <= L) {
            continue;  // subject was skipped (no test rows)
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
    fprintf(stderr,
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
            "\n"
            "Output:\n"
            "  -o, --output <path>         output JSON (default: build/moe-mmlu/expert_counts.json)\n"
            "\n"
            "Standard llama.cpp flags (from common_params_parse) are also accepted: -ngl, -c, --seed, etc.\n",
            argv[0], QUERIES_PER_SUBJECT, N_SHOT);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // ----- custom args: extract from argv BEFORE common_params_parse so that
    // unknown flags don't trip its strict parser.
    int         questions_per_subject = QUERIES_PER_SUBJECT;
    int         n_shot                = N_SHOT;
    std::string mmlu_path             = "build/moe-mmlu/mmlu.jsonl";
    std::string subjects_path         = "build/moe-mmlu/subjects.txt";
    std::string output_path           = "build/moe-mmlu/expert_counts.json";

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
        if (a == "--mmlu") {
            mmlu_path = next("path");
        } else if (a == "--subjects") {
            subjects_path = next("path");
        } else if (a == "--questions-per-subject") {
            questions_per_subject = std::stoi(next("N"));
        } else if (a == "--n-shots") {
            n_shot = std::stoi(next("N"));
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
        if (a == "--mmlu" || a == "--subjects" || a == "--questions-per-subject" || a == "--n-shots") {
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

    // index rows by (subject, split)
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

    // Wire the eval callback BEFORE context creation (params.cb_eval is consumed
    // when common_init_from_params builds the context).
    params.cb_eval           = moe_eval_callback;
    params.cb_eval_user_data = &g_acc;
    params.warmup            = false;  // skip the warmup decode

    // Force `cparams.embeddings = true` -> `output_all = true` in llama_decode.
    // Without this, only the last token has logits=1 so `inp_out_ids` has size 1
    // and the last MoE layer (gated by `ggml_get_rows(cur, inp_out_ids)` in
    // olmoe.cpp) only routes that single token. OLMoE is a generative model,
    // not an embedding model, but `embedding` is just a flag here - it doesn't
    // change the graph topology beyond flipping `output_all`.
    params.embedding = true;

    auto   init_result = common_init_from_params(params);
    auto * model       = init_result->model();
    auto * ctx         = init_result->context();
    if (!model || !ctx) {
        LOG_ERR("failed to load model/context\n");
        return 1;
    }

    // Discover MoE hparams. The public API doesn't expose n_expert /
    // n_expert_used, so we read them from GGUF metadata (arch-specific keys),
    // and fall back to common_ggml_ne-based detection from the first ffn_moe_topk
    // tensor shape if metadata is missing.
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
    // Read the model's architecture from general.architecture and derive the
    // MoE hparam keys as "<arch>.expert_count" / "<arch>.expert_used_count".
    // This matches the convention llama.cpp uses internally (see
    // src/llama-arch.cpp LLM_KV_EXPERT_COUNT / LLM_KV_EXPERT_USED_COUNT).
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

    // Sensible fallback for OLMoE (16/64/8) so the tool still works if metadata
    // keys are renamed in the future.
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

    // -------- chat template
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) {
        LOG_ERR("model has no chat template - cannot build few-shot prompts\n");
        return 1;
    }

    // -------- pre-allocate per-subject count matrices
    for (const auto & subj : subjects) {
        g_acc.counts[subj].assign(g_acc.n_layer, std::vector<int64_t>(g_acc.n_expert, 0));
        g_acc.intra_counts[subj].assign(
            g_acc.n_layer, std::vector<std::vector<int64_t>>(g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0)));
        // Adjacent has n_layer - 1 entries (no L + 1 for the last layer).
        if (g_acc.n_layer >= 2) {
            g_acc.adj_counts[subj].assign(
                g_acc.n_layer - 1,
                std::vector<std::vector<int64_t>>(g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0)));
        }
        g_acc.tokens[subj] = 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const bool          add_bos = llama_vocab_get_add_bos(vocab);

    // -------- main loop: for each subject, for each test question
    int        total_questions = 0;
    const auto t_run_start     = std::chrono::steady_clock::now();

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

            // Build chat messages: system + user with few-shot examples + empty assistant
            std::vector<llama_chat_message> msgs;
            std::string                     sys_text =
                "The following are multiple choice questions (with answers).\n"
                "Answer with only the letter A, B, C, or D.";
            msgs.push_back({ "system", strdup(sys_text.c_str()) });

            std::string user_text = n_shot > 0 ? build_fewshot_user_text(dev_rows, q) : build_question_text(q);
            msgs.push_back({ "user", strdup(user_text.c_str()) });
            msgs.push_back({ "assistant", strdup("") });

            // Apply chat template
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

            // Tokenize
            std::vector<llama_token> toks = common_tokenize(ctx, prompt, add_bos, true);
            if (toks.empty()) {
                LOG_WRN("  empty tokenization for %s q%d - skipping\n", subj.c_str(), qi);
                for (auto & m : msgs) {
                    free(const_cast<char *>(m.content));
                }
                continue;
            }

            // Safety: skip prompts that exceed the model's batch limit. The
            // GGML_ASSERT in llama-context.cpp:1730 is a hard crash, not a
            // recoverable error, so we must guard here.
            const int n_batch_max = llama_n_batch(ctx);
            if ((int) toks.size() > n_batch_max) {
                LOG_WRN("  skipping %s q%d - tokenized prompt (%zu) exceeds n_batch (%d)\n", subj.c_str(), qi,
                        toks.size(), n_batch_max);
                for (auto & m : msgs) {
                    free(const_cast<char *>(m.content));
                }
                continue;
            }

            // Decode (single batch, no generation). cb_eval fires for every MoE
            // layer and updates g_acc.counts[subj] in place.
            llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("  llama_decode failed for %s q%d\n", subj.c_str(), qi);
                for (auto & m : msgs) {
                    free(const_cast<char *>(m.content));
                }
                continue;
            }

            // After every layer's ffn_moe_topk-<il> has been filled into
            // g_acc.current_topk, walk it once to tally intra-layer and
            // adjacent-layer pair counts. Done on the main thread so we
            // don't race with the scheduler callback (which fired
            // synchronously inside llama_decode above).
            tally_pairs_for_question(g_acc);

            g_acc.tokens[subj] += (int64_t) toks.size();
            total_questions++;

            for (auto & m : msgs) {
                free(const_cast<char *>(m.content));
            }

            // Clear the KV cache for this sequence so it doesn't accumulate
            // across the 2850 questions (default n_ctx = 4096 would otherwise
            // overflow after a few hundred questions).
            if (llama_get_memory(ctx) != nullptr) {
                llama_memory_clear(llama_get_memory(ctx), /*data*/ true);
            }
        }
    }

    const auto   t_run_end = std::chrono::steady_clock::now();
    const double secs      = std::chrono::duration<double>(t_run_end - t_run_start).count();

    LOG_INF("done: %d questions in %.1f s (%.1f q/s)\n", total_questions, secs, total_questions / std::max(1.0, secs));

    // -------- write JSON
    // Ensure output directory exists.
    {
        std::string dir = output_path.substr(0, output_path.find_last_of('/'));
        if (!dir.empty()) {
            // best-effort mkdir -p
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

    int64_t total_tokens = 0;
    for (const auto & kv : g_acc.tokens) {
        total_tokens += kv.second;
    }
    int subjects_with_data = 0;
    for (const auto & kv : g_acc.counts) {
        if (kv.second.empty() == false) {
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
    std::fprintf(fout, "    \"few_shot_pool\": \"cais/mmlu dev split\",\n");
    std::fprintf(fout, "    \"prompt_format\": \"few_shot_chat\"\n");
    std::fprintf(fout, "  },\n");
    std::fprintf(fout, "  \"totals\": {\n");
    std::fprintf(fout, "    \"subjects_run\": %d,\n", subjects_with_data);
    std::fprintf(fout, "    \"questions_total\": %d,\n", total_questions);
    std::fprintf(fout, "    \"tokens_total\": %lld\n", (long long) total_tokens);
    std::fprintf(fout, "  },\n");

    // ---- aggregate: sum per-subject matrices into a dataset-wide view.
    // Schema mirrors the per-subject keys but uses `marginal_expert_counts`
    // for clarity (per-subject keeps `layer_expert_counts` for backward
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
        const auto & subj_name = kv.first;
        const auto & mat       = kv.second;
        if (mat.empty()) {
            continue;
        }
        // sum marginals
        for (int L = 0; L < g_acc.n_layer; ++L) {
            for (int e = 0; e < g_acc.n_expert; ++e) {
                marginal_total[L][e] += mat[L][e];
            }
        }
        // sum intra-pair
        const auto & intra_it = g_acc.intra_counts.find(subj_name);
        if (intra_it != g_acc.intra_counts.end() && (int) intra_it->second.size() == g_acc.n_layer) {
            for (int L = 0; L < g_acc.n_layer; ++L) {
                for (int e1 = 0; e1 < g_acc.n_expert; ++e1) {
                    for (int e2 = 0; e2 < g_acc.n_expert; ++e2) {
                        intra_total[L][e1][e2] += intra_it->second[L][e1][e2];
                    }
                }
            }
        }
        // sum adjacent-pair
        const auto & adj_it = g_acc.adj_counts.find(subj_name);
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

    std::fprintf(fout, "  \"subjects\": {\n");

    bool first_subj = true;
    for (const auto & kv : g_acc.counts) {
        const std::string & subj_name    = kv.first;
        const auto &        layer_counts = kv.second;
        if (layer_counts.empty()) {
            continue;
        }

        if (!first_subj) {
            std::fprintf(fout, ",\n");
        }
        first_subj = false;

        std::fprintf(fout, "    \"%s\": {\n", json_escape(subj_name).c_str());
        std::fprintf(fout, "      \"questions\": %d,\n",
                     (int) std::min<int64_t>(questions_per_subject, (int64_t) by_subject_test[subj_name].size()));
        std::fprintf(fout, "      \"n_tokens\": %lld,\n", (long long) g_acc.tokens[subj_name]);
        std::fprintf(fout, "      \"layer_expert_counts\": ");
        write_json_2d_int_array(fout, layer_counts);

        const auto & intra_it = g_acc.intra_counts.find(subj_name);
        if (intra_it != g_acc.intra_counts.end() && (int) intra_it->second.size() == g_acc.n_layer) {
            std::fprintf(fout, ",\n      \"intra_pair_counts\": ");
            write_json_3d_int_array(fout, intra_it->second);
        }
        const auto & adj_it = g_acc.adj_counts.find(subj_name);
        if (adj_it != g_acc.adj_counts.end() && (int) adj_it->second.size() == g_acc.n_layer - 1) {
            std::fprintf(fout, ",\n      \"adjacent_pair_counts\": ");
            write_json_3d_int_array(fout, adj_it->second);
        }
        std::fprintf(fout, "\n    }");
    }

    std::fprintf(fout, "\n  }\n");
    std::fprintf(fout, "}\n");
    std::fclose(fout);

    LOG_INF("wrote %s\n", output_path.c_str());

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
