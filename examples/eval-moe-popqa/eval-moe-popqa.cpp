// llama-eval-moe-popqa
//
// Runs an MoE LLM (e.g. allenai/OLMoE-1B-7B-0125-Instruct-GGUF) over a slice
// of the PopQA dataset and records, per relation type (prop), how often each
// expert cell was activated at each MoE layer. Output is a single JSON file
// suitable for post-hoc analysis in Python.
//
// Mechanism: every MoE forward pass in llama.cpp materialises a small int32
// tensor named "ffn_moe_topk-<il>" (shape [n_expert_used, n_tokens]). It is
// already exposed through ggml_backend_sched_eval_callback. We install a custom
// callback that filters by that tensor-name regex, copies the int32 data via
// ggml_backend_tensor_get, and tallies counts into a per-prop matrix.
//
// Unlike eval-moe-mmlu, this tool also autoregressively decodes up to
// --gen-tokens after prefill and substring-matches the completion against
// `possible_answers` (the canonical self-rag metric). Routing counts include
// both prefill and generation tokens.

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

// Number of test questions to run per PopQA relation type.
static constexpr int QUESTIONS_PER_PROP = 50;

// Number of generation tokens after prefill. Set to 0 to disable generation
// entirely (prompt-only mode, useful for control experiments).
static constexpr int GEN_TOKENS        = 16;
// Default for params.embedding. See comment at params.embedding below.
// Overridable via --embeddings / --no-embeddings CLI flag.
static bool          g_embeddings_mode = false;

// Default sampling: greedy, deterministic. Matches the canonical self-rag
// evaluation protocol.
static constexpr int DEFAULT_SEED = 42;

// ------------------------------------------------------------------- globals

struct moe_accumulator {
    int n_layer    = 0;
    int n_expert   = 0;  // total experts per MoE layer
    int n_expert_k = 0;  // top-k (experts chosen per token)

    // model architecture name from general.architecture
    std::string arch;

    // current prop (relation type) under evaluation
    std::string current_prop;

    // per prop: prop -> vector<vector<int64_t>> of size [n_layer][n_expert]
    std::map<std::string, std::vector<std::vector<int64_t>>> counts;
    // per prop: prop -> number of prefill tokens (prompt encoding)
    std::map<std::string, int64_t>                           tokens_prefill;
    // per prop: prop -> number of generated tokens (autoregressive decode)
    std::map<std::string, int64_t>                           tokens_generated;
    // per prop: prop -> number of questions whose completion matched any
    // possible_answer (substring containment after normalization)
    std::map<std::string, int>                               n_correct;
    // per prop: prop -> number of questions attempted (denominator)
    std::map<std::string, int>                               n_questions;

    // per prop: prop -> vector<vector<vector<int64_t>>> [n_layer][n_expert][n_expert]
    // intra_pair_counts[subj][L][e_i][e_j] = # tokens where both e_i and e_j fired
    //                                       in the top-k of layer L (k * k slots).
    std::map<std::string, std::vector<std::vector<std::vector<int64_t>>>> intra_counts;
    // per prop: prop -> vector<vector<vector<int64_t>>> [n_layer - 1][n_expert][n_expert]
    // adj_pair_counts[subj][L][e_i][e_j] = # tokens t such that e_i fired in layer L at t
    //                                     AND e_j fired in layer L + 1 at t (k * k slots).
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

    auto & layer_counts = g_acc.counts[g_acc.current_prop][il];
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
// current prop. Each token contributes k * k increments per layer pair (one
// for each (j1, j2) top-k slot pair). Must be called after every
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
        // Dense (non-MoE) layer: the ffn_moe_topk-<il> tensor never
        // materialised, so current_topk[L] is still empty. Return -1 so
        // callers' `if (e1 < 0 || ...) continue;` filter skips it.
        // Without this, e.g. deepseek-moe-16b's leading dense layer
        // (n_layer_dense_lead=1, layer 0) would crash on an out-of-bounds
        // vector read inside this lambda.
        if (layer.empty()) {
            return -1;
        }
        return layer[(size_t) tok * stride1 + (size_t) j];
    };

    const std::string & subj  = acc.current_prop;
    auto &              intra = acc.intra_counts[subj];
    auto &              adj   = acc.adj_counts[subj];

    // Intra-layer: for each layer L, for each token t, walk all (j1, j2) top-k
    // slot pairs from the same layer's slice.
    for (int L = 0; L < n_layer; ++L) {
        if ((int) intra.size() <= L) {
            continue;  // prop was skipped (no rows)
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

// ------------------------------------------------------------------- PopQA I/O

struct popqa_row {
    int                      id = 0;
    std::string              subj;
    std::string              prop;
    std::string              obj;
    int                      s_pop = 0;
    int                      o_pop = 0;
    std::string              question;
    std::vector<std::string> possible_answers;  // 1+ entries
};

// Minimal ad-hoc JSON line parser. Each row is one object on its own line.
// Fields are extracted by name; unknown fields are ignored.
static std::vector<popqa_row> load_popqa_jsonl(const std::string & path) {
    std::vector<popqa_row> rows;
    std::ifstream          in(path);
    if (!in) {
        LOG_ERR("cannot open popqa jsonl: %s\n", path.c_str());
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

    // Reuses the MMLU `extract_choices` pattern verbatim, keyed on
    // "possible_answers" instead.
    auto extract_possible_answers = [&](const std::string & line, size_t start) -> std::vector<std::string> {
        std::vector<std::string> out;
        std::string              needle = "\"possible_answers\"";
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
        while (p < line.size()) {
            p = skip_ws(line, p);
            if (p < line.size() && line[p] == ',') {
                ++p;
                p = skip_ws(line, p);
            }
            if (p >= line.size()) {
                break;
            }
            if (line[p] == ']') {
                ++p;
                break;
            }
            if (line[p] != '"') {
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
        popqa_row r;
        r.id               = extract_int_field(line, "id");
        r.subj             = extract_string_field(line, "subj");
        r.prop             = extract_string_field(line, "prop");
        r.obj              = extract_string_field(line, "obj");
        r.s_pop            = std::max(0, extract_int_field(line, "s_pop"));
        r.o_pop            = std::max(0, extract_int_field(line, "o_pop"));
        r.question         = extract_string_field(line, "question");
        r.possible_answers = extract_possible_answers(line, 0);
        if (r.prop.empty() || r.possible_answers.empty() || r.question.empty()) {
            LOG_WRN("skipping malformed popqa row: %.80s...\n", line.c_str());
            continue;
        }
        rows.push_back(std::move(r));
    }
    return rows;
}

static std::vector<std::string> load_props(const std::string & path) {
    std::vector<std::string> out;
    std::ifstream            in(path);
    if (!in) {
        LOG_ERR("cannot open props list: %s\n", path.c_str());
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

// ------------------------------------------------------------------- match

// Normalize for substring match (self-rag style): lowercase, remove articles
// (a/an/the), remove punctuation, collapse whitespace.
static std::string normalize_text(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    bool prev_ws = true;  // collapse leading ws too
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char) s[i];
        if (std::isspace(c)) {
            if (!prev_ws) {
                out += ' ';
            }
            prev_ws = true;
            ++i;
            continue;
        }
        // Skip articles only when they stand alone as a word.
        if (!prev_ws) {
            // inside a word - keep as-is (after punctuation strip)
            ;
        }
        // Strip ASCII punctuation; keep alphanumerics and the apostrophe
        // inside the word boundary check is unnecessary for substring match.
        bool punct = std::ispunct(c) && c != '\'';
        if (punct) {
            ++i;
            continue;
        }
        out += (char) std::tolower(c);
        prev_ws = false;
        ++i;
    }
    // Trim trailing whitespace
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    // Strip articles: " a ", " an ", " the " at word boundaries.
    auto strip_article = [&](const std::string & art) {
        size_t pos = 0;
        while ((pos = out.find(" " + art + " ", pos)) != std::string::npos) {
            out.erase(pos, art.size() + 2);  // remove " art " (replace " " with " ")
            // The previous char (now space) stays; collapse with the next.
            size_t next = pos;
            while (next < out.size() && out[next] == ' ') {
                out.erase(next, 1);
            }
        }
        // leading article
        if (out.size() > art.size() && out.compare(0, art.size() + 1, art + " ") == 0) {
            out.erase(0, art.size() + 1);
        }
        // trailing article
        if (out.size() > art.size() && out.compare(out.size() - art.size() - 1, art.size() + 1, " " + art) == 0) {
            out.resize(out.size() - art.size() - 1);
        }
    };
    strip_article("a");
    strip_article("an");
    strip_article("the");
    return out;
}

// Substring containment with normalization. Returns true if any gold answer
// appears as a substring of the normalized prediction.
static bool match_score(const std::string & prediction, const std::vector<std::string> & gold) {
    if (gold.empty() || prediction.empty()) {
        return false;
    }
    std::string pred = normalize_text(prediction);
    if (pred.empty()) {
        return false;
    }
    for (const auto & g : gold) {
        std::string norm_gold = normalize_text(g);
        if (norm_gold.empty()) {
            continue;
        }
        if (pred.find(norm_gold) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Early-stop heuristics for completion strings. Stops if the model emits a
// clear sentence boundary; the substring matcher tolerates trailing prose
// anyway, but stopping early saves decode compute and yields smaller
// generation-token counts (which can be sliced later).
static bool is_sentence_end(const std::string & s) {
    if (s.empty()) {
        return false;
    }
    char last = s.back();
    // Common stop points: '\n' immediately is a strong boundary (new question
    // or chat-template artifact); a sentence-ending punctuation followed by
    // space or end is a softer stop. We only stop on newline to stay
    // conservative and avoid truncating factual answers that legitimately
    // end with "." (e.g. "U.S.").
    return last == '\n';
}

// ------------------------------------------------------------------- prompt

// Zero-shot prompt matching the canonical PopQA evaluation protocol.
static std::string build_zero_shot_prompt(const popqa_row & r) {
    std::ostringstream os;
    os << "Question: " << r.question << "\nAnswer:";
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
            "PopQA inputs (defaults shown):\n"
            "      --popqa <path>          path to popqa.jsonl     (default: build/moe-popqa/popqa.jsonl)\n"
            "      --props <path>          path to props.txt       (default: build/moe-popqa/props.txt)\n"
            "\n"
            "Run control:\n"
            "      --questions-per-prop <N>   override QUESTIONS_PER_PROP (default %d)\n"
            "      --gen-tokens <N>           autoregressive decode cap  (default %d, 0 = prompt-only)\n"
            "\n"
            "Output:\n"
            "  -o, --output <path>         output JSON (default: build/moe-popqa/expert_counts.json)\n"
            "\n"
            "Standard llama.cpp flags (from common_params_parse) are also accepted: -ngl, -c, --seed, etc.\n",
            argv[0], QUESTIONS_PER_PROP, GEN_TOKENS);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // ----- custom args: extract from argv BEFORE common_params_parse so that
    // unknown flags don't trip its strict parser.
    int         questions_per_prop = QUESTIONS_PER_PROP;
    int         gen_tokens         = GEN_TOKENS;
    std::string popqa_path         = "build/moe-popqa/popqa.jsonl";
    std::string props_path         = "build/moe-popqa/props.txt";
    std::string output_path        = "build/moe-popqa/expert_counts.json";

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
        if (a == "--popqa") {
            popqa_path = next("path");
        } else if (a == "--props") {
            props_path = next("path");
        } else if (a == "--questions-per-prop") {
            questions_per_prop = std::stoi(next("N"));
        } else if (a == "--gen-tokens") {
            gen_tokens = std::stoi(next("N"));
        } else if (a == "-o" || a == "--output") {
            output_path = next("path");
        } else if (a == "--embeddings") {
            g_embeddings_mode = true;
        } else if (a == "--no-embeddings") {
            g_embeddings_mode = false;
        }
    }

    // Second pass: build a filtered argv without our custom args.
    std::vector<char *> filtered;
    filtered.reserve(argc);
    filtered.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--popqa" || a == "--props" || a == "--questions-per-prop" || a == "--gen-tokens") {
            ++i;  // skip value too
            continue;
        }
        if (a == "-o" || a == "--output") {
            ++i;  // skip value too
            continue;
        }
        // --embeddings / --no-embeddings are no-arg boolean toggles
        // consumed by the first pass; drop them from the argv that
        // common_params_parse sees (which doesn't know about them).
        if (a == "--embeddings" || a == "--no-embeddings") {
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
    LOG_INF("loading popqa jsonl: %s\n", popqa_path.c_str());
    auto popqa_rows = load_popqa_jsonl(popqa_path);
    if (popqa_rows.empty()) {
        LOG_ERR("no popqa rows loaded - run download_popqa.py first\n");
        return 1;
    }
    LOG_INF("  loaded %zu rows\n", popqa_rows.size());

    LOG_INF("loading props list: %s\n", props_path.c_str());
    auto props = load_props(props_path);
    if (props.empty()) {
        LOG_ERR("no props loaded\n");
        return 1;
    }
    LOG_INF("  %zu props\n", props.size());

    // index rows by prop
    std::map<std::string, std::vector<const popqa_row *>> by_prop;
    for (const auto & r : popqa_rows) {
        by_prop[r.prop].push_back(&r);
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
    //
    // NB: the cb_eval hook fires for every ffn_moe_topk-<il> tensor in the
    // compute graph regardless of params.embedding. Setting embedding=true
    // is therefore NOT required to capture routing counts - it only
    // affects whether the OUTPUT tensor is materialised. On long
    // multilingual prefills (e.g. Greek/INCLUDE questions), embedding=true
    // triggers a CUDA "illegal memory access" in the MoE routing kernel
    // (observed on A100 PCIe + A100 NVLink, both llama.cpp master and
    // e5df8bfb8). Default embedding=false (set --embeddings to opt back
    // in to the old behaviour for A/B testing).
    params.cb_eval           = moe_eval_callback;
    params.cb_eval_user_data = &g_acc;
    params.warmup            = false;
    params.embedding         = g_embeddings_mode;

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
    // canonical self-rag evaluation and gives a reproducible JSON across
    // runs with the same seed.
    auto sparams         = llama_sampler_chain_default_params();
    sparams.no_perf      = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const llama_token   eos_tok = llama_vocab_eos(vocab);

    // -------- pre-allocate per-prop count matrices + counters
    for (const auto & prop : props) {
        g_acc.counts[prop].assign(g_acc.n_layer, std::vector<int64_t>(g_acc.n_expert, 0));
        g_acc.tokens_prefill[prop]   = 0;
        g_acc.tokens_generated[prop] = 0;
        g_acc.n_correct[prop]        = 0;
        g_acc.n_questions[prop]      = 0;
        g_acc.intra_counts[prop].assign(
            g_acc.n_layer, std::vector<std::vector<int64_t>>(g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0)));
        // Adjacent has n_layer - 1 entries (no L + 1 for the last layer).
        if (g_acc.n_layer >= 2) {
            g_acc.adj_counts[prop].assign(
                g_acc.n_layer - 1,
                std::vector<std::vector<int64_t>>(g_acc.n_expert, std::vector<int64_t>(g_acc.n_expert, 0)));
        }
    }

    // -------- main loop: for each prop, for each test question
    int        total_questions        = 0;
    int        total_correct          = 0;
    int64_t    total_tokens_prefill   = 0;
    int64_t    total_tokens_generated = 0;
    const auto t_run_start            = std::chrono::steady_clock::now();

    for (size_t pi = 0; pi < props.size(); ++pi) {
        const std::string & prop = props[pi];
        g_acc.current_prop       = prop;

        const auto & prop_it = by_prop.find(prop);
        if (prop_it == by_prop.end() || prop_it->second.empty()) {
            LOG_WRN("prop %s has no rows - skipping\n", prop.c_str());
            continue;
        }
        const auto & rows = prop_it->second;
        const int    n_q  = std::min<int>(questions_per_prop, (int) rows.size());

        LOG_INF("[%zu/%zu] %s - %d questions\n", pi + 1, props.size(), prop.c_str(), n_q);

        for (int qi = 0; qi < n_q; ++qi) {
            const popqa_row & r = *rows[qi];

            // Build prompt + tokenize. We deliberately bypass the chat
            // template here and use the raw "Question: ... Answer:" format
            // from the canonical PopQA protocol (Mallen et al., 2022).
            const std::string        prompt  = build_zero_shot_prompt(r);
            const bool               add_bos = llama_vocab_get_add_bos(vocab);
            std::vector<llama_token> toks    = common_tokenize(ctx, prompt, add_bos, true);
            if (toks.empty()) {
                LOG_WRN("  empty tokenization for %s q%d - skipping\n", prop.c_str(), qi);
                continue;
            }

            // Safety: skip prompts that exceed the model's batch limit. The
            // GGML_ASSERT in llama-context.cpp is a hard crash, not a
            // recoverable error, so we must guard here.
            const int n_batch_max = llama_n_batch(ctx);
            if ((int) toks.size() > n_batch_max) {
                LOG_WRN("  skipping %s q%d - tokenized prompt (%zu) exceeds n_batch (%d)\n", prop.c_str(), qi,
                        toks.size(), n_batch_max);
                continue;
            }

            // -------- prefill
            llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("  llama_decode (prefill) failed for %s q%d\n", prop.c_str(), qi);
                continue;
            }

            // After every layer's ffn_moe_topk-<il> has been filled into
            // g_acc.current_topk, walk it once to tally intra-layer and
            // adjacent-layer pair counts. Done on the main thread so we
            // don't race with the scheduler callback (which fired
            // synchronously inside llama_decode above).
            tally_pairs_for_question(g_acc);

            g_acc.tokens_prefill[prop] += (int64_t) toks.size();
            total_tokens_prefill += (int64_t) toks.size();
            g_acc.n_questions[prop] += 1;

            // -------- autoregressive decode (skipped when --gen-tokens 0)
            std::string completion;
            int         n_gen = 0;
            if (gen_tokens > 0) {
                for (int step = 0; step < gen_tokens; ++step) {
                    llama_token id = llama_sampler_sample(smpl, ctx, -1);
                    if (id == eos_tok || llama_vocab_is_eog(vocab, id)) {
                        break;
                    }

                    const std::string piece = common_token_to_piece(ctx, id, /*special=*/true);
                    completion += piece;
                    n_gen++;
                    if (is_sentence_end(completion)) {
                        break;
                    }

                    // Feed sampled token back through the model so the
                    // eval callback captures routing for this generation
                    // step too.
                    llama_batch next_batch = llama_batch_get_one(&id, 1);
                    if (llama_decode(ctx, next_batch) != 0) {
                        LOG_ERR("  llama_decode (gen step %d) failed for %s q%d\n", step, prop.c_str(), qi);
                        break;
                    }
                    // Also walk the slice buffer to tally intra/adjacent
                    // pair counts for this single generated token. Same
                    // contract as the prefill call above.
                    tally_pairs_for_question(g_acc);
                }
            }

            g_acc.tokens_generated[prop] += n_gen;
            total_tokens_generated += n_gen;

            // -------- score via substring containment
            const bool hit = match_score(completion, r.possible_answers);
            if (hit) {
                g_acc.n_correct[prop] += 1;
                total_correct++;
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
            LOG_INF("  [%d/%d] id=%d hit=%s gen=%d | %s\n", qi + 1, n_q, r.id, hit ? "yes" : "no ", n_gen, preview);

            // Clear the KV cache for this sequence so it doesn't
            // accumulate across the many questions (default n_ctx = 4096
            // would otherwise overflow after a few hundred questions).
            if (llama_get_memory(ctx) != nullptr) {
                llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
            }
        }
    }

    llama_sampler_free(smpl);

    const auto   t_run_end = std::chrono::steady_clock::now();
    const double secs      = std::chrono::duration<double>(t_run_end - t_run_start).count();

    LOG_INF("done: %d questions in %.1f s (%.1f q/s) - accuracy %.3f (%d/%d)\n", total_questions, secs,
            total_questions / std::max(1.0, secs), total_questions > 0 ? double(total_correct) / total_questions : 0.0,
            total_correct, total_questions);

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

    int props_with_data = 0;
    for (const auto & kv : g_acc.counts) {
        if (!kv.second.empty()) {
            props_with_data++;
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
    std::fprintf(fout, "    \"questions_per_prop\": %d,\n", questions_per_prop);
    std::fprintf(fout, "    \"gen_tokens\": %d,\n", gen_tokens);
    std::fprintf(fout, "    \"prompt_format\": \"zero_shot_qa\",\n");
    std::fprintf(fout, "    \"match_metric\": \"substring_normalized\"\n");
    std::fprintf(fout, "  },\n");
    std::fprintf(fout, "  \"totals\": {\n");
    std::fprintf(fout, "    \"props_run\": %d,\n", props_with_data);
    std::fprintf(fout, "    \"questions_total\": %d,\n", total_questions);
    std::fprintf(fout, "    \"tokens_total_prefill\": %lld,\n", (long long) total_tokens_prefill);
    std::fprintf(fout, "    \"tokens_total_generated\": %lld,\n", (long long) total_tokens_generated);
    std::fprintf(fout, "    \"correct\": %d,\n", total_correct);
    std::fprintf(fout, "    \"accuracy\": %.6f\n", total_questions > 0 ? double(total_correct) / total_questions : 0.0);
    std::fprintf(fout, "  },\n");

    // ---- aggregate: sum per-prop matrices into a dataset-wide view.
    // Schema mirrors the per-prop keys but uses `marginal_expert_counts`
    // for clarity (per-prop keeps `layer_expert_counts` for backward
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
        const auto & prop_name = kv.first;
        const auto & mat       = kv.second;
        if (mat.empty()) {
            continue;
        }
        for (int L = 0; L < g_acc.n_layer; ++L) {
            for (int e = 0; e < g_acc.n_expert; ++e) {
                marginal_total[L][e] += mat[L][e];
            }
        }
        const auto & intra_it = g_acc.intra_counts.find(prop_name);
        if (intra_it != g_acc.intra_counts.end() && (int) intra_it->second.size() == g_acc.n_layer) {
            for (int L = 0; L < g_acc.n_layer; ++L) {
                for (int e1 = 0; e1 < g_acc.n_expert; ++e1) {
                    for (int e2 = 0; e2 < g_acc.n_expert; ++e2) {
                        intra_total[L][e1][e2] += intra_it->second[L][e1][e2];
                    }
                }
            }
        }
        const auto & adj_it = g_acc.adj_counts.find(prop_name);
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

    std::fprintf(fout, "  \"props\": {\n");

    bool first_prop = true;
    for (const auto & kv : g_acc.counts) {
        const std::string & prop_name    = kv.first;
        const auto &        layer_counts = kv.second;
        if (layer_counts.empty()) {
            continue;
        }

        if (!first_prop) {
            std::fprintf(fout, ",\n");
        }
        first_prop = false;

        const int    n_q = g_acc.n_questions[prop_name];
        const int    n_c = g_acc.n_correct[prop_name];
        const double mr  = n_q > 0 ? double(n_c) / n_q : 0.0;

        std::fprintf(fout, "    \"%s\": {\n", json_escape(prop_name).c_str());
        std::fprintf(fout, "      \"questions\": %d,\n", n_q);
        std::fprintf(fout, "      \"n_tokens_prefill\": %lld,\n", (long long) g_acc.tokens_prefill[prop_name]);
        std::fprintf(fout, "      \"n_tokens_generated\": %lld,\n", (long long) g_acc.tokens_generated[prop_name]);
        std::fprintf(fout, "      \"n_correct\": %d,\n", n_c);
        std::fprintf(fout, "      \"match_rate\": %.6f,\n", mr);
        std::fprintf(fout, "      \"layer_expert_counts\": ");
        write_json_2d_int_array(fout, layer_counts);

        const auto & intra_it = g_acc.intra_counts.find(prop_name);
        if (intra_it != g_acc.intra_counts.end() && (int) intra_it->second.size() == g_acc.n_layer) {
            std::fprintf(fout, ",\n      \"intra_pair_counts\": ");
            write_json_3d_int_array(fout, intra_it->second);
        }
        const auto & adj_it = g_acc.adj_counts.find(prop_name);
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
