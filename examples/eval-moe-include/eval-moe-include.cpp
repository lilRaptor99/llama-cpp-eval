// llama-eval-moe-include
//
// Runs an MoE LLM (e.g. allenai/OLMoE-1B-7B-0125-Instruct-GGUF) over a slice
// of the CohereLabs/include-base-44 dataset and records, per (language,
// domain) bucket, how often each expert cell was activated at each MoE
// layer. Output is a single JSON file suitable for post-hoc analysis in
// Python (expert-usage heatmaps, per-language / per-domain / per-(lang,dom)
// aggregations, accuracy bars).
//
// Mechanism: every MoE forward pass in llama.cpp materialises a small int32
// tensor named "ffn_moe_topk-<il>" (shape [n_expert_used, n_tokens]). It is
// already exposed through ggml_backend_sched_eval_callback. We install a
// custom callback that filters by that tensor-name regex, copies the int32
// data via ggml_backend_tensor_get, and tallies counts into a per-(lang,dom)
// matrix.
//
// Partitioning: the primary routing bucket is "<language>::<domain>"
// (~484 keys for the full 44-language set). The Python plotter aggregates
// to per-language (~44) and per-domain (~11) views on the fly. 5-shot
// in-language exemplars are drawn from each (lang, dom) bucket; the first 5
// rows (after sorting by question text) serve as exemplars, the remainder
// up to questions_per_langdom are tested. Answer matching: argmax
// restricted to {A,B,C,D} over the next-token logits, with an autoregressive
// --gen-tokens pass (default 16) to capture routing on the chosen letter +
// any reasoning tokens.
//
// Like eval-moe-popqa, this tool also **autoregressively decodes** up to
// --gen-tokens after prefill. The completion is substring-matched (after
// normalisation) against the gold choice letter. Routing counts include
// both prefill and generation tokens.
//
// It currently targets `olmoe`. Other archs (mixtral, qwen2moe, qwen3moe,
// deepseek2, gpt-oss) are discovered through their respective metadata
// keys; the top-k tensor name is the same, so routing counts will work for
// them too.

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

// Number of test questions to run per (language, domain) group, AFTER the
// five exemplars are taken from the same bucket. With ~484 langdom groups
// and the paper's 5-shot protocol the default is intentionally small.
static constexpr int QUESTIONS_PER_LANGDOM = 5;

// Number of few-shot exemplars drawn from each (lang, dom) test bucket.
// The paper's canonical protocol (Romanou et al. 2024 §4.2) is 5-shot;
// set --n-shots 0 for 0-shot (matches eval-moe-popqa default).
static constexpr int N_SHOT = 5;

// Number of generation tokens after prefill. Set to 0 to disable generation
// entirely (prompt-only mode, useful for control experiments). The default
// of 16 mirrors eval-moe-popqa and is enough headroom for the chosen letter
// plus a small amount of reasoning / trailing prose.
static constexpr int GEN_TOKENS = 16;

// Default sampling: greedy, deterministic. Matches the canonical self-rag
// / Harness evaluation protocol.
static constexpr int DEFAULT_SEED = 42;

// ------------------------------------------------------------------- globals

struct moe_accumulator {
    int n_layer    = 0;
    int n_expert   = 0;  // total experts per MoE layer
    int n_expert_k = 0;  // top-k (experts chosen per token)

    // model architecture name from general.architecture
    std::string arch;

    // current <language>::<domain> bucket under evaluation
    std::string current_langdom;

    // per (lang, dom): langdom -> vector<vector<int64_t>> of size [n_layer][n_expert]
    std::map<std::string, std::vector<std::vector<int64_t>>> counts;
    // per (lang, dom): langdom -> number of prefill tokens (prompt encoding)
    std::map<std::string, int64_t>                           tokens_prefill;
    // per (lang, dom): langdom -> number of generated tokens (autoregressive decode)
    std::map<std::string, int64_t>                           tokens_generated;
    // per (lang, dom): langdom -> number of questions whose completion
    // matched the gold choice letter (substring containment after
    // normalisation against the first non-empty line)
    std::map<std::string, int>                               n_correct;
    // per (lang, dom): langdom -> number of questions attempted (denominator)
    std::map<std::string, int>                               n_questions;

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

    auto & layer_counts = g_acc.counts[g_acc.current_langdom][il];
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

    return true;
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

// ------------------------------------------------------------------- INCLUDE I/O

struct include_row {
    int                      answer = 0;  // 0..3  (0=A, 1=B, 2=C, 3=D)
    std::string              split;       // "test" or "validation"
    std::string              language;    // config name (e.g. "French")
    std::string              country;
    std::string              domain;
    std::string              subject;
    std::string              regional_feature;
    std::string              level;
    std::string              question;
    std::vector<std::string> options;  // 4 entries, letter-prefixed
};

// Minimal ad-hoc JSON line parser. Each row is one object on its own line.
// Fields are extracted by name; unknown fields are ignored.
static std::vector<include_row> load_include_jsonl(const std::string & path) {
    std::vector<include_row> rows;
    std::ifstream            in(path);
    if (!in) {
        LOG_ERR("cannot open include jsonl: %s\n", path.c_str());
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

    // Same options pattern as eval-moe-popqa.cpp / eval-moe-bigbench.cpp:
    // walk the 4-element string array verbatim. Returns 4 entries or
    // empty on parse failure.
    auto extract_options = [&](const std::string & line, size_t start) -> std::vector<std::string> {
        std::vector<std::string> out;
        std::string              needle = "\"options\"";
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
        include_row r;
        r.split            = extract_string_field(line, "split");
        r.language         = extract_string_field(line, "language");
        r.country          = extract_string_field(line, "country");
        r.domain           = extract_string_field(line, "domain");
        r.subject          = extract_string_field(line, "subject");
        r.regional_feature = extract_string_field(line, "regional_feature");
        r.level            = extract_string_field(line, "level");
        r.question         = extract_string_field(line, "question");
        r.options          = extract_options(line, 0);
        r.answer           = extract_int_field(line, "answer");
        if (r.language.empty() || r.domain.empty() || r.question.empty() || r.options.size() != 4 || r.answer < 0 ||
            r.answer > 3) {
            LOG_WRN("skipping malformed include row: %.80s...\n", line.c_str());
            continue;
        }
        rows.push_back(std::move(r));
    }
    return rows;
}

static std::vector<std::string> load_keys(const std::string & path) {
    std::vector<std::string> out;
    std::ifstream            in(path);
    if (!in) {
        LOG_ERR("cannot open keys list: %s\n", path.c_str());
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

static std::string langdom_key(const std::string & lang, const std::string & dom) {
    return lang + "::" + dom;
}

// Split "<language>::<domain>" into (language, domain). Returns false if
// the separator isn't found.
static bool split_langdom(const std::string & key, std::string & lang, std::string & dom) {
    size_t p = key.find("::");
    if (p == std::string::npos) {
        return false;
    }
    lang = key.substr(0, p);
    dom  = key.substr(p + 2);
    return true;
}

// ------------------------------------------------------------------- scoring

// Normalize for substring match (self-rag style): lowercase, remove
// articles (a/an/the), remove punctuation, collapse whitespace.
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
        // Strip ASCII punctuation; keep alphanumerics and the apostrophe.
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
            out.erase(pos, art.size() + 2);
            size_t next = pos;
            while (next < out.size() && out[next] == ' ') {
                out.erase(next, 1);
            }
        }
        if (out.size() > art.size() && out.compare(0, art.size() + 1, art + " ") == 0) {
            out.erase(0, art.size() + 1);
        }
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

// Take the first non-empty line of a generated completion. Choice letters
// typically arrive on the first line; this lets us tolerate trailing prose.
static std::string first_non_empty_line(const std::string & s) {
    std::string current;
    for (char c : s) {
        if (c == '\n') {
            std::string t = current;
            while (!t.empty() && std::isspace((unsigned char) t.back())) {
                t.pop_back();
            }
            size_t p = 0;
            while (p < t.size() && std::isspace((unsigned char) t[p])) {
                ++p;
            }
            if (p < t.size()) {
                return t.substr(p);
            }
            current.clear();
        } else {
            current += c;
        }
    }
    std::string t = current;
    while (!t.empty() && std::isspace((unsigned char) t.back())) {
        t.pop_back();
    }
    size_t p = 0;
    while (p < t.size() && std::isspace((unsigned char) t[p])) {
        ++p;
    }
    return p < t.size() ? t.substr(p) : std::string{};
}

// ------------------------------------------------------------------- prompt

// 5-shot in-language prompt matching the canonical INCLUDE / Harness
// protocol (Romanou et al. 2024 §4.2, EleutherAI Harness
// lm_eval/tasks/include/default/<Lang>/_template_yaml). Header is the
// per-row `domain`, exemplars are the first N_SHOT rows of the (lang, dom)
// bucket after sorting by question text.
static std::string build_fewshot_user_text(const std::vector<const include_row *> & exemplars,
                                           const include_row &                      test_row) {
    std::ostringstream os;
    for (int i = 0; i < (int) exemplars.size(); ++i) {
        const auto & d = *exemplars[i];
        os << d.question << "\n";
        for (const auto & c : d.options) {
            os << c << "\n";
        }
        os << "Answer: " << char('A' + d.answer) << "\n\n";
    }
    os << test_row.question << "\n";
    for (const auto & c : test_row.options) {
        os << c << "\n";
    }
    os << "Answer:";
    return os.str();
}

// Zero-shot prompt (used when --n-shots 0).
static std::string build_zero_shot_user_text(const include_row & test_row) {
    std::ostringstream os;
    os << test_row.question << "\n";
    for (const auto & c : test_row.options) {
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
            "INCLUDE inputs (defaults shown):\n"
            "      --include <path>        path to include.jsonl        (default: build/moe-include/include.jsonl)\n"
            "      --langdom-list <path>   path to languages_domains.txt (default: "
            "build/moe-include/languages_domains.txt)\n"
            "\n"
            "Run control:\n"
            "      --questions-per-langdom <N>   override QUESTIONS_PER_LANGDOM (default %d)\n"
            "      --n-shots <N>                 override N_SHOT             (default %d, 0 = zero-shot)\n"
            "      --gen-tokens <N>              autoregressive decode cap   (default %d, 0 = prompt-only)\n"
            "\n"
            "Output:\n"
            "  -o, --output <path>         output JSON (default: build/moe-include/expert_counts.json)\n"
            "\n"
            "Standard llama.cpp flags (from common_params_parse) are also accepted: -ngl, -c, --seed, etc.\n",
            argv[0], QUESTIONS_PER_LANGDOM, N_SHOT, GEN_TOKENS);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // ----- custom args: extract from argv BEFORE common_params_parse so that
    // unknown flags don't trip its strict parser.
    int         questions_per_langdom = QUESTIONS_PER_LANGDOM;
    int         n_shot                = N_SHOT;
    int         gen_tokens            = GEN_TOKENS;
    std::string include_path          = "build/moe-include/include.jsonl";
    std::string langdom_list_path     = "build/moe-include/languages_domains.txt";
    std::string output_path           = "build/moe-include/expert_counts.json";

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
        if (a == "--include") {
            include_path = next("path");
        } else if (a == "--langdom-list") {
            langdom_list_path = next("path");
        } else if (a == "--questions-per-langdom") {
            questions_per_langdom = std::stoi(next("N"));
        } else if (a == "--n-shots") {
            n_shot = std::stoi(next("N"));
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
        if (a == "--include" || a == "--langdom-list" || a == "--questions-per-langdom" || a == "--n-shots" ||
            a == "--gen-tokens") {
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
    // Default n_batch=2048 can be too small for some 5-shot prompts; bump it.
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
    LOG_INF("loading include jsonl: %s\n", include_path.c_str());
    auto include_rows = load_include_jsonl(include_path);
    if (include_rows.empty()) {
        LOG_ERR("no include rows loaded - run download_include.py first\n");
        return 1;
    }
    LOG_INF("  loaded %zu rows\n", include_rows.size());

    LOG_INF("loading langdom list: %s\n", langdom_list_path.c_str());
    auto langdom_list = load_keys(langdom_list_path);
    if (langdom_list.empty()) {
        LOG_ERR("no langdom keys loaded\n");
        return 1;
    }
    LOG_INF("  %zu (language, domain) keys\n", langdom_list.size());

    // index rows by (language, domain) bucket. We use the config-language
    // as the bucket key (already trustworthy from the downloader), not the
    // row's `language` field which can be mislabelled upstream.
    std::map<std::string, std::vector<const include_row *>> by_langdom;
    for (const auto & r : include_rows) {
        const std::string key = langdom_key(r.language, r.domain);
        by_langdom[key].push_back(&r);
    }

    // sort each bucket by question text so exemplar selection is stable.
    for (auto & kv : by_langdom) {
        std::sort(kv.second.begin(), kv.second.end(),
                  [](const include_row * a, const include_row * b) { return a->question < b->question; });
    }

    // -------- init llama
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // Wire the eval callback BEFORE context creation (params.cb_eval is
    // consumed when common_init_from_params builds the context). With
    // params.embedding=true the cparams.embeddings flag forces
    // output_all=true on every llama_decode, so the prefill routes all
    // tokens through every MoE layer.
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
    // canonical self-rag / Harness evaluation and gives a reproducible
    // JSON across runs with the same seed.
    auto sparams         = llama_sampler_chain_default_params();
    sparams.no_perf      = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const llama_token   eos_tok = llama_vocab_eos(vocab);

    // -------- pre-allocate per-(lang, dom) count matrices + counters
    for (const auto & key : langdom_list) {
        g_acc.counts[key].assign(g_acc.n_layer, std::vector<int64_t>(g_acc.n_expert, 0));
        g_acc.tokens_prefill[key]   = 0;
        g_acc.tokens_generated[key] = 0;
        g_acc.n_correct[key]        = 0;
        g_acc.n_questions[key]      = 0;
    }

    // -------- main loop: for each (lang, dom) bucket, build exemplars then
    // run N test questions.
    int        total_questions        = 0;
    int        total_correct          = 0;
    int64_t    total_tokens_prefill   = 0;
    int64_t    total_tokens_generated = 0;
    const auto t_run_start            = std::chrono::steady_clock::now();

    const std::vector<std::string> choice_letters = { "A", "B", "C", "D" };

    int li = 0;
    for (const auto & key : langdom_list) {
        std::string language, domain;
        if (!split_langdom(key, language, domain)) {
            LOG_WRN("skipping malformed langdom key: %s\n", key.c_str());
            continue;
        }
        g_acc.current_langdom = key;

        const auto & bucket_it = by_langdom.find(key);
        if (bucket_it == by_langdom.end() || bucket_it->second.empty()) {
            LOG_WRN("langdom %s has no rows - skipping\n", key.c_str());
            continue;
        }
        const auto & rows        = bucket_it->second;
        const int    n_total     = (int) rows.size();
        const int    n_exemplars = std::min<int>(n_shot, n_total);
        // Reserve exemplars at the front; tests run from index n_exemplars
        // onward. questions_per_langdom caps the number of test questions
        // we ATTEMPT, not the available rows.
        const int    n_q         = std::min<int>(questions_per_langdom, std::max(0, n_total - n_exemplars));

        std::vector<const include_row *> exemplars;
        exemplars.reserve(n_exemplars);
        for (int j = 0; j < n_exemplars; ++j) {
            exemplars.push_back(rows[j]);
        }

        LOG_INF("[%zu/%zu] %s - %d exemplars + %d/%d test questions\n", (size_t) li + 1, langdom_list.size(),
                key.c_str(), n_exemplars, n_q, n_total - n_exemplars);
        ++li;

        const std::string header =
            "The following are multiple choice questions (with answers) about " + domain + ".\n\n";

        for (int qi = 0; qi < n_q; ++qi) {
            const include_row & r = *rows[n_exemplars + qi];

            // Build prompt + tokenize. Few-shot exemplars are drawn from
            // the same (lang, dom) bucket — matches the canonical
            // Harness / paper protocol.
            std::string prompt;
            if (n_exemplars > 0) {
                prompt = header + build_fewshot_user_text(exemplars, r);
            } else {
                prompt = build_zero_shot_user_text(r);
            }
            const bool               add_bos = llama_vocab_get_add_bos(vocab);
            std::vector<llama_token> toks    = common_tokenize(ctx, prompt, add_bos, true);
            if (toks.empty()) {
                LOG_WRN("  empty tokenization for %s q%d - skipping\n", key.c_str(), qi);
                continue;
            }

            // Safety: skip prompts that exceed the model's batch limit. The
            // GGML_ASSERT in llama-context.cpp is a hard crash, not a
            // recoverable error, so we must guard here.
            const int n_batch_max = llama_n_batch(ctx);
            if ((int) toks.size() > n_batch_max) {
                LOG_WRN("  skipping %s q%d - tokenized prompt (%zu) exceeds n_batch (%d)\n", key.c_str(), qi,
                        toks.size(), n_batch_max);
                continue;
            }

            // -------- prefill
            llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("  llama_decode (prefill) failed for %s q%d\n", key.c_str(), qi);
                continue;
            }

            g_acc.tokens_prefill[key] += (int64_t) toks.size();
            total_tokens_prefill += (int64_t) toks.size();
            g_acc.n_questions[key] += 1;

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

                    // Feed sampled token back through the model so the
                    // eval callback captures routing for this generation
                    // step too.
                    llama_batch next_batch = llama_batch_get_one(&id, 1);
                    if (llama_decode(ctx, next_batch) != 0) {
                        LOG_ERR("  llama_decode (gen step %d) failed for %s q%d\n", step, key.c_str(), qi);
                        break;
                    }
                }
            }

            g_acc.tokens_generated[key] += n_gen;
            total_tokens_generated += n_gen;

            // -------- score via substring containment against the gold choice letter(s)
            const std::vector<std::string> gold_letters = { std::string(1, char('A' + r.answer)) };
            const std::string              prediction   = first_non_empty_line(completion);
            const bool                     hit          = match_score(prediction, gold_letters);
            if (hit) {
                g_acc.n_correct[key] += 1;
                total_correct++;
            }
            total_questions++;

            // -------- logging
            char preview[200];
            std::strncpy(preview, completion.c_str(), sizeof(preview) - 1);
            preview[sizeof(preview) - 1] = '\0';
            for (size_t j = 0; j < strlen(preview); ++j) {
                if (preview[j] == '\n') {
                    preview[j] = ' ';
                }
            }
            LOG_INF("  [%d/%d] hit=%s (gold=%s) gen=%d | %s\n", qi + 1, n_q, hit ? "yes" : "no ",
                    gold_letters[0].c_str(), n_gen, preview);

            // Clear the KV cache for this sequence so it doesn't accumulate
            // across the many questions (default n_ctx = 4096 would
            // otherwise overflow after a few hundred questions).
            if (llama_get_memory(ctx) != nullptr) {
                llama_memory_clear(llama_get_memory(ctx), /*data=*/true);
            }
        }
    }

    llama_sampler_free(smpl);

    const auto   t_run_end = std::chrono::steady_clock::now();
    const double secs      = std::chrono::duration<double>(t_run_end - t_run_start).count();

    LOG_INF("done: %d questions in %.1f s (%.1f q/s) - accuracy %.3f (%d/%d)\n", total_questions, secs,
            total_questions > 0 ? double(total_questions) / std::max(1.0, secs) : 0.0,
            total_questions > 0 ? double(total_correct) / total_questions : 0.0, total_correct, total_questions);

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

    int                   langdoms_with_data  = 0;
    int                   languages_with_data = 0;
    std::set<std::string> seen_langs;
    for (const auto & kv : g_acc.counts) {
        if (!kv.second.empty()) {
            ++langdoms_with_data;
            std::string l, d;
            if (split_langdom(kv.first, l, d)) {
                seen_langs.insert(l);
            }
        }
    }
    languages_with_data = (int) seen_langs.size();

    std::fprintf(fout, "{\n");
    std::fprintf(fout, "  \"model\": \"%s\",\n", json_escape(params.model.get_name()).c_str());
    std::fprintf(fout, "  \"model_arch\": {\n");
    std::fprintf(fout, "    \"name\": \"%s\",\n", json_escape(g_acc.arch).c_str());
    std::fprintf(fout, "    \"n_layer\": %d,\n", g_acc.n_layer);
    std::fprintf(fout, "    \"n_expert\": %d,\n", g_acc.n_expert);
    std::fprintf(fout, "    \"n_expert_used\": %d\n", g_acc.n_expert_k);
    std::fprintf(fout, "  },\n");
    std::fprintf(fout, "  \"config\": {\n");
    std::fprintf(fout, "    \"questions_per_langdom\": %d,\n", questions_per_langdom);
    std::fprintf(fout, "    \"n_shot\": %d,\n", n_shot);
    std::fprintf(fout, "    \"gen_tokens\": %d,\n", gen_tokens);
    std::fprintf(fout, "    \"prompt_format\": \"few_shot_inlang_5shot\",\n");
    std::fprintf(fout, "    \"match_metric\": \"substring_normalized_first_letter\"\n");
    std::fprintf(fout, "  },\n");
    std::fprintf(fout, "  \"dataset\": \"CohereLabs/include-base-44\",\n");
    std::fprintf(fout, "  \"totals\": {\n");
    std::fprintf(fout, "    \"langdoms_run\": %d,\n", langdoms_with_data);
    std::fprintf(fout, "    \"languages_run\": %d,\n", languages_with_data);
    std::fprintf(fout, "    \"questions_total\": %d,\n", total_questions);
    std::fprintf(fout, "    \"tokens_total_prefill\": %lld,\n", (long long) total_tokens_prefill);
    std::fprintf(fout, "    \"tokens_total_generated\": %lld,\n", (long long) total_tokens_generated);
    std::fprintf(fout, "    \"correct\": %d,\n", total_correct);
    std::fprintf(fout, "    \"accuracy\": %.6f\n", total_questions > 0 ? double(total_correct) / total_questions : 0.0);
    std::fprintf(fout, "  },\n");
    std::fprintf(fout, "  \"by_langdom\": {\n");

    bool first = true;
    for (const auto & kv : g_acc.counts) {
        const std::string & key          = kv.first;
        const auto &        layer_counts = kv.second;
        if (layer_counts.empty()) {
            continue;
        }

        if (!first) {
            std::fprintf(fout, ",\n");
        }
        first = false;

        std::string lang, dom;
        if (!split_langdom(key, lang, dom)) {
            lang = key;
            dom  = "";
        }

        const int    n_q = g_acc.n_questions[key];
        const int    n_c = g_acc.n_correct[key];
        const double mr  = n_q > 0 ? double(n_c) / n_q : 0.0;

        std::fprintf(fout, "    \"%s\": {\n", json_escape(key).c_str());
        std::fprintf(fout, "      \"language\": \"%s\",\n", json_escape(lang).c_str());
        std::fprintf(fout, "      \"domain\": \"%s\",\n", json_escape(dom).c_str());
        std::fprintf(fout, "      \"questions\": %d,\n", n_q);
        std::fprintf(fout, "      \"n_tokens_prefill\": %lld,\n", (long long) g_acc.tokens_prefill[key]);
        std::fprintf(fout, "      \"n_tokens_generated\": %lld,\n", (long long) g_acc.tokens_generated[key]);
        std::fprintf(fout, "      \"n_correct\": %d,\n", n_c);
        std::fprintf(fout, "      \"match_rate\": %.6f,\n", mr);
        std::fprintf(fout, "      \"layer_expert_counts\": ");
        write_json_2d_int_array(fout, layer_counts);
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
