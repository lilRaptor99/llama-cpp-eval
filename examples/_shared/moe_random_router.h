#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <numeric>
#include <random>
#include <string_view>
#include <vector>

static inline uint64_t moe_random_splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t moe_random_hash_string(uint64_t seed, std::string_view s) {
    uint64_t h = moe_random_splitmix64(seed ^ (uint64_t) s.size());
    for (unsigned char c : s) {
        h ^= (uint64_t) c + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h = moe_random_splitmix64(h);
    }
    return h;
}

static inline uint64_t moe_random_seed_for_subject(uint64_t base_seed,
                                                   std::string_view model,
                                                   std::string_view arch,
                                                   std::string_view subject) {
    uint64_t h = moe_random_splitmix64(base_seed);
    h = moe_random_hash_string(h, model);
    h = moe_random_hash_string(h, arch);
    h = moe_random_hash_string(h, subject);
    return moe_random_splitmix64(h);
}

struct moe_random_router_state {
    bool                 enabled   = false;
    uint64_t             base_seed = 0;
    int                  n_expert  = 0;
    std::vector<int32_t> expert_ids;
    std::mt19937_64      rng;

    void configure(int n_expert_, uint64_t base_seed_) {
        n_expert  = std::max(0, n_expert_);
        base_seed = base_seed_;
        expert_ids.resize((size_t) n_expert);
        std::iota(expert_ids.begin(), expert_ids.end(), 0);
    }

    void seed_subject(std::string_view model, std::string_view arch, std::string_view subject) {
        rng.seed(moe_random_seed_for_subject(base_seed, model, arch, subject));
    }

    void fill_topk_buffer(int n_tokens, int n_expert_k, int32_t * out) {
        if (!enabled || out == nullptr || n_tokens <= 0 || n_expert <= 0) {
            return;
        }

        const int k = std::clamp(n_expert_k, 0, n_expert);
        const size_t stride1 = (size_t) n_expert;
        for (int tok = 0; tok < n_tokens; ++tok) {
            int32_t * row = out + (size_t) tok * stride1;
            std::fill(row, row + stride1, 0);
            if (k == 0) {
                continue;
            }
            if (k >= n_expert) {
                std::copy(expert_ids.begin(), expert_ids.end(), row);
                continue;
            }
            std::sample(expert_ids.begin(), expert_ids.end(), row, k, rng);
        }
    }
};