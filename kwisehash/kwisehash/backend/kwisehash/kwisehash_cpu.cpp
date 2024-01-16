#include "kwisehash/kwisehash_cpu.h"

#include <torch/torch.h>
#include <omp.h>

#define CHECK_CPU(x) TORCH_CHECK(x.device() == torch::kCPU, #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK(x.dtype() == torch::kLong, #x " must be long tensor")
#define CHECK_INPUT(x)   \
    CHECK_CPU(x);        \
    CHECK_CONTIGUOUS(x); \
    CHECK_LONG(x)

namespace ti = torch::indexing;

int64_t prime() { return MERSENNE_PRIME; }

torch::Tensor random_seeds_cpu(int64_t num_hashes, int64_t k)
{
    auto options =
        torch::TensorOptions()
            .dtype(torch::kLong)
            .layout(torch::kStrided)
            .device(torch::kCPU)
            .requires_grad(false);

    // The coefficients cannot be zero
    torch::Tensor a = torch::randint(1, MERSENNE_PRIME, {num_hashes, k - 1}, options);
    // # Only the constant can be zero
    torch::Tensor b = torch::randint(0, MERSENNE_PRIME, {num_hashes, 1}, options);

    return torch::cat({a, b}, 1);
}

/// @brief Evaluates the polynomial hash function
/// @param input
/// @param seed_ptr
/// @param k independence of the hash function
/// @return Hashed input
inline __attribute__((flatten)) int64_t poly_call(
    const int64_t input,
    const int64_t *__restrict__ const seed_ptr,
    const int64_t k)
{
    int64_t out = seed_ptr[0];

    for (int64_t i = 1; i < k; i++)
    {
        const __int128_t tmp0 = (__int128_t)out * input + seed_ptr[i];
        const int64_t tmp1 = ((int64_t)tmp0 & MERSENNE_PRIME) + (int64_t)(tmp0 >> 61);
        const int64_t tmp2 = tmp1 - MERSENNE_PRIME;
        out = tmp2 < 0 ? tmp1 : tmp2;
    }

    return out;
}

inline void permute_hash_cpu_impl(
    const int64_t *__restrict__ const input_ptr,
    const int64_t *__restrict__ const seed_ptr,
    const int64_t k,
    const int64_t n,
    const int64_t m,
    int64_t *__restrict__ const out_ptr)
{

#pragma omp parallel for if (n > GRAIN_SIZE) shared(input_ptr, seed_ptr, out_ptr)
    for (int64_t i = 0; i < n; i++)
    {
        for (int64_t j = 0; j < m; j++)
        {
            int64_t out = poly_call(input_ptr[i], seed_ptr + j * k, k);
            out_ptr[i + j * n] = out;
        }
    }
}

inline void sign_hash_cpu_impl(
    const int64_t *__restrict__ const input_ptr,
    const int64_t *__restrict__ const seed_ptr,
    const int64_t k,
    const int64_t n,
    const int64_t m,
    int64_t *__restrict__ const out_ptr)
{

#pragma omp parallel for if (n > GRAIN_SIZE) shared(input_ptr, seed_ptr, out_ptr)
    for (int64_t i = 0; i < n; i++)
    {
        for (int64_t j = 0; j < m; j++)
        {
            int64_t out = poly_call(input_ptr[i], seed_ptr + j * k, k);
            out = (out & 1) * 2 - 1;
            out_ptr[i + j * n] = out;
        }
    }
}

inline void bin_hash_cpu_impl(
    const int64_t *__restrict__ const input_ptr,
    const int64_t *__restrict__ const seed_ptr,
    const int64_t k,
    const int64_t n,
    const int64_t m,
    const int64_t b,
    int64_t *__restrict__ const out_ptr)
{

#pragma omp parallel for if (n > GRAIN_SIZE) shared(input_ptr, seed_ptr, out_ptr)
    for (int64_t i = 0; i < n; i++)
    {
        for (int64_t j = 0; j < m; j++)
        {
            int64_t out = poly_call(input_ptr[i], seed_ptr + j * k, k);
            out = out % b;
            out_ptr[i + j * n] = out;
        }
    }
}

inline void bin_hash_pow2_cpu_impl(
    const int64_t *__restrict__ const input_ptr,
    const int64_t *__restrict__ const seed_ptr,
    const int64_t k,
    const int64_t n,
    const int64_t m,
    const int64_t b,
    int64_t *__restrict__ const out_ptr)
{

#pragma omp parallel for if (n > GRAIN_SIZE) shared(input_ptr, seed_ptr, out_ptr)
    for (int64_t i = 0; i < n; i++)
    {
        for (int64_t j = 0; j < m; j++)
        {
            int64_t out = poly_call(input_ptr[i], seed_ptr + j * k, k);
            out = out & (b - 1);
            out_ptr[i + j * n] = out;
        }
    }
}

torch::Tensor permute_hash_cpu(torch::Tensor input, torch::Tensor seeds)
{
    // Ensure that the data is ordered as expected
    input = input.contiguous();
    seeds = seeds.contiguous();

    CHECK_INPUT(input);
    CHECK_INPUT(seeds);

    const int64_t n = input.size(0); // number of items
    const int64_t m = seeds.size(0); // number of hash functions
    const int64_t k = seeds.size(1); // independence of the hash functions

    auto options =
        torch::TensorOptions()
            .dtype(torch::kLong)
            .layout(torch::kStrided)
            .device(torch::kCPU)
            .requires_grad(false);

    torch::Tensor output = torch::empty({m, n}, options);

    int64_t *input_ptr = input.data_ptr<int64_t>();
    int64_t *seeds_ptr = seeds.data_ptr<int64_t>();
    int64_t *out_ptr = output.data_ptr<int64_t>();

    permute_hash_cpu_impl(input_ptr, seeds_ptr, k, n, m, out_ptr);

    return output;
}

torch::Tensor sign_hash_cpu(torch::Tensor input, torch::Tensor seeds)
{
    // Ensure that the data is ordered as expected
    input = input.contiguous();
    seeds = seeds.contiguous();

    CHECK_INPUT(input);
    CHECK_INPUT(seeds);

    const int64_t n = input.size(0); // number of items
    const int64_t m = seeds.size(0); // number of hash functions
    const int64_t k = seeds.size(1); // independence of the hash functions

    auto options =
        torch::TensorOptions()
            .dtype(torch::kLong)
            .layout(torch::kStrided)
            .device(torch::kCPU)
            .requires_grad(false);

    torch::Tensor output = torch::empty({m, n}, options);

    int64_t *input_ptr = input.data_ptr<int64_t>();
    int64_t *seeds_ptr = seeds.data_ptr<int64_t>();
    int64_t *out_ptr = output.data_ptr<int64_t>();

    sign_hash_cpu_impl(input_ptr, seeds_ptr, k, n, m, out_ptr);

    return output;
}

torch::Tensor bin_hash_cpu(torch::Tensor input, torch::Tensor seeds, int64_t num_bins)
{
    // Ensure that the data is ordered as expected
    input = input.contiguous();
    seeds = seeds.contiguous();

    CHECK_INPUT(input);
    CHECK_INPUT(seeds);

    const int64_t n = input.size(0); // number of items
    const int64_t m = seeds.size(0); // number of hash functions
    const int64_t k = seeds.size(1); // independence of the hash functions

    auto options =
        torch::TensorOptions()
            .dtype(torch::kLong)
            .layout(torch::kStrided)
            .device(torch::kCPU)
            .requires_grad(false);

    torch::Tensor output = torch::empty({m, n}, options);

    int64_t *input_ptr = input.data_ptr<int64_t>();
    int64_t *seeds_ptr = seeds.data_ptr<int64_t>();
    int64_t *out_ptr = output.data_ptr<int64_t>();

    bool is_pow_two = (num_bins & (num_bins - 1)) == 0;

    if (is_pow_two)
    {
        bin_hash_pow2_cpu_impl(input_ptr, seeds_ptr, k, n, m, num_bins, out_ptr);
    }
    else
    {
        bin_hash_cpu_impl(input_ptr, seeds_ptr, k, n, m, num_bins, out_ptr);
    }

    return output;
}
