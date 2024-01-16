#include "kwisehash/kwisehash_cuda.h"

#include <torch/torch.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK(x.dtype() == torch::kLong, #x " must be long tensor")
#define CHECK_INPUT(x)   \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x); \
    CHECK_LONG(x);

/** CUDA naive thread block size. */
#define BLOCK_SIZE (256)

/// @brief Evaluates the polynomial hash function
/// @param input
/// @param seed_ptr
/// @param k independence of the hash function
/// @return Hashed input
__device__ __forceinline__ int64_t poly_call(
    const int64_t input, 
    const int64_t *__restrict__ const seed_ptr, 
    const int64_t k)
{
    int64_t out = seed_ptr[0];

    for (int64_t j = 1; j < k; j += 1)
    {
        const __int128_t tmp0 = (__int128_t)out * input + seed_ptr[j];
        const int64_t tmp1 = ((int64_t)tmp0 & MERSENNE_PRIME) + (int64_t)(tmp0 >> 61);
        const int64_t tmp2 = tmp1 - MERSENNE_PRIME;
        out = tmp2 < 0 ? tmp1 : tmp2;
    }

    return out;
}

__global__ void permute_hash_cuda_kernel(
    const int64_t *__restrict__ const input_ptr,
    const int64_t *__restrict__ const seed_ptr,
    const int64_t k,
    const int64_t n,
    const int64_t m,
    int64_t *__restrict__ const out_ptr)
{
    const int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int seed_idx = blockIdx.y * k;
    const int output_idx = blockIdx.y * n;

    for (int64_t i = input_idx; i < n; i += stride)
    {
        const int64_t input = input_ptr[i];
        int64_t output = poly_call(input, seed_ptr + seed_idx, k);
        out_ptr[output_idx + i] = output;
    }
}

__global__ void sign_hash_cuda_kernel(
    const int64_t *__restrict__ const input_ptr,
    const int64_t *__restrict__ const seed_ptr,
    const int64_t k,
    const int64_t n,
    const int64_t m,
    int64_t *__restrict__ const out_ptr)
{
    const int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int seed_idx = blockIdx.y * k;
    const int output_idx = blockIdx.y * n;

    for (int64_t i = input_idx; i < n; i += stride)
    {
        const int64_t input = input_ptr[i];
        int64_t output = poly_call(input, seed_ptr + seed_idx, k);
        output = (output & 1) * 2 - 1;
        out_ptr[output_idx + i] = output;
    }
}

__global__ void bin_hash_cuda_kernel(
    const int64_t *__restrict__ const input_ptr,
    const int64_t *__restrict__ const seed_ptr,
    const int64_t k,
    const int64_t n,
    const int64_t m,
    const int64_t b,
    int64_t *__restrict__ const out_ptr)
{
    const int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int seed_idx = blockIdx.y * k;
    const int output_idx = blockIdx.y * n;

    for (int64_t i = input_idx; i < n; i += stride)
    {
        const int64_t input = input_ptr[i];
        int64_t output = poly_call(input, seed_ptr + seed_idx, k);
        int64_t tmp1 = output % b;
        int64_t tmp2 = tmp1 + b;
        out_ptr[output_idx + i] = (tmp1 < 0) ? tmp2 : tmp1;
    }
}

__global__ void bin_hash_pow2_cuda_kernel(
    const int64_t *__restrict__ const input_ptr,
    const int64_t *__restrict__ const seed_ptr,
    const int64_t k,
    const int64_t n,
    const int64_t m,
    const int64_t b,
    int64_t *__restrict__ const out_ptr)
{
    const int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int seed_idx = blockIdx.y * k;
    const int output_idx = blockIdx.y * n;

    for (int64_t i = input_idx; i < n; i += stride)
    {
        const int64_t input = input_ptr[i];
        int64_t output = poly_call(input, seed_ptr + seed_idx, k);
        output = output & (b - 1); // mod num_bins (when a power of 2)
        out_ptr[output_idx + i] = output;
    }
}

torch::Tensor permute_hash_cuda(torch::Tensor input, torch::Tensor seeds)
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
            .device(input.device())
            .requires_grad(false);

    torch::Tensor output = torch::empty({m, n}, options);

    int64_t *input_ptr = input.data_ptr<int64_t>();
    int64_t *seeds_ptr = seeds.data_ptr<int64_t>();
    int64_t *out_ptr = output.data_ptr<int64_t>();

    const dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, m);

    permute_hash_cuda_kernel<<<blocks, BLOCK_SIZE>>>(input_ptr, seeds_ptr, k, n, m, out_ptr);

    return output;
}

torch::Tensor sign_hash_cuda(torch::Tensor input, torch::Tensor seeds)
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
            .device(input.device())
            .requires_grad(false);

    torch::Tensor output = torch::empty({m, n}, options);

    int64_t *input_ptr = input.data_ptr<int64_t>();
    int64_t *seeds_ptr = seeds.data_ptr<int64_t>();
    int64_t *out_ptr = output.data_ptr<int64_t>();

    const dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, m);
    
    sign_hash_cuda_kernel<<<blocks, BLOCK_SIZE>>>(input_ptr, seeds_ptr, k, n, m, out_ptr);

    return output;
}

torch::Tensor bin_hash_cuda(torch::Tensor input, torch::Tensor seeds, int64_t num_bins)
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
            .device(input.device())
            .requires_grad(false);

    torch::Tensor output = torch::empty({m, n}, options);

    int64_t *input_ptr = input.data_ptr<int64_t>();
    int64_t *seeds_ptr = seeds.data_ptr<int64_t>();
    int64_t *out_ptr = output.data_ptr<int64_t>();

    const dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, m);

    bool is_pow_two = (num_bins & (num_bins - 1)) == 0;

    if (is_pow_two)
    {
        bin_hash_pow2_cuda_kernel<<<blocks, BLOCK_SIZE>>>(input_ptr, seeds_ptr, k, n, m, num_bins, out_ptr);
    }
    else
    {
        bin_hash_cuda_kernel<<<blocks, BLOCK_SIZE>>>(input_ptr, seeds_ptr, k, n, m, num_bins, out_ptr);
    }

    return output;
}
