/*
 * Software License
 * Commercial reservation
 *
 * This License governs use of the accompanying Software, and your use of the Software constitutes acceptance of this license.
 *
 * You may use this Software for any non-commercial purpose, subject to the restrictions in this license. Some purposes which can be non-commercial are teaching, academic research, and personal experimentation. 
 *
 * You may not use or distribute this Software or any derivative works in any form for any commercial purpose. Examples of commercial purposes would be running business operations, licensing, leasing, or selling the Software, or distributing the Software for use with commercial products. 
 *
 * You may modify this Software and distribute the modified Software for non-commercial purposes; however, you may not grant rights to the Software or derivative works that are broader than those provided by this License. For example, you may not distribute modifications of the Software under terms that would permit commercial use, or under terms that purport to require the Software or derivative works to be sublicensed to others.
 *
 * You agree:
 *
 * 1. Not remove any copyright or other notices from the Software.
 *
 * 2. That if you distribute the Software in source or object form, you will include a verbatim copy of this license.
 *
 * 3. That if you distribute derivative works of the Software in source code form you do so only under a license that includes all of the provisions of this License, and if you distribute derivative works of the Software solely in object form you do so only under a license that complies with this License.
 *
 * 4. That if you have modified the Software or created derivative works, and distribute such modifications or derivative works, you will cause the modified files to carry prominent notices so that recipients know that they are not receiving the original Software. Such notices must state: (i) that you have changed the Software; and (ii) the date of any changes.
 *
 * 5. THAT THIS PRODUCT IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS PRODUCT, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. YOU MUST PASS THIS LIMITATION OF LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
 *
 * 6. That if you sue anyone over patents that you think may apply to the Software or anyone's use of the Software, your license to the Software ends automatically.
 *
 * 7. That your rights under the License end automatically if you breach it in any way.
 *
 * 8. UC Irvine and The Regents of the University of California reserves all rights not expressly granted to you in this license.
 *
 * To obtain a commercial license to this software, please contact:
 * UCI Beall Applied Innovation
 * Attn: Director, Research Translation Group
 * 5270 California Ave, Suite 100
 * Irvine, CA 92697
 * Website: innovation.uci.edu
 * Phone: 949-824-COVE (2683) 
 * Email: cove@uci.edu
 *
 * Standard BSD License
 *
 * <OWNER> = The Regents of the University of California
 * <ORGANIZATION> = University of California, Irvine
 * <YEAR> = 2020
 *
 * Copyright (c) <2020>, The Regents of the University of California
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of The Regents of the University of California or the University of California, Irvine, nor the names of its contributors, may be used to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
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
