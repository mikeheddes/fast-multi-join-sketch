#pragma once

#include <torch/torch.h>

#include "kwisehash/constants.h"

/// @brief The prime number used by the polynomial hash functions
/// @link http://en.wikipedia.org/wiki/Mersenne_prime
/// @return Mersenne prime (2^61) - 1
int64_t prime();

/// @brief Sample the random seeds from the k-wise independent family of polynomial hash functions
/// @param m number of hash functions
/// @param k independence of the family of hash functions
/// @param device device on which to sample and store the seeds
/// @return 64-bit integer seeds
torch::Tensor random_seeds_cpu(int64_t m, int64_t k);

torch::Tensor permute_hash_cpu(torch::Tensor input, torch::Tensor seeds);

torch::Tensor sign_hash_cpu(torch::Tensor input, torch::Tensor seeds);

torch::Tensor bin_hash_cpu(torch::Tensor input, torch::Tensor seeds, int64_t num_bins);
