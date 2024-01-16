#pragma once

#include <torch/torch.h>

#include "kwisehash/constants.h"

torch::Tensor permute_hash_cuda(torch::Tensor input, torch::Tensor seeds);

torch::Tensor sign_hash_cuda(torch::Tensor input, torch::Tensor seeds);

torch::Tensor bin_hash_cuda(torch::Tensor input, torch::Tensor seeds, int64_t num_bins);
