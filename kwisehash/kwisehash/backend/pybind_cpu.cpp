#include <torch/extension.h>

#include "kwisehash/kwisehash_cpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prime", &prime);
  m.def("random_seeds_cpu", &random_seeds_cpu);
  m.def("permute_hash_cpu", &permute_hash_cpu);
  m.def("sign_hash_cpu", &sign_hash_cpu);
  m.def("bin_hash_cpu", &bin_hash_cpu);
}