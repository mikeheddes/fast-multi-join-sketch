#pragma once

// http://en.wikipedia.org/wiki/Mersenne_prime
constexpr int64_t MERSENNE_PRIME = 0x1fffffffffffffff; // (2^61) - 1
// note: make sure this value stays in-sync with the one in the Python source

// See PyTorch: aten/src/ATen/TensorIterator.h
// This parameter is heuristically chosen to determine the minimum number of
// work that warrants parallelism. For example, when summing an array, it is
// deemed inefficient to parallelise over arrays shorter than 32768. 
// Further, no parallel algorithm (such as parallel_reduce) should split work 
// into smaller than GRAIN_SIZE chunks.
constexpr int64_t GRAIN_SIZE = 32768;
