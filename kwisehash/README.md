# `kwisehash`: K-Wise Independent Hashing

Efficient implementation of k-wise independent polynomial hash functions, also known as [universal hash functions](https://en.wikipedia.org/wiki/Universal_hashing). The functions use the [Mersenne prime](http://en.wikipedia.org/wiki/Mersenne_prime) `2^61-1`. The implementation is based on [The Power of Hashing with Mersenne Primes](https://arxiv.org/abs/2008.08654).

## Compile hash functions

The following steps assume that a conda environment with Python and PyTorch is available. To install PyTorch, [consult their documentation](https://pytorch.org/get-started/locally/).

```bash
# Create/activate conda environment

# Install a C++ compiler
conda install cxx-compiler

# Optional, required for cuda build 
# Ensure version matches pytorch cuda version (use conda list)
conda install -c nvidia cuda-toolkit=x.x

# Compile the source code
pip install . 
```


## Benchmark

The file `test.py` tests both the correctness of the implementation and benchmarks the available implementations. The settings are `k=4`, `m=5` and `n=1048575`, meaning that five 4-way independent hash functions are each evaluated for 1,048,575 random inputs, resulting in 5,242,875 total hash function evaluations. The following table shows the average time in milliseconds:

| Device | Python | C++ |
| --- | --- | --- |
| Intel Xeon Gold 6240R CPU @ 2.40GHz | 27.19 | 3.821 |
| Intel Xeon Gold 6148 CPU @ 2.40GHz | 27.76 | 3.863 |
| Intel Xeon Gold 6326 CPU @ 2.90GHz | 12.84 | 3.178 |
| Apple M2 Pro | 5.677 | 2.177 |
| Nvidia Tesla V100 (16GB) | 0.1404 | 0.01560 |
| Nvidia A100 (80GB) | 0.05855 | 0.007452 |


Note: make sure to recompile for the specific Nvidia device. For some reason I don't understand yet, the compiled code for one does not run on the other.
