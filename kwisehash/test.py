import os, sys

self_dir = os.path.dirname(os.path.realpath(__file__))
if self_dir in sys.path:
    sys.path.remove(self_dir)

import torch
from torch import Tensor, LongTensor
import torch.utils.benchmark as benchmark
from tqdm import trange

MERSENNE_PRIME = (1 << 61) - 1
ITERS = 10

def true(input:LongTensor, seeds: LongTensor) -> LongTensor:
    k = seeds.size(0)

    input = input.tolist()
    seeds = seeds.tolist()
    output = torch.empty(len(seeds), len(input), dtype=torch.long)

    for i in trange(len(input)):
        for m in range(len(seeds)):
            out = seeds[m][0]
            for k in range(1, len(seeds[m])):
                out = out * input[i] + seeds[m][k]
                out = out % MERSENNE_PRIME
            
            output[m, i] = out
            
    return output


def pytorch(input:LongTensor, seeds: LongTensor) -> LongTensor:
    k = seeds.size(1)

    input = input.unsqueeze(0)
    output = seeds[:, 0, None].repeat(1, input.size(1))

    # Nested computation of the exponentiation of the input:
    # i.e. ((a)x + b)x + c = ax^2 + bx + c
    for i in range(1, k):
        output = output * input
        output = output + seeds[:, i, None]
        # Prevent overflows by computing the modulo after each step
        output = output % MERSENNE_PRIME

    return output

m = 5
k = 4
size = (m, k - 1)
a = torch.randint(1, MERSENNE_PRIME, size)

# Only the constant can be zero~
size = (m, 1)
b = torch.randint(0, MERSENNE_PRIME, size)

params = torch.cat((a, b), dim=1)

# Shift by 1 to ensure it does not align with the cache lines
x = torch.linspace(0, MERSENNE_PRIME - 1, (1<<20) - 1, dtype=torch.long)[1:]

target = true(x, params)
# print(target)

y = pytorch(x, params)
duration = benchmark.Timer(stmt="permute(x, params)", globals=dict(permute=pytorch, x=x, params=params)).timeit(ITERS).mean
print("Python (cpu)\t", duration * 1000 / ITERS, torch.all(y == target).item())
# print(y)

import kwisehash.backend

y = kwisehash.backend.permute_hash_cpu(x, params)
duration = benchmark.Timer(stmt="permute(x, params)", globals=dict(permute=kwisehash.backend.permute_hash_cpu, x=x, params=params)).timeit(ITERS).mean
print("C++ (cpu)\t", duration * 1000 / ITERS, torch.all(y == target).item())
# print(y)

if torch.backends.mps.is_available():
    device = torch.device("mps")

    y = pytorch(x.to(device), params.to(device))
    duration = benchmark.Timer(stmt="permute(x, params)", globals=dict(permute=pytorch, x=x.to(device), params=params.to(device))).timeit(ITERS).mean
    print("Python (mps)\t", duration * 1000 / ITERS, torch.all(y.cpu() == target).item())

if torch.cuda.is_available():
    device = torch.device("cuda")

    y = pytorch(x.to(device), params.to(device))
    duration = benchmark.Timer(stmt="permute(x, params)", globals=dict(permute=pytorch, x=x.to(device), params=params.to(device))).timeit(ITERS).mean
    print("Python (cuda)\t", duration * 1000 / ITERS, torch.all(y.cpu() == target).item())

    y = kwisehash.backend.permute_hash_cuda(x.to(device), params.to(device))
    duration = benchmark.Timer(stmt="permute(x, params)", globals=dict(permute=kwisehash.backend.permute_hash_cuda, x=x.to(device), params=params.to(device))).timeit(ITERS).mean
    print("C++ (cuda)\t", duration * 1000 / ITERS, torch.all(y.cpu() == target).item())

