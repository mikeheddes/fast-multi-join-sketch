import math
from functools import cached_property
import torch
import torch.nn as nn
from torch import Tensor, LongTensor, BoolTensor

from kwisehash.version import __version__

import kwisehash.backend


class KWiseHash(object):

    size: tuple[int]
    seeds: LongTensor

    def __init__(self, *size: int, k: int, device: torch.device = None) -> None:
        """Universal family of hash functions which are k-independent.

        See Carter and Wegman 1977, Universal classes of hash functions.
        See https://en.wikipedia.org/wiki/Universal_hashing
        
        Args:
            *size: determines the number of times each input value will be hashed as well as the output shape.
            k: the level of independence.
            device: torch device.
        """

        self.size = size

        if device == None:
            device = torch.device("cpu")

        m = math.prod(self.size)
        self.seeds = kwisehash.backend.random_seeds_cpu(m, k).to(device)

    @property
    def k(self):
        return self.seeds.size(1)
    
    @cached_property
    def prime(self):
        return kwisehash.backend.prime()
    
    def hash(self, input: LongTensor) -> LongTensor:
        """Computes h: [p] -> [p] where p is the Mersenne prime 2^61 - 1.
        
        Note: for input values >= p or < 0, the behavior of this function is undefined.
        """
        if input.dim() != 1 or input.dtype != torch.long:
            raise ValueError("Input must be a one-dimensional long-tensor")

        if input.device.type == "cpu":
            output = kwisehash.backend.permute_hash_cpu(input, self.seeds)

        elif input.device.type == "cuda":
            output = kwisehash.backend.permute_hash_cuda(input, self.seeds)
        
        # If device is not implemented, fallback to PyTorch implementation
        else:
            input = input.unsqueeze(0)
            output = self.seeds[:, 0, None].repeat(1, input.size(1))

            # Nested computation of the exponentiation of the input:
            # i.e. ((a)x + b)x + c = ax^2 + bx + c
            for i in range(1, self.k):
                output = output * input
                output = output + self.seeds[:, i, None]
                # Prevent overflows by computing the modulo after each step
                output = output % self.prime

        return output.reshape(*self.size, -1)
    
    def __call__(self, input: LongTensor) -> LongTensor:
        return self.hash(input)
    
    def bin(self, input: LongTensor, num_bins: int) -> LongTensor:
        """Computes h: [p] -> [m] where m are the num_bins, and p is the Mersenne prime 2^61 - 1.

        A more efficient implementation is used when num_bins is a power of two.
        
        Note: for input values >= p or < 0, the behavior of this function is undefined.
        """

        if input.dim() != 1 or input.dtype != torch.long:
            raise ValueError("Input must be a one-dimensional long-tensor")

        if input.device.type == "cpu":
            output = kwisehash.backend.bin_hash_cpu(input, self.seeds, num_bins)
            return output.reshape(*self.size, -1)

        elif input.device.type == "cuda":
            output = kwisehash.backend.bin_hash_cuda(input, self.seeds, num_bins)
            return output.reshape(*self.size, -1)
        
        # If device is not implemented, fallback to PyTorch implementation
        return self.hash(input) % num_bins

    def sign(self, input: LongTensor) -> LongTensor:
        """Computes h: [p] -> {-1, +1} where p is the Mersenne prime 2^61 - 1.
        
        Note: for input values >= p or < 0, the behavior of this function is undefined.
        """

        if input.dim() != 1 or input.dtype != torch.long:
            raise ValueError("Input must be a one-dimensional long-tensor")

        if input.device.type == "cpu":
            output = kwisehash.backend.sign_hash_cpu(input, self.seeds)
            return output.reshape(*self.size, -1)

        elif input.device.type == "cuda":
            output = kwisehash.backend.sign_hash_cuda(input, self.seeds)
            return output.reshape(*self.size, -1)

        # If device is not implemented, fallback to PyTorch implementation
        return (self.hash(input) & 1) * 2 - 1

    def bool(self, input: LongTensor, prob: float) -> BoolTensor:
        """Computes h: [p] -> {0, 1} where prob is the probability of 1, and p is the Mersenne prime 2^61 - 1.
        
        Note: for input values >= p or < 0, the behavior of this function is undefined.
        """

        if prob < 0 or 1 < prob:
            raise ValueError("Probability must be between 0 and 1")
        
        return self.hash(input) < int(self.prime * prob)


__all__ = [
    "__version__",
    "KWiseHash",
]