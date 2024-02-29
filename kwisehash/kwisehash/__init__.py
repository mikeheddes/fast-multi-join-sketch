#
# Software License
# Commercial reservation
#
# This License governs use of the accompanying Software, and your use of the Software constitutes acceptance of this license.
#
# You may use this Software for any non-commercial purpose, subject to the restrictions in this license. Some purposes which can be non-commercial are teaching, academic research, and personal experimentation. 
#
# You may not use or distribute this Software or any derivative works in any form for any commercial purpose. Examples of commercial purposes would be running business operations, licensing, leasing, or selling the Software, or distributing the Software for use with commercial products. 
#
# You may modify this Software and distribute the modified Software for non-commercial purposes; however, you may not grant rights to the Software or derivative works that are broader than those provided by this License. For example, you may not distribute modifications of the Software under terms that would permit commercial use, or under terms that purport to require the Software or derivative works to be sublicensed to others.
#
# You agree:
#
# 1. Not remove any copyright or other notices from the Software.
#
# 2. That if you distribute the Software in source or object form, you will include a verbatim copy of this license.
#
# 3. That if you distribute derivative works of the Software in source code form you do so only under a license that includes all of the provisions of this License, and if you distribute derivative works of the Software solely in object form you do so only under a license that complies with this License.
#
# 4. That if you have modified the Software or created derivative works, and distribute such modifications or derivative works, you will cause the modified files to carry prominent notices so that recipients know that they are not receiving the original Software. Such notices must state: (i) that you have changed the Software; and (ii) the date of any changes.
#
# 5. THAT THIS PRODUCT IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS PRODUCT, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. YOU MUST PASS THIS LIMITATION OF LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
#
# 6. That if you sue anyone over patents that you think may apply to the Software or anyone's use of the Software, your license to the Software ends automatically.
#
# 7. That your rights under the License end automatically if you breach it in any way.
#
# 8. UC Irvine and The Regents of the University of California reserves all rights not expressly granted to you in this license.
#
# To obtain a commercial license to this software, please contact:
# UCI Beall Applied Innovation
# Attn: Director, Research Translation Group
# 5270 California Ave, Suite 100
# Irvine, CA 92697
# Website: innovation.uci.edu
# Phone: 949-824-COVE (2683) 
# Email: cove@uci.edu
#
# Standard BSD License
#
# <OWNER> = The Regents of the University of California
# <ORGANIZATION> = University of California, Irvine
# <YEAR> = 2020
#
# Copyright (c) <2020>, The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of The Regents of the University of California or the University of California, Irvine, nor the names of its contributors, may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
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