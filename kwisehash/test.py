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

