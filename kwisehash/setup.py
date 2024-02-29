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

# Adopted from: https://github.com/mit-han-lab/torchsparse/tree/master

import sys
import os
import glob

import torch
import torch.cuda
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

version_file = open("./kwisehash/version.py")
version = version_file.read().split('"')[-2]
print("kwisehash version:", version)

has_cuda = torch.cuda.is_available() and CUDA_HOME is not None
forces_cuda_build = os.getenv("FORCE_CUDA", "0") == "1"
if has_cuda or forces_cuda_build:
    device = "cuda"
    pybind_fn = f"pybind_{device}.cu"
else:
    device = "cpu"
    pybind_fn = f"pybind_{device}.cpp"

sources = [os.path.join("kwisehash", "backend", pybind_fn)]
for fpath in glob.glob(os.path.join("kwisehash", "backend", "**", "*")):
    # The cpu files are build for both devices
    select_cpu = fpath.endswith("_cpu.cpp") and device in ["cpu", "cuda"]
    select_cuda = fpath.endswith("_cuda.cu") and device == "cuda"

    if select_cpu or select_cuda:
        sources.append(fpath)

extension_type = CUDAExtension if device == "cuda" else CppExtension

extra_compile_args = {
    "cxx": [
        "-O3",
        "-Wno-unused-variable",
        "-march=native",
        "-ffast-math",
        "-fno-finite-math-only",
        # this has to be before -fopenmp to enable it to use openmp on MacOS (tested M2 Pro)
        "-Xclang" if sys.platform == "darwin" else None,
        "-fopenmp",
    ],
    "nvcc": ["-O3", "-std=c++17"],
}

# Remove None entries
for key in extra_compile_args.keys():
    extra_compile_args[key] = list(filter(lambda x: x, extra_compile_args[key]))

VENV_PREFIX = os.path.abspath(os.path.join(sys.executable, "..", ".."))
current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "kwisehash", "backend"))

setup(
    name="kwisehash",
    version=version,
    packages=find_packages(),
    ext_modules=[
        extension_type(
            "kwisehash.backend",
            sorted(sources),
            extra_compile_args=extra_compile_args,
            include_dirs=[os.path.join(VENV_PREFIX, "include"), current_dir],
            library_dirs=[os.path.join(VENV_PREFIX, "lib")],
            libraries=["omp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
