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
version = version_file.read().split('"')[1]
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
