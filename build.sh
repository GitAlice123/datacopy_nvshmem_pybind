#!/bin/bash

# rm -rf build
# mkdir build
cd build

export NVSHMEM_HOME=/usr/local/nvshmem
export Python_ROOT_DIR=$(python3.12 -c "import sys; print(sys.prefix)")

export TORCH_CUDA_ARCH_LIST="9.0"

# cmake .. \
#     -G Ninja \  # use ninja instead of make to get faster
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CUDA_ARCHITECTURES=90 \
#     -DPython_EXECUTABLE=$(which python3.12) \
#     -DTorch_DIR=/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch \
#     -Dpybind11_DIR=$(python3.12 -c "import pybind11; print(pybind11.get_cmake_dir())")

ninja -j$(nproc)