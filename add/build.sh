#!/bin/bash

# 清除旧构建
rm -rf build
mkdir build
cd build

# 设置环境变量
export NVSHMEM_HOME=/usr/local/nvshmem
export Python_ROOT_DIR=$(python3.12 -c "import sys; print(sys.prefix)")

# 关键修改：设置 Torch 架构环境变量
export TORCH_CUDA_ARCH_LIST="9.0"  # 点分隔格式

# 配置阶段 - 使用Ninja构建系统
cmake .. \
    -G Ninja \  # 使用Ninja替代Make
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DPython_EXECUTABLE=$(which python3.12) \
    -DTorch_DIR=/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch \
    -Dpybind11_DIR=$(python3.12 -c "import pybind11; print(pybind11.get_cmake_dir())")

# 编译阶段 - 使用所有核心并行编译
ninja -j$(nproc)

# 验证构建
find . -name "pydatacopys*" -ls