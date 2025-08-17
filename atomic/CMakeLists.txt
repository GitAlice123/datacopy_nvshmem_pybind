cmake_minimum_required(VERSION 3.18)
project(nvshmem_extension)

# 1. 设置关键路径
set(CUDA_HOME "/usr/local/cuda")
set(NVSHMEM_DIR "/usr/local/nvshmem")
set(TORCH_CMAKE_DIR "/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch")

# 2. 将 Torch CMake 路径添加到搜索路径
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${TORCH_CMAKE_DIR}")

# 3. 查找 PyTorch
find_package(Torch REQUIRED CONFIG)
message(STATUS "Found Torch: ${Torch_DIR}")
message(STATUS "Torch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

# 4. 启用 CUDA 支持
enable_language(CUDA)

# 5. 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")

# 6. 手动定义 cudadevrt 目标（关键修复）
if(NOT TARGET CUDA::cudadevrt)
    message(STATUS "Creating manual target for CUDA::cudadevrt")
    
    # 查找 libcudadevrt.a
    find_library(CUDA_DEVRT_LIB 
        NAMES cudadevrt
        PATHS "${CUDA_HOME}/lib64"
        REQUIRED)
    message(STATUS "Found cudadevrt: ${CUDA_DEVRT_LIB}")
    
    # 创建导入目标
    add_library(CUDA::cudadevrt STATIC IMPORTED)
    set_target_properties(CUDA::cudadevrt PROPERTIES
        IMPORTED_LOCATION "${CUDA_DEVRT_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDA_HOME}/include")
endif()

# 7. 设置 GPU 架构
set(CMAKE_CUDA_ARCHITECTURES "90")  # 90 for Hopper, 80 for Ampere

# 8. 设置编译选项
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fPIC>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>")

# 9. 添加扩展源文件
add_library(nvshmem_extension SHARED
    nvshmem_extension.cpp
)

# 10. 设置包含目录
target_include_directories(nvshmem_extension PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${NVSHMEM_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 11. 设置链接库（使用绝对路径）
target_link_libraries(nvshmem_extension PRIVATE
    ${TORCH_LIBRARIES}
    ${NVSHMEM_DIR}/lib/libnvshmem.a
    ${NVSHMEM_DIR}/lib/libnvshmem_device.a
    ${NVSHMEM_DIR}/lib/nvshmem_bootstrap_uid.so
    ${CUDA_HOME}/lib64/libcudart.so
    CUDA::cudadevrt  # 现在应该可以找到了
)

# 12. 设置链接选项
target_link_options(nvshmem_extension PRIVATE
    -Wl,-rpath,${CUDA_HOME}/lib64
    -Wl,-rpath,${NVSHMEM_DIR}/lib
    -lpthread -ldl -lrt
)

# 13. 设置输出目录
set_target_properties(nvshmem_extension PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 14. 安装配置
install(TARGETS nvshmem_extension
    LIBRARY DESTINATION lib
)