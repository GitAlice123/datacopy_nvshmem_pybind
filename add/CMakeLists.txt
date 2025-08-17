cmake_minimum_required(VERSION 3.18)
project(pydatacopy LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE OFF)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    message(STATUS "Using ccache for compilation acceleration")
endif()

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")

set(TORCH_DIR "/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch")
message(STATUS "Using Torch at: ${TORCH_DIR}")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})

set(CMAKE_CUDA_ARCHITECTURES "90")
set(TORCH_CUDA_ARCH_LIST "9.0")


find_package(Torch REQUIRED CONFIG)


find_package(pybind11 REQUIRED CONFIG)


set(NVSHMEM_HOME "/usr/local/nvshmem")
message(STATUS "Using NVSHMEM at: ${NVSHMEM_HOME}")
set(NVSHMEM_LIB "${NVSHMEM_HOME}/lib/libnvshmem.a")


find_package(CUDAToolkit REQUIRED)


set(SOURCES 
    pybind_data_copy.cpp
    data_copy.cu
)

# 7. 获取 Python 模块后缀
execute_process(
    COMMAND python3.12 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_MODULE_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python module suffix: ${PYTHON_MODULE_SUFFIX}")

# 8. 创建模块
add_library(pydatacopy SHARED ${SOURCES})
set_target_properties(pydatacopy PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_SUFFIX}"
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# 9. 添加包含目录
target_include_directories(pydatacopy PRIVATE
    ${Python_INCLUDE_DIRS}
    ${Torch_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${NVSHMEM_HOME}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 10. 添加链接库
target_link_libraries(pydatacopy PRIVATE
    ${TORCH_LIBRARIES}
    ${Python_LIBRARIES}
    cudart
    ${NVSHMEM_LIB}
    dl
    pybind11::pybind11
)

# 11. 添加编译选项 - 强制指定架构
target_compile_options(pydatacopy PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-fPIC
        --expt-relaxed-constexpr
        --ptxas-options=-O3  # 优化PTX汇编
        --maxrregcount=64    # 限制寄存器使用以提高并行度
        --use_fast_math      # 使用快速数学（精度略低但更快）
        --gpu-architecture=compute_90  # 强制指定计算能力
        --gpu-code=sm_90    # 强制指定目标架构
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -fPIC
        -O3
        -march=native        # 使用本地CPU架构优化
    >
)

# 12. 设置输出目录
set_target_properties(pydatacopy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 13. 安装规则
install(TARGETS pydatacopy 
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 14. 配置汇总 - 仅显示关键信息
message(STATUS "== 配置汇总 ==")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "NVSHMEM library: ${NVSHMEM_LIB}")