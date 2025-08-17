from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob

# 设置关键路径
nvshmem_dir = os.getenv("NVSHMEM_DIR", "/usr/local/nvshmem")
cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")

print(f"[CONFIG] NVSHMEM_DIR: {nvshmem_dir}")
print(f"[CONFIG] CUDA_HOME: {cuda_home}")

# 查找所有可能的 NVSHMEM 库
nvshmem_libs = []
for lib_name in ["libnvshmem.a", "libnvshmem_host.so*", "libnvshmem_device.a"]:
    nvshmem_libs.extend(glob.glob(f"{nvshmem_dir}/lib/{lib_name}"))
print(f"Found NVSHMEM libraries: {nvshmem_libs}")

setup(
    name='nvshmem_extension',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            'nvshmem_extension',
            sources=['nvshmem_extension.cpp'],
            include_dirs=[
                f"{nvshmem_dir}/include",
                f"{cuda_home}/include",
                # 添加 Torch 头文件路径 - 关键添加
                os.path.join(os.path.dirname(__file__), "torch/include"),
            ],
            library_dirs=[
                f"{nvshmem_dir}/lib",
                f"{cuda_home}/lib64"
            ],
            extra_compile_args={
                'cxx': [
                    '-O3', 
                    '-fPIC', 
                    '-std=c++17',
                    '-D_FORCE_INLINES'
                ],
                'nvcc': [
                    '-O3',
                    '-Xcompiler', '-fPIC',
                    '-std=c++17',
                    '-rdc=true',  # 启用可重定位设备代码
                    f'-arch=sm_90',  # 替换为您的GPU架构
                    '--expt-relaxed-constexpr',
                    # 关键：添加设备链接选项
                    '--generate-code=arch=compute_90,code=sm_90'
                ]
            },
            extra_link_args=[
                # 1. 首先链接设备运行时 - 关键顺序调整
                f"{cuda_home}/lib64/libcudadevrt.a",
                
                # 2. NVSHMEM 主库
                *[f"{lib}" for lib in nvshmem_libs],
                
                # 3. NVSHMEM 引导库
                f"{nvshmem_dir}/lib/nvshmem_bootstrap_uid.so",
                
                # 4. CUDA 运行时库
                f"{cuda_home}/lib64/libcudart.so",
                
                # 5. 系统库
                "-ldl", "-lpthread", "-lrt",
                
                # 6. 运行时路径
                f"-Wl,-rpath,{cuda_home}/lib64",
                f"-Wl,-rpath,{nvshmem_dir}/lib",
                
                # 7. 设备链接选项 - 关键添加
                "-dlink",
                f"-L{cuda_home}/lib64",
                "-lcudadevrt"
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(
            use_ninja=False,
            no_python_abi_suffix=True,
            # 关键：强制使用 nvcc 作为链接器
            use_cuda_linker=True
        )
    }
)