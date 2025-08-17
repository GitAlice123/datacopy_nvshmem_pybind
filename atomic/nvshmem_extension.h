#include <torch/extension.h>
#include <nvshmem.h>

// 初始化 NVSHMEM
void nvshmem_init_py();

// 获取 PE 信息
void nvshmem_get_info();

// 绑定 Python 接口
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvshmem_init_py", &nvshmem_init_py, "Initialize NVSHMEM");
    m.def("nvshmem_get_info", &nvshmem_get_info, "Get NVSHMEM PE info");
}