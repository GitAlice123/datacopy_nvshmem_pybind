#include "nvshmem_extension.h"
#include <iostream>

// NVSHMEM 全局状态
bool nvshmem_initialized = false;

// 仅包含主机端初始化函数
void nvshmem_init_py() {
    if (!nvshmem_initialized) {
        // 初始化 NVSHMEM
        nvshmem_init();
        
        // 获取当前进程信息
        int my_pe = nvshmem_my_pe();
        int n_pes = nvshmem_n_pes();
        
        std::cout << "[NVSHMEM] Initialized: PE " << my_pe 
                  << " of " << n_pes << std::endl;
        
        nvshmem_initialized = true;
    }
}

// 仅测试获取 PE 信息
void nvshmem_get_info() {
    if (!nvshmem_initialized) {
        throw std::runtime_error("NVSHMEM not initialized");
    }
    
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    
    std::cout << "[NVSHMEM] PE: " << my_pe 
              << "/" << n_pes << std::endl;
}