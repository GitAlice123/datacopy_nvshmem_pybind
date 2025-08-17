#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>
#include <assert.h>
#include <torch/extension.h>
#include <vector>
#include <cstring>

// NVSHMEM 团队变量
nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
nvshmem_team_config_t cpu_rdma_team_config;

// 生成唯一标识符的函数
std::vector<uint8_t> get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
}

// 初始化 NVSHMEM
int init_nvshmem(const std::vector<uint8_t> &root_unique_id_val, int rank, int world_size) {
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    
    // 复制唯一标识符
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    
    // 设置初始化属性
    nvshmemx_set_attr_uniqueid_args(rank, world_size, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    
    // 创建子团队（如果需要）
    const int NUM_MAX_NVL_PEERS = 8;  // 最大NVL连接数
    if (world_size > NUM_MAX_NVL_PEERS) {
        // 确保团队尚未创建
        assert(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
        // 确保PE数量可被整除
        assert(world_size % NUM_MAX_NVL_PEERS == 0);
        
        // 创建子团队
        nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 
                                   rank % NUM_MAX_NVL_PEERS, 
                                   NUM_MAX_NVL_PEERS,
                                   world_size / NUM_MAX_NVL_PEERS, 
                                   &cpu_rdma_team_config, 
                                   0, 
                                   &cpu_rdma_team);
        
        // 确保团队创建成功
        assert(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
    }
    
    // 全局同步
    nvshmem_barrier_all();
    return nvshmem_my_pe();
}

// CUDA Kernel 实现跨机通信
__global__ void cross_device_put_kernel(float* data, int target_pe, int rank) {
    if (threadIdx.x == 0) {
        // 所有 PE 的同步点
        nvshmem_barrier_all();
        
        if (rank == 0) {
            // 在 GPU0 上准备要发送的数据
            float send_value = 3.14159f;
            
            // 打印调试信息
            printf("[GPU0] Sending value %.5f to PE %d\n", send_value, target_pe);
            
            // 使用 NVSHMEM 的 put 接口发送数据
            nvshmem_float_p(data, send_value, target_pe);
            
            // 确保数据发送完成
            nvshmem_quiet();
        }else{
            // sleep for a while
            auto cur_time = clock64();
            while (clock64() - cur_time < 10000000) {
                // busy wait
            }
        }
    }
}

// 测试跨机通信
void test_cross_device_comm(float* d_data, int rank, int world_size) {
    // 分配主机内存用于验证
    float *h_data = (float*)malloc(sizeof(float));
    *h_data = 0.0f;
    
    // 重置设备数据
    cudaMemset(d_data, 0, sizeof(float));
    
    // 设置目标 PE (rank 0 -> rank 1)
    int target_pe = (rank == 0) ? 1 : 0;
    
    // 启动核函数
    dim3 block(32, 1, 1);
    dim3 grid(1, 1, 1);
    cross_device_put_kernel<<<grid, block>>>(d_data, target_pe, rank);
    
    // 等待内核完成
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(h_data, d_data, sizeof(float), cudaMemcpyDeviceToHost);
    
    // 等待所有 PE 完成
    nvshmem_barrier_all();
    
    if (rank == 0) {
        // GPU0 验证发送成功
        printf("[GPU0] Test completed. Sent value: %.5f\n", 3.14159f);
    } else {
        // GPU1 验证接收的数据
        const float expected = 3.14159f;
        const float tolerance = 1e-5f;
        float received = *h_data;
        
        // 验证数据
        if (fabs(received - expected) < tolerance) {
            printf("[GPU1] Test PASSED! Received value: %.5f\n", received);
        } else {
            printf("[GPU1] Test FAILED! Expected %.5f, got %.5f\n", expected, received);
        }
    }
    
    free(h_data);
}

// 清理 NVSHMEM
void finalize_nvshmem() {
    if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
        nvshmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = NVSHMEM_TEAM_INVALID;
    }
    nvshmem_finalize();
}

// 公开给 Pybind 的接口
void cuda_cross_device_test(int rank, int world_size, const std::vector<uint8_t>& root_unique_id) {
    // 初始化 NVSHMEM
    int my_pe = init_nvshmem(root_unique_id, rank, world_size);
    printf("[NVSHMEM] Rank %d initialized as PE %d\n", rank, my_pe);
    
    // 分配对称内存 (所有 PE 可见)
    float *d_data = (float*)nvshmem_malloc(sizeof(float));
    
    // 测试跨机通信
    test_cross_device_comm(d_data, rank, world_size);
    
    // 清理
    nvshmem_free(d_data);
    finalize_nvshmem();
}