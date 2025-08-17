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

std::vector<uint8_t> get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
}

int init_nvshmem(const std::vector<uint8_t> &root_unique_id_val, int rank, int world_size) {
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    
    nvshmemx_set_attr_uniqueid_args(rank, world_size, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    
    const int NUM_MAX_NVL_PEERS = 8;
    if (world_size > NUM_MAX_NVL_PEERS) {
        assert(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
        assert(world_size % NUM_MAX_NVL_PEERS == 0);
        
        nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 
                                   rank % NUM_MAX_NVL_PEERS, 
                                   NUM_MAX_NVL_PEERS,
                                   world_size / NUM_MAX_NVL_PEERS, 
                                   &cpu_rdma_team_config, 
                                   0, 
                                   &cpu_rdma_team);
        
        assert(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
    }
    
    nvshmem_barrier_all();
    return nvshmem_my_pe();
}

// CUDA Kernel communicate using NET
__global__ void cross_device_put_kernel(float* data, int target_pe, int rank) {
    if (threadIdx.x == 0) {
        nvshmem_barrier_all();
        
        if (rank == 0) {
            float send_value = 3.14159f;
            
            printf("[GPU0] Sending value %.5f to PE %d\n", send_value, target_pe);
            
            nvshmem_float_p(data, send_value, target_pe);
            
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

void test_cross_device_comm(float* d_data, int rank, int world_size) {
    float *h_data = (float*)malloc(sizeof(float));
    *h_data = 0.0f;
    
    cudaMemset(d_data, 0, sizeof(float));
    
    int target_pe = (rank == 0) ? 1 : 0;
    
    dim3 block(32, 1, 1);
    dim3 grid(1, 1, 1);
    cross_device_put_kernel<<<grid, block>>>(d_data, target_pe, rank);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data, d_data, sizeof(float), cudaMemcpyDeviceToHost);
    
    nvshmem_barrier_all();
    
    if (rank == 0) {
        printf("[GPU0] Test completed. Sent value: %.5f\n", 3.14159f);
    } else {
        const float expected = 3.14159f;
        const float tolerance = 1e-5f;
        float received = *h_data;
        
        if (fabs(received - expected) < tolerance) {
            printf("[GPU1] Test PASSED! Received value: %.5f\n", received);
        } else {
            printf("[GPU1] Test FAILED! Expected %.5f, got %.5f\n", expected, received);
        }
    }
    
    free(h_data);
}

void finalize_nvshmem() {
    if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
        nvshmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = NVSHMEM_TEAM_INVALID;
    }
    nvshmem_finalize();
}

void cuda_cross_device_test(int rank, int world_size, const std::vector<uint8_t>& root_unique_id) {
    int my_pe = init_nvshmem(root_unique_id, rank, world_size);
    printf("[NVSHMEM] Rank %d initialized as PE %d\n", rank, my_pe);
    
    float *d_data = (float*)nvshmem_malloc(sizeof(float));
    
    test_cross_device_comm(d_data, rank, world_size);
    
    nvshmem_free(d_data);
    finalize_nvshmem();
}