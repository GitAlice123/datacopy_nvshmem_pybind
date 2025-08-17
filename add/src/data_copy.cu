#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>
#include <assert.h>
#include <torch/extension.h>
#include <vector>
#include <cstring>
#include "ibgda_device.cuh"

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

__global__ void ibgda_initialize_recv_queue(int rank) {
    auto thread_idx = static_cast<int>(threadIdx.x);
    auto num_threads = static_cast<int>(blockDim.x);

    auto dst_rank = static_cast<int>(blockIdx.x);
    if (dst_rank != rank) {
        for (int qp_id = thread_idx; qp_id < datacopy::ibgda_get_state()->num_rc_per_pe; qp_id += num_threads) {
            auto qp = datacopy::ibgda_get_rc(dst_rank, qp_id);

            // Clean some necessary variables
            for (int i = 0; i < qp->rx_wq.nwqes; ++ i)
                datacopy::ibgda_write_empty_recv_wqe(datacopy::ibgda_get_wqe_ptr(qp, i));
            qp->mvars.rx_wq.resv_head = 0;
            qp->mvars.rx_wq.cons_idx = 0;

            // Allocate receive slots
            datacopy::nvshmemi_ibgda_allocate_recvs(qp);
        }
    }
}

int init_nvshmem(const std::vector<uint8_t> &root_unique_id_val, int rank, int world_size) {
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    
    nvshmemx_set_attr_uniqueid_args(rank, world_size, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    
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

    nvshmemi_device_host_state_t* dev_state_ptr = nullptr;
    CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&dev_state_ptr), nvshmemi_device_state_d));

    bool ibgda_is_initialized = false;
    cudaMemcpy(&dev_state_ptr->ibgda_is_initialized, &ibgda_is_initialized, sizeof(bool), cudaMemcpyHostToDevice);    

    ibgda_initialize_recv_queue<<<world_size, 128>>>(rank);
    
    nvshmem_barrier_all();
    return nvshmem_my_pe();
}

// CUDA Kernel communicate using NET
__global__ void cross_device_put_kernel(int* data, int target_pe, int rank) {
    if (threadIdx.x == 0) {
        nvshmem_barrier_all();
        
        if (rank == 0) {
            int send_value = 12345;
            
            printf("[GPU0] Sending value %d to PE %d\n", send_value, target_pe);
            
            // nvshmem_int_p(data, send_value, target_pe);
            // nvshmemi_ibgda_rma_p(int *rptr, const int value, int dst_pe, int qp_id, uint32_t imm = std::numeric_limits<uint32_t>::max())
            datacopy::nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(data), send_value, target_pe, 0);
            
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

void test_cross_device_comm(int* d_data, int rank, int world_size) {
    int *h_data = (int*)malloc(sizeof(int));
    *h_data = 0.0f;
    
    cudaMemset(d_data, 0, sizeof(int));
    
    int target_pe = (rank == 0) ? 1 : 0;
    
    dim3 block(32, 1, 1);
    dim3 grid(1, 1, 1);
    cross_device_put_kernel<<<grid, block>>>(d_data, target_pe, rank);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    
    nvshmem_barrier_all();
    
    if (rank == 0) {
        printf("[GPU0] Test completed. Sent value: 12345\n");
    } else {
        const int expected = 12345;
        const int tolerance = 1;
        int received = *h_data;
        
        if (fabs(received - expected) < tolerance) {
            printf("[GPU1] Test PASSED! Received value: %d\n", received);
        } else {
            printf("[GPU1] Test FAILED! Expected %d, got %d\n", expected, received);
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
    
    int *d_data = (int*)nvshmem_malloc(sizeof(int));
    
    test_cross_device_comm(d_data, rank, world_size);
    
    nvshmem_free(d_data);
    finalize_nvshmem();
}