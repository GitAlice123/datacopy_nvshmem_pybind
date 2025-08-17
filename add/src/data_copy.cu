#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>
#include <assert.h>
#include <torch/extension.h>
#include <vector>
#include <cstring>
#include <atomic>
#include "ibgda_device.cuh"
#include "double_buffer_manager.h"
__global__ void send_recv_clean_kernel(DoubleBufferState state, int dst_pe);

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

    nvshmemi_device_host_state_t* dev_state_ptr = nullptr;
    CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&dev_state_ptr), nvshmemi_device_state_d));

    bool ibgda_is_initialized = false;
    cudaMemcpy(&dev_state_ptr->ibgda_is_initialized, &ibgda_is_initialized, sizeof(bool), cudaMemcpyHostToDevice);    

    ibgda_initialize_recv_queue<<<world_size, 128>>>(rank);
    
    nvshmem_barrier_all();
    return nvshmem_my_pe();
}


std::vector<uint8_t> get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
}

// 实现 DoubleBufferManager 的方法
DoubleBufferManager::DoubleBufferManager() : initialized(false), n_tokens(0), token_size(0) {}

DoubleBufferManager::~DoubleBufferManager() {
    cleanup();
}

void DoubleBufferManager::init(int n_tokens, int token_size) {
    if (initialized) {
        cleanup();
    }
    this->n_tokens = n_tokens;
    this->token_size = token_size;
    init_double_buffer(&state, n_tokens, token_size);
    initialized = true;
}

void DoubleBufferManager::cleanup() {
    if (initialized) {
        cleanup_double_buffer(&state);
        initialized = false;
    }
}

void DoubleBufferManager::test_bandwidth(int rank, int world_size) {
    if (!initialized) {
        throw std::runtime_error("DoubleBufferManager not initialized");
    }
    int dst_pe = (rank == 0) ? 1 : 0;
    
    // 计算线程配置
    int total_need_threads = n_tokens * 2;
    int block_size = 256;
    int grid_size = (total_need_threads + block_size - 1) / block_size;
    
    // 启动内核
    send_recv_clean_kernel<<<grid_size, block_size>>>(state, dst_pe);
    
    // 等待内核完成
    cudaDeviceSynchronize();

    // 切换缓冲区
    state.current_buffer = 1 - state.current_buffer;
}

void DoubleBufferManager::init_double_buffer(DoubleBufferState* state, int n_tokens, int token_size) {
    state->n_tokens = n_tokens;
    state->token_size = token_size;
    state->current_buffer = 0;
    
    // 检查参数有效性
    if (n_tokens <= 0 || token_size <= 0) {
        fprintf(stderr, "Error: Invalid parameters: n_tokens=%d, token_size=%d\n", n_tokens, token_size);
        exit(EXIT_FAILURE);
    }
    
    // 分配缓冲区0
    state->buffer0_send = (char*)nvshmem_malloc(n_tokens * token_size);
    state->buffer0_recv = (char*)nvshmem_malloc(n_tokens * token_size);
    state->buffer0_signals = (volatile int*)nvshmem_malloc(n_tokens * sizeof(int));
    
    // 检查分配是否成功
    if (!state->buffer0_send || !state->buffer0_recv || !state->buffer0_signals) {
        fprintf(stderr, "Error: Failed to allocate buffer0\n");
        exit(EXIT_FAILURE);
    }
    
    // 分配缓冲区1
    state->buffer1_send = (char*)nvshmem_malloc(n_tokens * token_size);
    state->buffer1_recv = (char*)nvshmem_malloc(n_tokens * token_size);
    state->buffer1_signals = (volatile int*)nvshmem_malloc(n_tokens * sizeof(int));
    
    if (!state->buffer1_send || !state->buffer1_recv || !state->buffer1_signals) {
        fprintf(stderr, "Error: Failed to allocate buffer1\n");
        exit(EXIT_FAILURE);
    }

    state->buffer_tmp_signals = (volatile int*)nvshmem_malloc(n_tokens * sizeof(int));
    
    if (!state->buffer_tmp_signals) {
        fprintf(stderr, "Error: Failed to allocate buffer_tmp_signals\n");
        exit(EXIT_FAILURE);
    }
    
    // 在主机上创建临时缓冲区并初始化
    int* host_signals = (int*)malloc(n_tokens * sizeof(int));
    if (!host_signals) {
        fprintf(stderr, "Error: Failed to allocate host_signals\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < n_tokens; ++i) {
        host_signals[i] = 0;
    }
    
    // 使用 cudaMemcpy 将数据从主机复制到设备
    cudaError_t err;
    
    err = cudaMemcpy((void*)state->buffer0_signals, host_signals, n_tokens * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy error (buffer0_signals): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy((void*)state->buffer1_signals, host_signals, n_tokens * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy error (buffer1_signals): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy((void*)state->buffer_tmp_signals, host_signals, n_tokens * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy error (buffer_tmp_signals): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // 释放主机内存
    free(host_signals);
    
    // 确保内存同步
    nvshmem_barrier_all();
}

void DoubleBufferManager::cleanup_double_buffer(DoubleBufferState* state) {
    nvshmem_free(state->buffer0_send);
    nvshmem_free(state->buffer0_recv);
    nvshmem_free((void*)state->buffer0_signals);
    
    nvshmem_free(state->buffer1_send);
    nvshmem_free(state->buffer1_recv);
    nvshmem_free((void*)state->buffer1_signals);

    nvshmem_free((void*)state->buffer_tmp_signals);
}

// 内核函数实现
__global__ void send_recv_clean_kernel(DoubleBufferState state, int dst_pe) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // 确定当前和下一个缓冲区
    int current_idx = state.current_buffer;
    int next_idx = 1 - current_idx;
    
    // 获取当前缓冲区指针
    char* send_buffer = (current_idx == 0) ? state.buffer0_send : state.buffer1_send;
    char* recv_buffer = (current_idx == 0) ? state.buffer0_recv : state.buffer1_recv;
    volatile int* signals = (current_idx == 0) ? state.buffer0_signals : state.buffer1_signals;
    
    // 获取下一个缓冲区指针（用于清理）
    volatile int* next_signals = (next_idx == 0) ? state.buffer0_signals : state.buffer1_signals;
    
    // 线程分组：前一半用于通信，后一半用于清理
    int comm_threads = total_threads / 2;
    int clean_threads = comm_threads;
    
    if (tid < comm_threads) {
        // 通信线程：执行发送和接收
        if (tid < state.n_tokens) {
            // 发送数据
            char* token_data_send = send_buffer + tid * state.token_size;
            char* token_data_recv = recv_buffer + tid * state.token_size;
            
            datacopy::nvshmemi_ibgda_put_nbi_thread(
                (uint64_t)token_data_recv, (uint64_t)token_data_send, state.token_size, dst_pe, tid, tid);
            
            // 发送信号
            int signal = tid + 1;  // 信号值 = token 索引 + 1
            datacopy::nvshmemi_ibgda_amo_nonfetch_add((void*)&signals[tid], signal, dst_pe, tid, false);
            
            // 轮询接收信号 - 使用直接访问而不是 __ldg
            while (signals[tid] != tid + 1);
        }
    } else {
        // 清理线程：清理下一个缓冲区
        int clean_tid = tid - comm_threads;
        for(int i = clean_tid; i < state.n_tokens; i += clean_threads) {
            next_signals[i] = 0; // 重置信号
        }
    }
}