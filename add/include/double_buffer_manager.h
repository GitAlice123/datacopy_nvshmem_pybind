// double_buffer_manager.h
#ifndef DOUBLE_BUFFER_MANAGER_H
#define DOUBLE_BUFFER_MANAGER_H

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <vector>

// 双缓冲状态结构体
struct DoubleBufferState {
    int n_tokens;            // token 总数
    int token_size;          // 每个 token 大小 (字节)
    volatile int* buffer_tmp_signals; // 临时信号缓冲区
    
    // 缓冲区0
    char* buffer0_send;      // 发送数据缓冲区0
    char* buffer0_recv;      // 接收数据缓冲区0
    volatile int* buffer0_signals; // 信号缓冲区0
    
    // 缓冲区1
    char* buffer1_send;      // 发送数据缓冲区1
    char* buffer1_recv;      // 接收数据缓冲区1
    volatile int* buffer1_signals; // 信号缓冲区1
    
    int current_buffer;      // 当前使用的缓冲区索引 (0 或 1)
};

class DoubleBufferManager {
public:
    DoubleBufferManager();
    ~DoubleBufferManager();
    
    void init(int n_tokens, int token_size);
    void cleanup();
    void test_bandwidth(int rank, int world_size);

private:
    void init_double_buffer(DoubleBufferState* state, int n_tokens, int token_size);
    void cleanup_double_buffer(DoubleBufferState* state);

    DoubleBufferState state;
    bool initialized;
    int n_tokens;
    int token_size;
};

#endif // DOUBLE_BUFFER_MANAGER_H
