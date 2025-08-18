// double_buffer_manager.h
#ifndef DOUBLE_BUFFER_MANAGER_H
#define DOUBLE_BUFFER_MANAGER_H

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <vector>

struct DoubleBufferState {
    int n_tokens;
    int token_size;
    volatile int* buffer_tmp_signals;
    
    char* buffer0_send;
    char* buffer0_recv;
    volatile int* buffer0_signals;
    
    char* buffer1_send;
    char* buffer1_recv;
    volatile int* buffer1_signals;
    
    int current_buffer;
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
