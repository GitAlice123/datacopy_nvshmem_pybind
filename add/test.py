import os
import sys
import time
import torch
import torch.distributed as dist
import numpy as np
import pydatacopy

def init_dist(local_rank: int, num_local_ranks: int):
    """初始化分布式环境"""
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    
    # 初始化分布式环境
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank
    )
    
    # 设置默认设备和数据类型
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)
    
    return dist.get_rank(), dist.get_world_size()

def bench(fn, num_warmups=20, num_tests=30, post_fn=None):
    """基准测试函数，测量函数执行时间"""
    # 刷新 L2 缓存
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    
    # 预热
    for _ in range(num_warmups):
        # 预热时也同步，确保状态一致
        dist.all_reduce(torch.tensor(0, device='cuda'))
        fn()
    
    # 刷新缓存
    cache.zero_()
    torch.cuda.synchronize()
    
    # 测试
    times = []
    for _ in range(num_tests):
        # 在计时前同步
        dist.all_reduce(torch.tensor(0, device='cuda'))
        torch.cuda.synchronize()  # 确保同步完成

        start_time = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
        if post_fn is not None:
            post_fn()
    
    # 计算统计信息
    times = np.array(times)
    avg_t = np.mean(times)
    min_t = np.min(times)
    max_t = np.max(times)
    
    return avg_t, min_t, max_t

def main():
    if len(sys.argv) < 5:
        print("Usage: python test.py [local_rank] [num_qps_per_rank] [n_tokens] [token_size]")
        sys.exit(1)
    
    local_rank = int(sys.argv[1])
    num_qps_per_rank = int(sys.argv[2])
    n_tokens = int(sys.argv[3])
    token_size = int(sys.argv[4])
    
    # 初始化分布式环境
    rank, world_size = init_dist(local_rank, 1)
    group = dist.new_group(list(range(world_size)))

    # 设置 NVSHMEM 环境变量
    os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'
    os.environ['NVSHMEM_DISABLE_P2P'] = '1'
    os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
    os.environ['NVSHMEM_IBGDA_NIC_HANDLER'] = 'gpu'
    os.environ['NVSHMEM_QP_DEPTH'] = '1024'
    os.environ['NVSHMEM_CUMEM_GRANULARITY'] = f'{2 ** 29}'
    
    # 广播唯一标识符
    nvshmem_unique_ids = [None] * world_size
    
    if rank == 0:
        root_unique_id = pydatacopy.get_unique_id()
    else:
        root_unique_id = None
    
    dist.all_gather_object(nvshmem_unique_ids, root_unique_id, group)
    
    root_unique_id = nvshmem_unique_ids[0]
    
    if root_unique_id is not None:
        print(f"Rank {rank}: Using unique ID of length {len(root_unique_id)}")
    else:
        print(f"Rank {rank}: Failed to get unique ID")
        dist.destroy_process_group()
        sys.exit(1)
    
    # 初始化 NVSHMEM
    my_pe = pydatacopy.init_nvshmem(root_unique_id, rank, world_size)
    print(f"[Rank {rank}] Initialized as PE {my_pe}")
    
    try:
        # 创建双缓冲管理器
        buffer_manager = pydatacopy.DoubleBufferManager()
        
        # 初始化缓冲区（不计入带宽测试时间）
        buffer_manager.init(n_tokens, token_size)
        
        # 定义测试函数
        def test_func():
            # 在测试开始前同步所有GPU
            dist.all_reduce(torch.tensor(0, device='cuda'))
            buffer_manager.test_bandwidth(rank, world_size)
        
        # 运行基准测试
        avg_t, min_t, max_t = bench(test_func)
        
        # 计算带宽
        total_bytes = n_tokens * token_size * 2  # 双向通信
        bandwidth_GBps = total_bytes / (1024**3 * avg_t)  # GB/s
        
        # 打印结果
        print(f'[Rank {rank}] Bandwidth: {bandwidth_GBps:.2f} GB/s, '
              f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)

    finally:
        # 清理资源
        buffer_manager.cleanup()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()