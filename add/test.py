import torch
import torch.distributed as dist
import os
import sys
import pydatacopy
import numpy as np

def init_dist(local_rank: int, num_local_ranks: int):
    # 设置分布式环境参数
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

def main():
    # 获取命令行参数
    if len(sys.argv) < 2:
        print("Usage: python test.py [local_rank]")
        sys.exit(1)
    
    local_rank = int(sys.argv[1])
    
    # 初始化分布式环境
    rank, world_size = init_dist(local_rank, 1)  # 每个节点只有一个GPU
    
    # 设置模式（低延迟模式或标准模式）
    low_latency_mode = True  # 默认使用低延迟模式
    
    # 创建通信组
    group = dist.new_group(list(range(world_size)))
    
    # 同步使用根ID
    nvshmem_unique_ids = [None] * world_size
    
    # 确定根节点
    if low_latency_mode:
        # 低延迟模式：全局rank0作为根节点
        if rank == 0:
            root_unique_id = pydatacopy.get_unique_id()
        else:
            root_unique_id = None
    else:
        # 标准模式：RDMA rank0作为根节点
        # 这里简化处理，假设每个节点只有一个GPU，所以RDMA rank0就是全局rank0
        if rank == 0:
            root_unique_id = pydatacopy.get_unique_id()
        else:
            root_unique_id = None
    
    # 收集所有节点的唯一标识符
    dist.all_gather_object(nvshmem_unique_ids, root_unique_id, group)
    
    # 选择最终的根标识符
    if low_latency_mode:
        # 低延迟模式：使用全局rank0的标识符
        root_unique_id = nvshmem_unique_ids[0]
    else:
        # 标准模式：使用RDMA根节点的标识符
        # 这里简化处理，假设RDMA根节点是全局rank0
        root_unique_id = nvshmem_unique_ids[0]
    
    # 打印调试信息
    if root_unique_id is not None:
        print(f"Rank {rank}: Using unique ID of length {len(root_unique_id)}")
    else:
        print(f"Rank {rank}: Failed to get unique ID")
        dist.destroy_process_group()
        sys.exit(1)
    
    # 执行跨设备通信测试
    pydatacopy.cross_device_test(rank, world_size, root_unique_id)
    
    # 清理分布式环境
    dist.destroy_process_group()

if __name__ == "__main__":
    main()