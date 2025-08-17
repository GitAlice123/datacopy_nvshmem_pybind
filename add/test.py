import torch
import torch.distributed as dist
import os
import sys
import pydatacopy
import numpy as np

def init_dist(local_rank: int, num_local_ranks: int):
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank
    )
    
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)
    
    return dist.get_rank(), dist.get_world_size()

def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py [local_rank] [num_qps_per_rank]")
        sys.exit(1)
    
    local_rank = int(sys.argv[1])
    num_qps_per_rank = int(sys.argv[2])
    rank, world_size = init_dist(local_rank, 1)
    group = dist.new_group(list(range(world_size)))

    os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'
    os.environ['NVSHMEM_DISABLE_P2P'] = '1'
    os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
    os.environ['NVSHMEM_IBGDA_NIC_HANDLER'] = 'gpu'
    os.environ['NVSHMEM_QP_DEPTH'] = '1024'
    os.environ['NVSHMEM_CUMEM_GRANULARITY'] = f'{2 ** 29}'
    
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
    
    # 执行跨设备通信测试
    pydatacopy.cross_device_test(rank, world_size, root_unique_id)
    
    # 清理分布式环境
    dist.destroy_process_group()

if __name__ == "__main__":
    main()