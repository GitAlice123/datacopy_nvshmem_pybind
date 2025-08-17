import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from nvshmem_extension import nvshmem_init_py, nvshmem_get_info

def run(rank, world_size):
    print(f"[Rank {rank}] Starting...")
    
    # 初始化分布式环境
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://29.225.97.56:61007',
        world_size=world_size,
        rank=rank
    )
    
    # 设置当前 GPU 设备
    torch.cuda.set_device(rank)
    
    # 初始化 NVSHMEM
    try:
        nvshmem_init_py()
        print(f"[Rank {rank}] NVSHMEM initialized successfully")
    except Exception as e:
        print(f"[Rank {rank}] NVSHMEM init failed: {str(e)}")
        return
    
    # 获取并打印信息
    try:
        nvshmem_get_info()
    except Exception as e:
        print(f"[Rank {rank}] NVSHMEM get_info failed: {str(e)}")
    
    # 清理
    dist.destroy_process_group()
    print(f"[Rank {rank}] Finished")

if __name__ == "__main__":
    # 配置环境变量
    import os
    os.environ['MASTER_ADDR'] = '29.225.97.56'
    os.environ['MASTER_PORT'] = '61007'
    
    # 设置世界大小 (2 个节点)
    world_size = 2
    
    # 启动分布式进程
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)