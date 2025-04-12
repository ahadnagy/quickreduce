import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import quickreduce as qr
import numpy as np

def setup(rank, world_size):
    # Initialize PyTorch distributed
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12345',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

class CustomComms:
    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        qr.init(world_size, rank)

    def get_comm_handle(self):
        return qr.get_comm_handle()

    def set_comm_handles(self, comm_handles):
        qr.set_comm_handles(comm_handles)

    def allreduce(self, profile, tensor):
        #tensor = torch.ones(1024, dtype=torch.float16).cuda()
        result = qr.allreduce(profile, tensor)
        return result

def run_allreduce_comparison(rank, world_size):
    setup(rank, world_size)
    
    # Create CustomComms instance
    custom = CustomComms(world_size, rank)
    
    # Exchange comm handles using allgather
    local_handle = custom.get_comm_handle()
    handles = [None] * world_size
    dist.all_gather_object(handles, local_handle)
    custom.set_comm_handles(handles)
    
    # Test quickreduce allreduce
    for profile in [1, 2, 3, 4, 5]:  # Test all profiles
        torch_tensor = torch.rand(1024, dtype=torch.float16).cuda()
        # QuickReduce allreduce
        qr_out = torch_tensor.clone()
        qr_result = custom.allreduce(profile, qr_out)
        
        # PyTorch allreduce for comparison
        torch_tensor = torch_tensor.clone()
        dist.all_reduce(torch_tensor, op=dist.ReduceOp.SUM)
        
        # Verify results match
        if not torch.allclose(qr_result, torch_tensor, rtol=2.5e-1):
            print(f"Rank {rank}: QuickReduce (profile {profile}) result doesn't match PyTorch")
            print(f"QR: {qr_result[:10].cpu().numpy()}...")
            print(f"PyTorch: {torch_tensor[:10].cpu().numpy()}...")
        else:
            print(f"Rank {rank}: QuickReduce profile {profile} matches PyTorch allreduce")
        
        # Verify correctness (sum of ones should equal world_size)
        expected = torch.ones(1024, dtype=torch.float16).cuda() * world_size
        if not torch.allclose(qr_result, expected, rtol=1e-3):
            print(f"Rank {rank}: QuickReduce (profile {profile}) result incorrect")
        else:
            print(f"Rank {rank}: QuickReduce profile {profile} produced correct result")

def main():
    world_size = 4
    mp.spawn(
        run_allreduce_comparison,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()