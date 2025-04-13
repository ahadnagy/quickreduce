import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import quickreduce as qr
import numpy as np

from torch import Tensor
from typing import Tuple, Optional


from hf_rocm_kernels import skinny_gemm

def fp8_quantize(
    x_full_precision: Tensor,
    scale: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Quantizes a tensor (x_full_precision) according to a tensor-wise (scale) to float8_e4m3fnuz format. This function is
    meant to mimic the behavior of TGI and thus was inspired by it:
    https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/layers/fp8.py
    For reference on dtypes: https://onnx.ai/onnx/technical/float8.html
    """
    # Scale and clamp in full precision
    finfo = torch.finfo(torch.float8_e4m3fn)
    x_quantized = (x_full_precision * scale.reciprocal()).clamp(min=finfo.min, max=finfo.max)
    # Convert to float8_e4m3fn format, without removing signed zeros
    x_quantized = x_quantized.to(torch.float8_e4m3fn)
    # Remove signed zeros, which correspond to NaNs in float8_e4m3fnuz format
    weight_as_int8 = x_quantized.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    mask = weight_as_int8 == ROCM_FP8_NAN_AS_INT
    weight_as_int8[mask] = 0
    x_quantized = weight_as_int8.view(torch.float8_e4m3fnuz)
    # For the same bits representation, e4m3fnuz value is half of the e4m3fn value, so we should double the scaling
    # factor to get the same dequantized value.
    return x_quantized, scale * 2.0


def generate_skinny_gemm_data(
    m: int, n: int, k: int, seed: Optional[int] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generates random inputs for the skinny_gemm operation. The generated input's shape is determined by (m), (n) and
    (k), and one can pass a (seed) to ensure repeatability."""
    if seed is not None:
        torch.manual_seed(seed)
    scale_tensor = torch.ones(size=(1,), device="cuda", dtype=torch.float32).mul(2).add(1)
    skinny_a = fp8_quantize(
        #torch.ones(size=(m, k), device="cuda", dtype=torch.float32),
        torch.ones(size=(m, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0]
    b = fp8_quantize(
        torch.ones(size=(n, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0].t()
    output = torch.zeros(size=(m, n), dtype=torch.float16, device="cuda")
    return skinny_a, b, scale_tensor, output

def skinny_gemm_and_ar_pytorch(a, b, d, scale):
    # Perform GEMM
    skinny_gemm(
                skinny_a=a,
                b=b,
                scale_tensor=scale,
                output=d,
                split_k=1,
                b_lanes=5,
            )
    dist.all_reduce(d, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

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
    
    def fused_gemm_ar(self, skinny_a, b, out, scale_tensor, b_lanes, split_k):
        qr.fused_gemm_ar(skinny_a, b, out, scale_tensor, b_lanes, split_k)
        

def run_allreduce_comparison(rank, world_size):
    setup(rank, world_size)
    
    # Create CustomComms instance
    custom = CustomComms(world_size, rank)
    
    # Exchange comm handles using allgather
    local_handle = custom.get_comm_handle()
    handles = [None] * world_size
    dist.all_gather_object(handles, local_handle)
    custom.set_comm_handles(handles)
    
    profile = 1
    
    m = 8
    n = 16384
    k = 16384
    b_lanes = 5
    split_k = 1
    skinny_a, b, scale_tensor, out = generate_skinny_gemm_data(m, n, k, seed=0)
    # QuickReduce allreduce
    qr_out = out.clone()
    qr_result = custom.fused_gemm_ar(skinny_a, b, qr_out, scale_tensor, b_lanes, split_k)
    
    # PyTorch allreduce for comparison
    torch_result = out.clone()
    skinny_gemm_and_ar_pytorch(skinny_a, b, torch_result, scale_tensor)
    
    print(qr_out)
    
    # Verify results match
    if not torch.allclose(qr_out, torch_result, rtol=2.5e-1):
        print(f"Rank {rank}: QuickReduce (profile {profile}) result doesn't match PyTorch")
        print(f"QR: {qr_out[:10].cpu().numpy()}...")
        print(f"PyTorch: {torch_result[:10].cpu().numpy()}...")
    else:
        print(f"Rank {rank}: QuickReduce profile {profile} matches PyTorch allreduce")
    
    # Verify correctness (sum of ones should equal world_size)
    #expected = torch.ones(1024, dtype=torch.float16).cuda() * world_size
    #if not torch.allclose(qr_result, expected, rtol=1e-3):
    #    print(f"Rank {rank}: QuickReduce (profile {profile}) result incorrect")
    #else:
    #    print(f"Rank {rank}: QuickReduce profile {profile} produced correct result")

        
def main():
    world_size = 8
    mp.spawn(
        run_allreduce_comparison,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()