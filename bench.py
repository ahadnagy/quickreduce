import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import quickreduce as qr
import numpy as np

from torch import Tensor
from typing import Tuple, Optional
import time

import matplotlib.pyplot as plt

from hf_rocm_kernels import skinny_gemm

import matplotlib.ticker as ticker


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

def generate_random_skinny_gemm_data(
    m: int, n: int, k: int, seed: Optional[int] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generates random inputs for the skinny_gemm operation. The generated input's shape is determined by (m), (n) and
    (k), and one can pass a (seed) to ensure repeatability."""
    if seed is not None:
        torch.manual_seed(seed)
    scale_tensor = torch.ones(size=(1,), device="cuda", dtype=torch.float32).mul(2).add(1)
    skinny_a = fp8_quantize(
        torch.normal(0, 1, size=(m, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0]
    b = fp8_quantize(
        torch.normal(0, 1, size=(n, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0].t()
    output = torch.zeros(size=(m, n), dtype=torch.float16, device="cuda")
    return skinny_a, b, scale_tensor, output

def generate_skinny_gemm_zeros(
    m: int, n: int, k: int, seed: Optional[int] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generates random inputs for the skinny_gemm operation. The generated input's shape is determined by (m), (n) and
    (k), and one can pass a (seed) to ensure repeatability."""
    if seed is not None:
        torch.manual_seed(seed)
    scale_tensor = torch.zeros(size=(1,), device="cuda", dtype=torch.float32).mul(2).add(1)
    skinny_a = fp8_quantize(
        #torch.ones(size=(m, k), device="cuda", dtype=torch.float32),
        torch.ones(size=(m, k), device="cuda", dtype=torch.float32),
        scale_tensor,
    )[0]
    b = fp8_quantize(
        torch.zeros(size=(n, k), device="cuda", dtype=torch.float32),
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
        qr.fused_gemm_ar(skinny_a, b, out, scale_tensor, b_lanes, split_k, False)
        

def benchmark_allreduce(rank, world_size, n_values, results_dict):
    try:
        setup(rank, world_size)
        
        custom = CustomComms(world_size, rank)
        local_handle = custom.get_comm_handle()
        handles = [None] * world_size
        dist.all_gather_object(handles, local_handle)
        custom.set_comm_handles(handles)

        b_lanes = 4
        split_k = 1
        m = 8
        #n = 16384
        k = 16384
        num_iters = 10

        local_results = {"custom": [], "torch": []}

        for n in n_values:
            skinny_a, b, scale_tensor, out = generate_skinny_gemm_zeros(m, n, k, seed=0)

            # WARMUP: Custom
            _ = out.clone()
            custom.fused_gemm_ar(skinny_a, b, _, scale_tensor, b_lanes, split_k)

            # Benchmark: Custom
            timings = []
            for _ in range(num_iters):
                qr_out = out.clone()
                torch.cuda.synchronize()
                start = time.perf_counter()
                custom.fused_gemm_ar(skinny_a, b, qr_out, scale_tensor, b_lanes, split_k)
                torch.cuda.synchronize()
                end = time.perf_counter()
                timings.append(end - start)
            avg_custom = sum(timings) / num_iters
            local_results["custom"].append(avg_custom)

            # WARMUP: Torch
            _ = out.clone()
            skinny_gemm_and_ar_pytorch(skinny_a, b, _, scale_tensor)

            # Benchmark: Torch
            timings = []
            for _ in range(num_iters):
                torch_out = out.clone()
                torch.cuda.synchronize()
                start = time.perf_counter()
                skinny_gemm_and_ar_pytorch(skinny_a, b, torch_out, scale_tensor)
                torch.cuda.synchronize()
                end = time.perf_counter()
                timings.append(end - start)
            avg_torch = sum(timings) / num_iters
            local_results["torch"].append(avg_torch)

            # Validate correctness (only once)
            torch.testing.assert_close(qr_out, torch_out, rtol=2.5e-5, atol=20)


        # Gather results to rank 0
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_results)

        if rank == 0:
            # Average results across all ranks
            for kind in ["custom", "torch"]:
                avg_times = [
                    sum(worker[kind][i] for worker in gathered) / world_size
                    for i in range(len(n_values))
                ]
                results_dict[kind] = avg_times
    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        raise
    finally:
        dist.destroy_process_group()

def main():
    world_size = 8
    n_values = [1024, 4096, 16384, 32768]
    manager = mp.Manager()
    results_dict = manager.dict()

    mp.spawn(
        benchmark_allreduce,
        args=(world_size, n_values, results_dict),
        nprocs=world_size,
        join=True
    )

    # Plotting (only after spawn joins)
    custom_times_us = [t * 1e6 for t in results_dict["custom"]]
    torch_times_us = [t * 1e6 for t in results_dict["torch"]]
    speedup = [torch / custom for torch, custom in zip(results_dict["torch"], results_dict["custom"])]

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xscale("log", base=2)
    ax1.plot(n_values, custom_times_us, label="Fused Skinny GEMM + QR", marker='o')
    ax1.plot(n_values, torch_times_us, label="Skinny GEMM + PyTorch AllReduce", marker='s')
    ax1.set_xlabel("m (skinny dimension)")
    ax1.set_ylabel("Time (µs)")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Secondary Y-axis for speedup
    ax2 = ax1.twinx()
    ax2.plot(n_values, speedup, label="Speedup (Torch / Fused)", color="black", linestyle="--", marker='^')
    ax2.set_ylabel("Speedup (x)")
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1fx'))

    # X-tick formatting
    ax1.set_xticks(n_values)
    ax1.set_xticklabels([str(m) for m in n_values])

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    plt.title("AllReduce GEMM Benchmark with Speedup")
    plt.tight_layout()
    plt.savefig("benchmark_with_speedup_us.png", dpi=300)

if __name__ == "__main__":
    main()