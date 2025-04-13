import torch
from .device import (
    init,
    get_world_size,
    get_rank,
    get_comm_handle,
    set_comm_handles,
    allreduce,
    fused_gemm_ar,
)
