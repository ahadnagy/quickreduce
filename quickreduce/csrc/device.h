#pragma once

#include <torch/torch.h>
#include "quickreduce.h"

/*
===============================================================
Device
*/
struct Device {
    hipStream_t stream;
    quickreduce::DeviceComms comms;

    Device() {}
    ~Device() {}
    void init(int world_size, int rank);
};

/*
===============================================================
API
*/
using comm_handle = std::array<uint8_t, HIP_IPC_HANDLE_SIZE>;

void init(int world_size, int rank);
int get_world_size();
int get_rank();
comm_handle get_comm_handle();
void set_comm_handles(std::vector<comm_handle> const& comm_handles);
torch::Tensor allreduce(int profile, torch::Tensor const& A);
void fused_gemm_ar(torch::Tensor const& A, torch::Tensor const& B,
              torch::Tensor& D, torch::Tensor& scale_tensor,
              size_t b_lanes, size_t split_k);
