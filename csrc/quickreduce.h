#pragma once

#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>

#define HIP_CHECK(err)                                                              \
    do {                                                                            \
        hipError_t err_ = (err);                                                    \
        if (err_ != hipSuccess) {                                                   \
            std::printf("HIP error %d at %s:%d\n", err_, __FILE__, __LINE__);       \
            throw std::runtime_error("HIP error");                                  \
        }                                                                           \
    } while (0)


namespace quickreduce {

enum struct QuickReduceProfile {
    TWOSHOT_FP16 = 1,
    TWOSHOT_FP8 = 2,
    TWOSHOT_Q8 = 3,
    TWOSHOT_Q6 = 4,
    TWOSHOT_Q4 = 5
};

/*
===============================================================
Desc:
    Device Comms Handle
*/
struct DeviceComms {
    // Workgroup scope = Tile = (256 threads x 16B x 8 atoms)
    static long constexpr kTileSize = 256 * 16 * 8;

    // Max problem size is 512MB (in bytes)
    static long constexpr kMaxProblemSize = 536870912;
    static long constexpr kMaxTiles = kMaxProblemSize / kTileSize;

    // Max TP-8
    static int constexpr kMaxWorldSize = 8;

    bool initialized = false;
    int flag_color = 1;
    int world_size;
    int rank;

    uint8_t* dbuffer;
    uint8_t** dbuffer_list;
    hipIpcMemHandle_t buffer_ipc_handle;
    std::vector<hipIpcMemHandle_t> all_buffer_ipc_handles;
    std::vector<uint8_t*> buffer_list;
    long data_offset;

    DeviceComms() : initialized(false), world_size(1), rank(0) {}
    ~DeviceComms() { destroy(); }

    void init(int world_size, int rank);
    int get_world_size() { return world_size; }
    int get_rank() { return rank; }
    bool status() { return initialized; }
    void destroy();

    hipIpcMemHandle_t const get_handle() { return buffer_ipc_handle; }
    void open_ipc_handles(std::vector<hipIpcMemHandle_t> const& ipc_handles);
    void allreduce(int profile, hipStream_t stream, half const* A, half* B, int N);
    void fused_gemm_ar(torch::Tensor const& A, torch::Tensor const& B, torch::Tensor& D, torch::Tensor& scale_tensor,
        size_t b_lanes, size_t split_k, hipStream_t stream);
};

}  // namespace quickreduce