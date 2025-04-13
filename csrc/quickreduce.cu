#include <hip/hip_runtime.h>
#include <core/allreduce.h>
#include "quickreduce.h"
#include <ATen/cuda/CUDAContext.h>

#include "skinny_gemm/skinny_gemm.h"

namespace quickreduce {

// ============================================================
// CONTEXT
// ============================================================
void DeviceComms::init(int world_size, int rank) {
    destroy();
    this->world_size = world_size;
    this->rank = rank;

    // Allocate buffer size for worst case: Twoshot FP16 2-stage buffer.
    long flags_buffer_size = 2 * world_size * kMaxTiles * sizeof(int);
    long data_buffer_size = 2 * kMaxProblemSize;
    long total_buffer_size = flags_buffer_size + data_buffer_size;
    data_offset = flags_buffer_size;
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    HIP_CHECK(hipThreadExchangeStreamCaptureMode(&mode));
    HIP_CHECK(hipExtMallocWithFlags((void**)&dbuffer, total_buffer_size, hipDeviceMallocUncached));


    // Clear the flags buffer.
    hipMemsetAsync(dbuffer, 0, flags_buffer_size, stream);

    // Device-side list of IPC buffers.
    buffer_list.resize(world_size);
    hipMalloc(&dbuffer_list, world_size * sizeof(uint8_t*));
    HIP_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

    // Allocate device-side flags buffer.
    cudaMalloc(&dflag_color, sizeof(int));

    // Create IPC handles for rank's communication buffer.
    all_buffer_ipc_handles.resize(world_size);
    hipIpcGetMemHandle(&buffer_ipc_handle, dbuffer);

    initialized = true;
}

void DeviceComms::destroy() {
    if (initialized) {
        for (int i = 0; i < world_size; i++) {
            if (i != rank) {
                hipIpcCloseMemHandle(dbuffer_list[i]);
            }
        }

        hipFree(dbuffer);
        hipFree(dbuffer_list);

        initialized = false;
    }
}

void DeviceComms::open_ipc_handles(std::vector<hipIpcMemHandle_t> const& ipc_handles) {
    for (int i = 0; i < world_size; i++) {
        all_buffer_ipc_handles[i] = ipc_handles[i];
    }

    // Open device memory access to the IPC communication buffers.
    // Note: For our own rank, we do not need to open a handle.
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            hipIpcOpenMemHandle((void**)&buffer_list[i], all_buffer_ipc_handles[i], hipIpcMemLazyEnablePeerAccess);
        } else {
            buffer_list[i] = dbuffer;
        }
    }

    hipMemcpy(dbuffer_list, buffer_list.data(), world_size * sizeof(uint8_t*), hipMemcpyHostToDevice);
}

// ============================================================
// KERNEL
// ============================================================
template <typename AllReduceKenel>
__global__ __quickreduce_launch_bounds__
static void allreduce_prototype(half const* A, half* B, int N, int num_blocks,
        int world_size, int rank, uint8_t** dbuffer_list, long data_offset, int *flag_color, bool capturing) {

    int block = blockIdx.x;
    int grid = gridDim.x;

    while (block < num_blocks) {
        AllReduceKenel::run(A, B, N, block, num_blocks, world_size, rank, dbuffer_list, data_offset, flag_color, capturing);
        block += grid;
    }
}

__global__ void incrementKernel(int* d_flag_color) {
    // Only thread 0 increments the counter per invocation
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(d_flag_color, 1);
    }
}

// ============================================================
// DISPATCH
// ============================================================
#define TWOSHOT_DISPATCH(__codec)                                               \
    if (world_size == 2) {                                                      \
        using LineCodec = __codec<2>;                                           \
        using AllReduceKernel = AllReduceTwoshot<LineCodec>;                    \
        hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>),              \
            dim3(grid), dim3(kBlock), 0, stream,                                \
            A, B, N, num_blocks, world_size, rank, dbuffer_list,                \
            data_offset, dflag_color, capturing);                                          \
    }                                                                           \
    else if (world_size == 4) {                                                 \
        using LineCodec = __codec<4>;                                           \
        using AllReduceKernel = AllReduceTwoshot<LineCodec>;                    \
        hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>),              \
            dim3(grid), dim3(kBlock), 0, stream,                                \
            A, B, N, num_blocks, world_size, rank, dbuffer_list,                \
            data_offset, dflag_color, capturing);                                          \
    }                                                                           \
    else if (world_size == 8) {                                                 \
        using LineCodec = __codec<8>;                                           \
        using AllReduceKernel = AllReduceTwoshot<LineCodec>;                    \
        hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>),              \
            dim3(grid), dim3(kBlock), 0, stream,                                \
            A, B, N, num_blocks, world_size, rank, dbuffer_list,                \
            data_offset, dflag_color, capturing);                                          \
    }

void DeviceComms::allreduce(int profile, hipStream_t stream, half const* A, half* B, int N) {
    if (world_size != 2 && world_size != 4 && world_size != 8) {
        throw std::runtime_error("All Reduce not supported for world_size = " + std::to_string(world_size));
    }

    // Configuration.
    long msg_size = N * sizeof(half);
    int num_blocks = divceil(msg_size, kTileSize);
    int grid = min(304 * 4, num_blocks);

    bool capturing = false;

    // -------------------------------------------------
    // All reduce dispatch.
    QuickReduceProfile dprofile = static_cast<QuickReduceProfile>(profile);

    cudaMemcpyAsync(dflag_color, &flag_color, sizeof(int), cudaMemcpyHostToDevice, stream);

    switch (dprofile) {
        case QuickReduceProfile::TWOSHOT_FP8:
            TWOSHOT_DISPATCH(TwoshotFP8LineCodec)
            break;
        case QuickReduceProfile::TWOSHOT_Q8:
            TWOSHOT_DISPATCH(TwoshotQ8LineCodec)
            break;
        case QuickReduceProfile::TWOSHOT_Q6:
            TWOSHOT_DISPATCH(TwoshotQ6LineCodec)
            break;
        case QuickReduceProfile::TWOSHOT_Q4:
            TWOSHOT_DISPATCH(TwoshotQ4LineCodec)
            break;
        default:
            TWOSHOT_DISPATCH(TwoshotFP16LineCodec)
            break;
    }

    // -------------------------------------------------
    // Rotate the flag color.
    incrementKernel<<<1, 1, 0, stream>>>(dflag_color);
    cudaMemcpyAsync(&flag_color, dflag_color, sizeof(int), cudaMemcpyDeviceToHost, stream);
}

void DeviceComms::fused_gemm_ar(torch::Tensor const& A, torch::Tensor const& B, torch::Tensor& D, torch::Tensor& scale_tensor,
    size_t b_lanes, size_t split_k, hipStream_t stream, bool capturing) {
        //skinny_gemm(A, B, D, scale_tensor, b_lanes, split_k, this->rank, this->world_size, stream);
        const int m = A.size(0);
        const int n = B.size(1);
        const int k = A.size(1);
        const int b_stride = B.stride(1);
    
        const fp8* __restrict__ A_ = (const fp8* __restrict__)A.data_ptr();
        const fp8* __restrict__ B_ = (const fp8* __restrict__)B.data_ptr();
        half* __restrict__ D_ = (half* __restrict__)D.data_ptr();
        float* __restrict__ scale_tensor_ = (float* __restrict__)scale_tensor.data_ptr();
    
        // Check shape
        if (m > WARPTILE_M) {
            std::cerr << "m = " << k << " is greater than WARPTILE_M = " << WARPTILE_M << std::endl;
            exit(1);
        }
        if (k % WARPTILE_K != 0) {
            std::cerr << "k = " << k << " is not divisible by WARPTILE_K = " << WARPTILE_K << std::endl;
            exit(1);
        }
    
        // Prepare kernel launch
        dim3 grid(CU, 1, 1);
        dim3 block(1, 1, 1);
    
        // Launch kernel (branched on B_LANES)
        switch (b_lanes) {
            case 2:
                launch_tsr(2, 3, 8, 4, 8, 5);
            case 3:
                launch_tsr(3, 3, 5, 2, 8, 4);  // Perforamnce on MI300: 8_13312_16384:57.54
            case 4:
                launch_tsr(4, 2, 6, 3, 8, 3);  // Perforamnce on MI300: 8_16384_6656:29.5
            case 5:
                launch_tsr(5, 2, 6, 2, 8, 2);
            default:
                break;
        }

        if (world_size != 2 && world_size != 4 && world_size != 8) {
            throw std::runtime_error("All Reduce not supported for world_size = " + std::to_string(world_size));
        }
    
        // Configuration.
        long msg_size = D.numel() * sizeof(half);
        int num_blocks = divceil(msg_size, kTileSize);
        int ar_grid = min(304 * 4, num_blocks);

        // using LineCodec = TwoshotFP16LineCodec<8>;
        // using AllReduceKernel = AllReduceTwoshot<LineCodec>;
        // hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>),
        //     dim3(ar_grid), dim3(kBlock), 0, stream,
        //     D_, D_, D.numel(), num_blocks, world_size, rank, dbuffer_list,
        //     data_offset, flag_color);

        cudaMemcpyAsync(dflag_color, &flag_color, sizeof(int), cudaMemcpyHostToDevice, stream);
        using LineCodec = TwoshotFP16LineCodec<8>;
        using AllReduceKernel = ReduceGather<LineCodec>;
        allreduce_prototype<AllReduceKernel><<<dim3(ar_grid), dim3(kBlock), 0, stream>>>(D_, D_, D.numel(), num_blocks, world_size, rank, dbuffer_list,
        data_offset, dflag_color, capturing);

        incrementKernel<<<1, 1, 0, stream>>>(dflag_color);
        cudaMemcpyAsync(&flag_color, dflag_color, sizeof(int), cudaMemcpyDeviceToHost, stream);
        return;
    }


// void DeviceComms::fused_gemm_ar(torch::Tensor const& A, torch::Tensor const& B, torch::Tensor& D, torch::Tensor& scale_tensor,
//     size_t b_lanes, size_t split_k, hipStream_t stream) {
//         //skinny_gemm(A, B, D, scale_tensor, b_lanes, split_k, this->rank, this->world_size, stream);
//         const int m = A.size(0);
//         const int n = B.size(1);
//         const int k = A.size(1);
//         const int b_stride = B.stride(1);
    
//         const fp8* __restrict__ A_ = (const fp8* __restrict__)A.data_ptr();
//         const fp8* __restrict__ B_ = (const fp8* __restrict__)B.data_ptr();
//         half* __restrict__ D_ = (half* __restrict__)D.data_ptr();
//         float* __restrict__ scale_tensor_ = (float* __restrict__)scale_tensor.data_ptr();
    
//         // Check shape
//         if (m > WARPTILE_M) {
//             std::cerr << "m = " << k << " is greater than WARPTILE_M = " << WARPTILE_M << std::endl;
//             exit(1);
//         }
//         if (k % WARPTILE_K != 0) {
//             std::cerr << "k = " << k << " is not divisible by WARPTILE_K = " << WARPTILE_K << std::endl;
//             exit(1);
//         }

//         cudaStream_t graph_stream;
//         cudaStreamCreate(&graph_stream);

//         // Begin capturing a graph
//         cudaGraph_t graph;
//         cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal);
    
//         // Prepare kernel launch
//         dim3 grid(CU, 1, 1);
//         dim3 block(1, 1, 1);
    
//         // Launch kernel (branched on B_LANES)
//         switch (b_lanes) {
//             case 2:
//                 launch_tsr(2, 3, 8, 4, 8, 5);
//             case 3:
//                 launch_tsr(3, 3, 5, 2, 8, 4);  // Perforamnce on MI300: 8_13312_16384:57.54
//             case 4:
//                 launch_tsr(4, 2, 6, 3, 8, 3);  // Perforamnce on MI300: 8_16384_6656:29.5
//             case 5:
//                 launch_tsr(5, 2, 6, 2, 8, 2);
//             default:
//                 break;
//         }

//         if (world_size != 2 && world_size != 4 && world_size != 8) {
//             throw std::runtime_error("All Reduce not supported for world_size = " + std::to_string(world_size));
//         }
    
//         // Configuration.
//         long msg_size = D.numel() * sizeof(half);
//         int num_blocks = divceil(msg_size, kTileSize);
//         int ar_grid = min(304 * 4, num_blocks);

//         // using LineCodec = TwoshotFP16LineCodec<8>;
//         // using AllReduceKernel = AllReduceTwoshot<LineCodec>;
//         // hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>),
//         //     dim3(ar_grid), dim3(kBlock), 0, stream,
//         //     D_, D_, D.numel(), num_blocks, world_size, rank, dbuffer_list,
//         //     data_offset, flag_color);

//         cudaMemcpyAsync(dflag_color, &flag_color, sizeof(int), cudaMemcpyHostToDevice, graph_stream);
//         using LineCodec = TwoshotFP16LineCodec<8>;
//         using AllReduceKernel = ReduceGather<LineCodec>;
//         //hipLaunchKernelGGL((allreduce_prototype<AllReduceKernel>),
//         //    dim3(ar_grid), dim3(kBlock), 0, stream,
//         //    D_, D_, D.numel(), num_blocks, world_size, rank, dbuffer_list,
//         //    data_offset, dflag_color);
//         allreduce_prototype<AllReduceKernel><<<dim3(ar_grid), dim3(kBlock), 0, graph_stream>>>(D_, D_, D.numel(), num_blocks, world_size, rank, dbuffer_list,
//         data_offset, dflag_color);

//         incrementKernel<<<1, 1, 0, graph_stream>>>(dflag_color);
//         cudaMemcpyAsync(&flag_color, dflag_color, sizeof(int), cudaMemcpyDeviceToHost, graph_stream);

//         hipStreamCaptureStatus status;
//         cudaStreamIsCapturing(graph_stream, &status);


//         // End capture
//         cudaStreamEndCapture(graph_stream, &graph);

//         // Instantiate the graph
//         cudaGraphExec_t graphExec;
//         cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

//         // Replay the graph (can be done multiple times)
//         for (int i = 0; i < 20; ++i) {
//             cudaGraphLaunch(graphExec, graph_stream);
//             cudaStreamSynchronize(graph_stream);
//         }

//         cudaGraphExecDestroy(graphExec);
//         cudaGraphDestroy(graph);
//         cudaStreamDestroy(graph_stream);

//         printf("Flag color: %d, capturing: %d\n", flag_color, status == cudaStreamCaptureStatusActive);

//         return;
//     }
}  // namespace quickreduce
