#include "./consumer.h"
#include "./producer.h"
#include <core/allreduce.h>

using namespace quickreduce;

#define launch_tsr(BL, AP, BP, C, COMM, QS)                                                                                  \
    block.x = WARPSIZE * (AP + BP + C) + COMM;                                                                                \
    _tsr_kernel<BL, AP, BP, C, COMM, QS><<<grid, block, 0, stream>>>(A_, B_, D_, scale_tensor_, m, n, k, b_stride, split_k, rank, world_size, dbuffer_list, data_offset); \
    break;

template <int B_LANES, int A_PRODUCERS, int B_PRODUCERS, int CONSUMERS, int COMMS, int QSIZE>
void __global__ _tsr_kernel(const fp8* __restrict__ A, const fp8* __restrict__ B, half* __restrict__ D,
                            const float* scale_tensor, const int m, const int n, const int k, const int b_stride,
                            const int split_k, const int rank, const int world_size, uint8_t** __restrict__ comms_buffer_list, long const data_offset) {
    // Initialize shared queue
    __shared__ int queue[2 * B_LANES * QSIZE];
    if (threadIdx.x < 2 * B_LANES * QSIZE) {
        queue[threadIdx.x] = 0;
    }
    // Declare shared buffer
    __shared__ fp8 A_buffer[WARPTILE_M * WARPTILE_K * QSIZE];
    __shared__ fp8 B_buffer[(OP_N * B_LANES) * WARPTILE_K * QSIZE];
    __syncthreads();

    // Infer index and p-state
    int role_id;
    int index;
    int p_state;

    // A producer warp
    if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
        role_id = threadIdx.x / WARPSIZE;
        index = (OPS == 1 ? 2 : 1) * role_id;
        p_state = 0;
    }
    // B producer warp
    else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
        role_id = (threadIdx.x / WARPSIZE) - A_PRODUCERS;
        index = role_id;
        p_state = 0;
    }
    // Consumers warp
    else {
        role_id = (threadIdx.x / WARPSIZE) - (A_PRODUCERS + B_PRODUCERS);
        index = role_id;
        p_state = 1;
    }


    // Tiles loop
    int curr_n, curr_k, k_blocks, dropped_rows, dropped_cols;
    const int warptile_per_row = CDIV(n, (OP_N * B_LANES));
    const int tiles = warptile_per_row * split_k;
    const int tpw = max(CDIV(tiles, CU), 1);

    for (int warptile = (tpw * blockIdx.x); warptile < min(tiles, tpw * (blockIdx.x + 1)); warptile++) {
        // Compute tile position
        curr_n = (warptile % warptile_per_row) * (OP_N * B_LANES);
        curr_k = (warptile / warptile_per_row) * WARPTILE_K * K_BLOCKS(k, split_k);
        k_blocks = ((warptile / warptile_per_row) == (split_k - 1))
                       ? (k / WARPTILE_K) - (split_k - 1) * K_BLOCKS(k, split_k)
                       : K_BLOCKS(k, split_k);

        // Account for column overflow
        dropped_rows = max(0, 0 + WARPTILE_M - m);
        dropped_cols = max(0, curr_n + (OP_N * B_LANES) - n);
        curr_n -= dropped_cols;

        // A producer warp
        if (threadIdx.x < A_PRODUCERS * WARPSIZE) {
            _tsr_A_producer<A_PRODUCERS, B_LANES, QSIZE>(A + curr_k, &A_buffer[0], &queue[0], index, p_state, role_id,
                                                         dropped_rows, k, k_blocks);
        }
        // B producer warp
        else if (threadIdx.x < A_PRODUCERS * WARPSIZE + B_PRODUCERS * WARPSIZE) {
            _tsr_B_producer<B_PRODUCERS, B_LANES, QSIZE>(B + curr_n * b_stride + curr_k, &B_buffer[0], &queue[1], index,
                                                         p_state, role_id, b_stride, k_blocks);
        }
        // Consumers warp
        else if (threadIdx.x < (A_PRODUCERS + B_PRODUCERS + CONSUMERS) * WARPSIZE) {
            _tsr_consumer<CONSUMERS, B_LANES, QSIZE>(&A_buffer[0], &B_buffer[0], D + curr_n, scale_tensor[0], &queue[0],
                                                     index, p_state, role_id, n, dropped_rows, dropped_cols, k,
                                                     k_blocks);
        }
        asm volatile("s_waitcnt vmcnt(0)");
        __syncthreads();
        size_t tid = threadIdx.x - (A_PRODUCERS + B_PRODUCERS + CONSUMERS) * WARPSIZE;
        //if (tid >= 0 && tid < COMMS) {
        if (tid==0) {
            quickreduce::ReduceGatherSkinnyCodec<8> codec(0, rank);
            quickreduce::BufferResource d_buffer(D, m * n * sizeof(half));
            for (int row=0; row<8; row++) {
                static constexpr size_t kAtoms = ((16 * B_LANES) * sizeof(half)) / sizeof(int32x4_t);
                //printf("kAtoms %zu\n", kAtoms);
                int32x4_t tA[kAtoms];
            
                size_t chunk_bytes = (16 * B_LANES) * sizeof(half);
                size_t chunk_offset = (curr_n + row * n) * sizeof(half);
            
                //------------------------
                //Load from D to tA
                //------------------------
                int src_offset = chunk_offset;
                for (int i = 0; i < kAtoms; i++) {
                    tA[i] = buffer_load_dwordx4(d_buffer.descriptor, src_offset, 0, 0);
                    //tA[i] = {rank, rank, rank, rank};
                    src_offset += sizeof(int32x4_t);
                }

                long block = chunk_offset / ReduceGatherSkinnyCodec<8>::kTileSize;
                long comm_data0_offset = data_offset + block * ReduceGatherSkinnyCodec<8>::kTileSize;
                long tile_offset = chunk_offset % ReduceGatherSkinnyCodec<8>::kRankTileSize;

                static int constexpr kAtomStride = 256;

                //for (int r = 0; r < ReduceGatherSkinnyCodec<8>::kWorldSize; r++) {
                //    int32x4_t* send_buffer = reinterpret_cast<int32x4_t*>(comms_buffer_list[r] + comm_data0_offset + rank * ReduceGatherSkinnyCodec<8>::kRankTileSize + tile_offset);
                //    codec.send(send_buffer, &tA[r * ReduceGatherSkinnyCodec<8>::kRankAtoms]);
                //}

                for (int a = 0; a < kAtoms; a++) {
                    size_t target_rank = ((chunk_offset + a * sizeof(int32x4_t))/(256 * sizeof(int32x4_t)))%8;
                    
                    int32x4_t* send_buffer = reinterpret_cast<int32x4_t*>(comms_buffer_list[target_rank] + comm_data0_offset + rank * ReduceGatherSkinnyCodec<8>::kRankTileSize + tile_offset + a * sizeof(int32x4_t));
                    //if(rank==0) {
                    //    printf("Rank %d, tid %d, block %lx, comm_data0_offset %d, tile_offset %d, warptile %d, row %d, chunk_offset: %d, total_effset %d, target rank %d\n", rank, tid, block, comm_data0_offset, tile_offset, warptile, row, chunk_offset, comm_data0_offset + rank * ReduceGatherSkinnyCodec<8>::kRankTileSize + tile_offset, target_rank);
                    //}
                    codec.send(send_buffer, &tA[a * ReduceGatherSkinnyCodec<8>::kRankAtoms]);
                }
                
            }
        }
        __syncthreads();
    }
}