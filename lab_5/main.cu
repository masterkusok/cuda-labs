#include <iostream>
#include <vector>
#include <cstdio>
#include <cuda_runtime.h>

#define MAX_VAL (1 << 24)
#define BLOCK_SIZE 512
#define LOG_NUM_BANKS 5

#define CSC(call) do { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(res), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__device__ inline int bankOffset(int idx) {
    return idx >> LOG_NUM_BANKS;
}

__global__ void kernel_histogram(const int* input,
                                 int n,
                                 unsigned int* counts) {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int pos = thread_id; pos < n; pos += step) {
        int value = input[pos];

        if (value >= 0 && value < MAX_VAL) {
            atomicAdd(counts + value, 1);
        }
    }
}

__global__ void kernel_scan(unsigned int* data,
                            unsigned int* block_sums,
                            int n) {

    __shared__ unsigned int shared_data[
        BLOCK_SIZE * 2 +
        ((BLOCK_SIZE * 2) >> LOG_NUM_BANKS)
    ];

    int tid = threadIdx.x;
    int block_offset = 2 * blockIdx.x * BLOCK_SIZE;

    int left = tid;
    int right = tid + BLOCK_SIZE;

    int left_bank = left + bankOffset(left);
    int right_bank = right + bankOffset(right);

    int global_left = block_offset + left;
    int global_right = block_offset + right;

    shared_data[left_bank] =
        (global_left < n) ? data[global_left] : 0;

    shared_data[right_bank] =
        (global_right < n) ? data[global_right] : 0;

    int offset = 1;

    for (int stride = BLOCK_SIZE; stride > 0; stride >>= 1) {

        __syncthreads();

        if (tid < stride) {

            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            int ai_bank = ai + bankOffset(ai);
            int bi_bank = bi + bankOffset(bi);

            shared_data[bi_bank] += shared_data[ai_bank];
        }

        offset <<= 1;
    }

    if (tid == 0) {

        int last = BLOCK_SIZE * 2 - 1;
        last += bankOffset(last);

        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = shared_data[last];
        }

        shared_data[last] = 0;
    }

    for (int stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {

        offset >>= 1;

        __syncthreads();

        if (tid < stride) {

            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            int ai_bank = ai + bankOffset(ai);
            int bi_bank = bi + bankOffset(bi);

            unsigned int temp = shared_data[ai_bank];

            shared_data[ai_bank] = shared_data[bi_bank];
            shared_data[bi_bank] += temp;
        }
    }

    __syncthreads();

    if (global_left < n) {
        data[global_left] = shared_data[left_bank];
    }

    if (global_right < n) {
        data[global_right] = shared_data[right_bank];
    }
}

__global__ void kernel_add_block_sums(unsigned int* data,
                                      const unsigned int* block_sums,
                                      int n) {

    int base = blockIdx.x * BLOCK_SIZE * 2;
    unsigned int add_value = block_sums[blockIdx.x];

    int first = base + threadIdx.x;
    int second = first + BLOCK_SIZE;

    if (first < n) {
        data[first] += add_value;
    }

    if (second < n) {
        data[second] += add_value;
    }
}

void recursive_scan(unsigned int* d_data, int n) {

    if (n <= 0) {
        return;
    }

    int blocks =
        (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    unsigned int* d_block_sums = nullptr;

    if (blocks > 1) {
        CSC(cudaMalloc(&d_block_sums,
                       blocks * sizeof(unsigned int)));
    }

    kernel_scan<<<blocks, BLOCK_SIZE>>>(
        d_data,
        d_block_sums,
        n
    );

    if (blocks > 1) {

        recursive_scan(d_block_sums, blocks);

        kernel_add_block_sums<<<blocks, BLOCK_SIZE>>>(
            d_data,
            d_block_sums,
            n
        );

        CSC(cudaFree(d_block_sums));
    }
}

__global__ void kernel_reconstruct(const unsigned int* prefix_sums,
                                   const unsigned int* counts,
                                   int* output,
                                   int total_n) {

    int value =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (value >= MAX_VAL) {
        return;
    }

    unsigned int start = prefix_sums[value];
    unsigned int count = counts[value];

    for (unsigned int i = 0; i < count; ++i) {

        unsigned int pos = start + i;

        if (pos < total_n) {
            output[pos] = value;
        }
    }
}

int main() {

    int n;

    if (fread(&n, sizeof(int), 1, stdin) != 1) {
        return 0;
    }

    if (n <= 0) {
        return 0;
    }

    std::vector<int> h_input(n);

    fread(h_input.data(),
          sizeof(int),
          n,
          stdin);

    int* d_input = nullptr;
    int* d_output = nullptr;

    unsigned int* d_counts = nullptr;
    unsigned int* d_prefix_sums = nullptr;

    CSC(cudaMalloc(&d_input, n * sizeof(int)));
    CSC(cudaMalloc(&d_output, n * sizeof(int)));

    CSC(cudaMalloc(&d_counts,
                   MAX_VAL * sizeof(unsigned int)));

    CSC(cudaMalloc(&d_prefix_sums,
                   MAX_VAL * sizeof(unsigned int)));

    CSC(cudaMemcpy(d_input,
                   h_input.data(),
                   n * sizeof(int),
                   cudaMemcpyHostToDevice));

    CSC(cudaMemset(d_counts,
                   0,
                   MAX_VAL * sizeof(unsigned int)));

    kernel_histogram<<<512, 512>>>(
        d_input,
        n,
        d_counts
    );

    CSC(cudaGetLastError());

    CSC(cudaMemcpy(d_prefix_sums,
                   d_counts,
                   MAX_VAL * sizeof(unsigned int),
                   cudaMemcpyDeviceToDevice));

    recursive_scan(d_prefix_sums, MAX_VAL);

    kernel_reconstruct<<<
        (MAX_VAL + 255) / 256,
        256
    >>>(
        d_prefix_sums,
        d_counts,
        d_output,
        n
    );

    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    std::vector<int> h_output(n);

    CSC(cudaMemcpy(h_output.data(),
                   d_output,
                   n * sizeof(int),
                   cudaMemcpyDeviceToHost));

    fwrite(h_output.data(),
           sizeof(int),
           n,
           stdout);

    CSC(cudaFree(d_input));
    CSC(cudaFree(d_output));
    CSC(cudaFree(d_counts));
    CSC(cudaFree(d_prefix_sums));

    return 0;
}