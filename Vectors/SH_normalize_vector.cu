#include <iostream>
#include <cmath>
#include <stdio.h>

#define CHECK_CUDA(call) {                                              \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d in %s: %s\n",              \
                __FILE__, __LINE__, #call, cudaGetErrorString(err));    \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

__global__ void reduce_sum_of_squares(float* vec, float* partial_sums, int N) {
    extern __shared__ float cache[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    cache[tid] = (idx < N) ? vec[idx] * vec[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* vcache = cache;
        vcache[tid] += vcache[tid + 32];
        vcache[tid] += vcache[tid + 16];
        vcache[tid] += vcache[tid + 8];
        vcache[tid] += vcache[tid + 4];
        vcache[tid] += vcache[tid + 2];
        vcache[tid] += vcache[tid + 1];
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = cache[0];
    }
}


__global__ void reduce_final(float* input, float* output, int N) {
    extern __shared__ float cache[];
    int tid = threadIdx.x;
    int idx = threadIdx.x;

    cache[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* vcache = cache;
        vcache[tid] += vcache[tid + 32];
        vcache[tid] += vcache[tid + 16];
        vcache[tid] += vcache[tid + 8];
        vcache[tid] += vcache[tid + 4];
        vcache[tid] += vcache[tid + 2];
        vcache[tid] += vcache[tid + 1];
    }

    if (tid == 0) {
        output[0] = cache[0];
    }
}


__global__ void apply_normalize(float* result, float* vec, float norm, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        result[idx] = vec[idx] / norm;
    }
}

int main() {
    int N;
    std::cin >> N;

    float* h_vec = new float[N];
    float* h_result = new float[N];
    for (int i = 0; i < N; i++) {
        h_vec[i] = i + 1;
    }

    float *d_vec, *d_result, *d_partial_sums, *d_sum_final;
    CHECK_CUDA(cudaMalloc(&d_vec, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, N * sizeof(float)));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    CHECK_CUDA(cudaMalloc(&d_partial_sums, numBlocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sum_final, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_vec, h_vec, N * sizeof(float), cudaMemcpyHostToDevice));


    reduce_sum_of_squares<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_vec, d_partial_sums, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    int finalBlockSize = 256; 
    reduce_final<<<1, finalBlockSize, finalBlockSize * sizeof(float)>>>(d_partial_sums, d_sum_final, numBlocks);
    CHECK_CUDA(cudaDeviceSynchronize());

    float host_sum;
    CHECK_CUDA(cudaMemcpy(&host_sum, d_sum_final, sizeof(float), cudaMemcpyDeviceToHost));

    float norm = sqrtf(host_sum);

    apply_normalize<<<numBlocks, blockSize>>>(d_result, d_vec, norm, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        std::cout << h_result[i] << "\n";
    }

    delete[] h_vec;
    delete[] h_result;
    cudaFree(d_vec);
    cudaFree(d_result);
    cudaFree(d_partial_sums);
    cudaFree(d_sum_final);

    return 0;
}
