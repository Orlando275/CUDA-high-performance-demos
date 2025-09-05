#include <iostream>
#include <stdio.h>
#include <cmath>
#define CHECK_CUDA(call)                                                 \
    {                                                                    \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess)                                          \
        {                                                                \
            fprintf(stderr, "CUDA error at %s:%d in %s: %s\n",           \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }
__global__ void normalize_Vector_Sharememory(float *Vector, float *sumT, int N)
{
    extern __shared__ float cache[];
    int tid = threadIdx.x;

    int idx = blockDim.x * blockIdx.x + tid;
    cache[tid] = (idx < N) ? Vector[idx] * Vector[idx] : 0.0f;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 32; i >>= 1)
    {
        if (tid < i)
        {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
    }

    float val = cache[tid];
    if (tid < 32)
    {
        for (int i = 16; i > 0; i >>= 1)
        {
            val += __shfl_down_sync(0xffffffff, val, i);
        }
    }
    if (tid == 0)
    {
        sumT[blockIdx.x] = val;
    }
}

__global__ void result_Total_Parcial_Sum(float *parcialSum, float *sumTotal, int N)
{
    extern __shared__ float cache[];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    cache[tid] = (idx < N) ? parcialSum[idx] : 0.0f;

    __syncthreads();

    for (int i = blockDim.x / 2; i > 32; i >>= 1)
    {
        if (tid < i)
        {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
    }
    float val = cache[tid];
    if (tid < 32)
    {
        for (int i = 16; i > 0; i >>= 1)
        {
            val += __shfl_down_sync(0xffffffff, val, i);
        }
    }
    if (tid == 0)
    {
        sumTotal[0] = val;
    }
}

__global__ void aplly_Normalize(float *vectorResult, float *vecInit, float raizX, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        vectorResult[idx] = vecInit[idx] / raizX;
    }
}

int main()
{
    int N = 2 << 24;
    float sumCuadrados = 0;

    float *arrA = new float[N];
    float *arrResult = new float[N];
    for (int i = 0; i < N; i++)
    {
        *(arrA + i) = i;
    }
    float *D_A;
    float *D_S;
    float *D_Result;

    int blockSize = 32*32;
    int numBlock = (N + blockSize - 1) / blockSize;

    CHECK_CUDA(cudaMalloc((void **)&D_A, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&D_S, numBlock * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&D_Result, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(D_A, arrA, N * sizeof(float), cudaMemcpyHostToDevice));

    normalize_Vector_Sharememory<<<numBlock, blockSize, blockSize * sizeof(float)>>>(D_A, D_S, N);
    CHECK_CUDA(cudaDeviceSynchronize());


    int threadsFinal = min(256, numBlock);

    result_Total_Parcial_Sum<<<1, threadsFinal, threadsFinal * sizeof(float)>>>(D_S, D_Result, numBlock);
    CHECK_CUDA(cudaDeviceSynchronize());


    CHECK_CUDA(cudaMemcpy(&sumCuadrados, D_Result, sizeof(float), cudaMemcpyDeviceToHost));
    float raizX = sqrt((sumCuadrados));

    aplly_Normalize<<<numBlock, blockSize>>>(D_Result, D_A, raizX, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(arrResult, D_Result, N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(D_A));
    CHECK_CUDA(cudaFree(D_S));
    CHECK_CUDA(cudaFree(D_Result));
    delete[] arrA;
    delete[] arrResult;
    return 0;
}
