#include <iostream>
#include <stdio.h>
using namespace std;
#define CHECK_CUDA(call)                                                    \
    {                                                                       \
        cudaError_t err=(call);                                             \
        if (err!=cudaSuccess)                                               \
        {                                                                   \
            fprintf(stderr,"CUDA ERROR at %s:%d in %s: %s\n",__FILE__,      \
            __LINE__,#call, cudaGetErrorString(err));                       \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
                                                                            \
    }                                                                       \





__global__ void sum_parcial (int *arrA,int* arrR,int N){
    extern __shared__ int cache[];
    int tid = threadIdx.x;
    int idx = blockDim.x*blockIdx.x +tid;
    cache[tid] = (idx < N) ? arrA[idx] : 0 ;
    __syncthreads();
    
    for (int i = blockDim.x/2; i >= 32; i >>= 1)
    {
        if (tid < i)
        {
            cache[tid] += cache[tid+i];
        }
        __syncthreads();
    }
    
    int val=cache[tid];
    if (tid < 32 )
    {
        for (int i = 16; i > 0; i >>= 1)
        {
            val += __shfl_down_sync(0xffffffff, val , i );
        }
        
    }
    if (tid == 0)
    {
        arrR[blockIdx.x] = val;
    }
    
    
    

}

__global__ void sum_total (int* arrA, int *result,int N){
    extern __shared__ int cache[];
    int tid= threadIdx.x;
    int idx = blockDim.x * blockIdx.x +tid;

    cache[tid] = (idx < N) ? arrA[idx] : 0 ;
    __syncthreads();

    for (int i = blockDim.x/2 ; i >= 32; i >>= 1)
    {
        if (tid < i)
        {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
    }

    int val=cache[tid];
    if (tid < 32)
    {
        for (int i = 16; i > 0; i >>= 1)
        {
            val +=__shfl_down_sync(0xffffffff, val , i);
        }
        
    }
    
    if (tid==0)
    {
        *result=val;
    }
    
    
}



int main(){
    int N = 2<<24;
    int* arrA=new int[N];
    int arrR=0;
    for (int i = 0; i < N; i++)
    {
        *(arrA +i)=i;
    }
    int* D_A;
    int* D_R;
    int* D_Result;

    int blockSize=32*22;
    int numBlock = (N+blockSize-1) / blockSize;

    CHECK_CUDA(cudaMalloc((void**)&D_A,N*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&D_R,numBlock*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&D_Result,sizeof(int)));

    CHECK_CUDA(cudaMemcpy(D_A,arrA,N*sizeof(int),cudaMemcpyHostToDevice));



    
    sum_parcial<<<numBlock,blockSize,blockSize*sizeof(int)>>>(D_A,D_R,N);
    CHECK_CUDA(cudaDeviceSynchronize());

    

    int numThreads=min(1024,numBlock);
    sum_total<<<1,numThreads,numThreads * sizeof(int)>>>(D_R,D_Result,numBlock);
    CHECK_CUDA(cudaDeviceSynchronize());

 
    CHECK_CUDA(cudaMemcpy(&arrR,D_Result,sizeof(int),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(D_A));
    CHECK_CUDA(cudaFree(D_R));
    delete []arrA;
    return 0;
}