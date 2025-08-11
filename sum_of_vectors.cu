#include <iostream>
#include <stdio.h>
using namespace std;
#define CHECK_CUDA(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", \
                __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void sumVect(float* A,float* B,float* C,int N){
int idx=blockIdx.x*blockDim.x+threadIdx.x;
if (idx<N)
{
    C[idx]=A[idx]+B[idx];
}

}

int main(){
    int n;
    cin>>n;
    float* arrA=new float[n];
    float* arrB=new float[n];
    float* arrC=new float[n];

    for (int i = 0; i < n; i++)
    {
        *(arrA+i)=i+2;
        *(arrB+i)=i+2;
    }
    float *d_A,*d_B,*d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A,n*sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_B,n*sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_C,n*sizeof(float)));
CHECK_CUDA(cudaMemcpy(d_A, arrA, n * sizeof(float), cudaMemcpyHostToDevice));
CHECK_CUDA(cudaMemcpy(d_B, arrB, n * sizeof(float), cudaMemcpyHostToDevice));

int blockSize=64;
int numBlock=(n+blockSize-1)/blockSize;
sumVect<<<numBlock,blockSize>>>(d_A,d_B,d_C,n);

CHECK_CUDA(cudaDeviceSynchronize());
CHECK_CUDA(cudaMemcpy(arrC,d_C,n*sizeof(float),cudaMemcpyDeviceToHost));
for (int i = 0; i < n; i++)
{
    cout<<arrC[i]<<"\n";
}
delete[]arrA;
delete[]arrB;
delete[]arrC;
CHECK_CUDA(cudaFree(d_A));
CHECK_CUDA(cudaFree(d_B));
CHECK_CUDA(cudaFree(d_C));
return 0;
}