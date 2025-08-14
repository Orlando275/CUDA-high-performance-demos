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

__global__ void mult_matrix(unsigned long long* matrixA,unsigned long long* matrixB,unsigned long long* matrixResult,int N,int rows,int cols){
   
    int tid=threadIdx.x;
    int idx=blockIdx.x*blockDim.x+tid;
    int totalSize=rows*cols;

    if (idx<(totalSize)){
        int Row=idx/cols;
        int Col=idx%cols;
        unsigned long long sum=0;
        for (int i = 0; i < N; i++)
        {
            sum+=matrixA[Row*N+i]*matrixB[i*cols+Col];
           
        }
        matrixResult[Row*cols+Col]=sum;
    }
    







}

int main(){
    int N,row,col;
    cin>>row;
    cin>>N;
    cin>>col;
    int sizeTotal=row*col;
    unsigned long long* arrA=new unsigned long long[row*N];
    unsigned long long* arrB=new unsigned long long[N*col];
    unsigned long long* arrTotal=new unsigned long long[sizeTotal];
    
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < N; j++)
        {
            arrA[i*N+j]=i*2;
        }
        
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < col; j++)
        {
            arrB[i*col+j]=i;
        }
        
    }

    unsigned long long* D_A;
    unsigned long long* D_B;
    unsigned long long* D_Total;
    //design the place to save and the size of array
    CHECK_CUDA(cudaMalloc((void**)&D_A,row*N*sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc((void**)&D_B,N*col*sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc((void**)&D_Total,sizeTotal*sizeof(unsigned long long)));
    
    //then we need send what we need to save on device because we've already design the size of arrays on cudamalloc
    CHECK_CUDA(cudaMemcpy(D_A,arrA,row*N*sizeof(unsigned long long),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D_B,arrB,N*col*sizeof(unsigned long long),cudaMemcpyHostToDevice));

    int blocksize=512;
    int numBlock=(sizeTotal+blocksize-1)/blocksize;
    //blocksize*sizeof(float) tamaño en bytes de la memoria compartida dinámica que se asignará a cache.
    mult_matrix<<<numBlock,blocksize>>>(D_A,D_B,D_Total,N,row,col);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(arrTotal,D_Total,sizeTotal*sizeof(unsigned long long),cudaMemcpyDeviceToHost));

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            cout<<arrTotal[i*col+j]<< (j+1==col ? "" : " ");
        }
         cout << '\n';
    }
    
    delete[] arrA;
    delete[] arrB;
    delete[] arrTotal;
    cudaFree(D_A);
    cudaFree(D_B);
    cudaFree(D_Total);

}