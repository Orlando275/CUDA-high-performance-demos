#include <iostream>
#include <stdio.h>
#include <cstdint>
using namespace std;

#define CHECK_CUDA(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", \
                __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

const int TILE_SIZE=16;

__global__ void matrix_multiplication(uint32_t* matrixA,uint32_t* matrixB,uint32_t* matrixTotal,int row,int col,int N){
    extern __shared__ uint32_t sh[];

    uint32_t* tileA=sh;
    uint32_t* tileB=sh+TILE_SIZE*TILE_SIZE;
    int bx=blockIdx.x, by=blockIdx.y;
    int tx=threadIdx.x,ty=threadIdx.y;
    int Row=by*TILE_SIZE+ty;
    int Col=bx*TILE_SIZE+tx;
    uint32_t sum=0;

    int numPhases=(N+TILE_SIZE-1)/TILE_SIZE;
    for (int ph = 0; ph < numPhases; ph++)
    {
        int aCol=ph *TILE_SIZE+tx;
        int bRow=ph*TILE_SIZE+ty;

        tileA[ty*TILE_SIZE+tx]=(Row<row && aCol<N)? matrixA[Row*N+aCol] : 0;
        tileB[ty*TILE_SIZE+tx]=(bRow<N && Col<col) ? matrixB[bRow*col+Col]:0;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum+= tileA[ty*TILE_SIZE+k]*tileB[k*TILE_SIZE+tx];
        }
        __syncthreads();
    }
    if (Row<row && Col<col)
    {
        matrixTotal[Row*col+Col]=sum;
    }
    








}


int main(){
    int N,row,col;
    cin>>row;
    cin>>N;
    cin>>col;
    int sizeTotal=row*col;
    uint32_t* arrA=new uint32_t[row*N];
    uint32_t* arrB=new uint32_t[N*col];
    uint32_t* arrTotal=new uint32_t[sizeTotal];
    

    uint32_t* d_A;
    uint32_t* d_B;
    uint32_t* d_Result;
    
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < N; j++)
        {
            arrA[i*N+j]=i%30;
        }
        
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < col; j++)
        {
            arrB[i*col+j]=i%20;
        }
        
    }

    CHECK_CUDA(cudaMalloc(&d_A,row*N*sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_B,col*N*sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_Result,sizeTotal*sizeof(uint32_t)));

    CHECK_CUDA(cudaMemcpy(d_A,arrA,row*N*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B,arrB,N*col*sizeof(uint32_t),cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE,TILE_SIZE);
    dim3 grid((col+TILE_SIZE-1)/TILE_SIZE,(row+TILE_SIZE-1)/TILE_SIZE);

    uint32_t sizeSH= 2 *TILE_SIZE* TILE_SIZE*sizeof(uint32_t);

    matrix_multiplication<<<grid,block,sizeSH>>>(d_A,d_B,d_Result,row,col,N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(arrTotal,d_Result,sizeTotal*sizeof(uint32_t),cudaMemcpyDeviceToHost));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << arrTotal[i * col + j] << " ";
        }
        std::cout << "\n";
    }

    // Liberar memoria
    delete[] arrA;
    delete[] arrB;
    delete[] arrTotal;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Result);


}