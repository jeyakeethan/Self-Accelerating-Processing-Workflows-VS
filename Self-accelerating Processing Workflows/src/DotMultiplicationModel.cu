#include <DotMultiplicationModel.h>
#include <kernels.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Constants.h>
#include <stdio.h>

DotMultiplicationModel::DotMultiplicationModel() {}
DotMultiplicationModel::DotMultiplicationModel(int *in1, int *in2, int *out, int length): localA(in1), localB(in2), localC(out), localL(length) { }
DotMultiplicationModel::~DotMultiplicationModel() {}

void DotMultiplicationModel::CPUImplementation(){
    printf("Hello CPU IMPL \n");
    long temp = 0;
    for (int i = 0; i < localL; i++)
    {
        temp += localA[i] * localB[i];
    }
    printf("\n%d\n", temp);
}
void DotMultiplicationModel::GPUImplementation(){
    printf("Hello GPU IMPL \n");

    // Allocate memory for arrays d_A, d_B, and d_result on device
    int* d_A, * d_B, * d_result;
    size_t bytes = localL * sizeof(int);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_result, sizeof(int));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, localA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, localB, bytes, cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //    thr_per_blk: number of CUDA threads per grid block
    //    blk_in_grid: number of blocks in grid
    int thr_per_blk = THREADS_PER_BLOCK;
    int blk_in_grid = ceil(float(N) / thr_per_blk);

    // Launch kernel
    dot_product <<< blk_in_grid, thr_per_blk >>> (d_A, d_B, d_result);
    printf("%d", d_result);
    // copy back to host
    cudaMemcpy(localC, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
}
