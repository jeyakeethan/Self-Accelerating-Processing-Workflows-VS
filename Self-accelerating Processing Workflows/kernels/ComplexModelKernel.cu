#include <kernels.h>
#include <Constants.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif // !(__CUDACC_RTC__)

#include <device_functions.h>

__global__ void complex_model_kernel(numericalType1* A, numericalType1* B,  numericalType1* X, numericalType1* OUT, const int widthA, const int widthB) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x < widthB) {
        numericalType1 sum = 0;

        for (int k = 0; k < widthA; k++) {
            sum += A[y * widthA + k] * B[k * widthB + x];
        }

        OUT[y * widthB + x] = sum;
    }
    __syncthreads();

    // add mat X to the out matrix (matrix addition)
    OUT += X[y * widthA + x];
    return;
}