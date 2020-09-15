#include <kernels.h>
#include <Constants.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void matrix_multiplication(float* C, float* A, float* B, const int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0;

    for (int k = 0; k < width; k++) {
        sum += A[y * width + k] * B[k * width + x];
    }

    C[y * width + x] = sum;
}