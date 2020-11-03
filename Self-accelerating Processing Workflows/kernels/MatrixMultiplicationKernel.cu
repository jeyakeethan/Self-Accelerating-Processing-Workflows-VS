#include <kernels.h>
#include <Constants.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void matrix_multiplication(numericalType1* A, numericalType1* B, numericalType1* C, const int widthA, const int widthB) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x < widthB) {
        numericalType1 sum = 0;

        for (int k = 0; k < widthA; k++) {
            sum += A[y * widthA + k] * B[k * widthB + x];
        }

        C[y * widthB + x] = sum;
    }
    return;
}