#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void matrix_multiplication(float* C, float* A, float* B, const int widthA, const int widthB);
__global__ void Vector_Addition(const int* dev_a, const int* dev_b, int* dev_c);
__global__ void dot_product(int* a, int* b, int* res);
