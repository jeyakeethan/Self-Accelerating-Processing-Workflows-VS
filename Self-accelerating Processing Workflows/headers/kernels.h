#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Constants.h>

__global__ void matrix_multiplication(numericalType1* C, numericalType1* A, numericalType1* B, const int widthA, const int widthB);
__global__ void Vector_Addition(const int* dev_a, const int* dev_b, int* dev_c);
__global__ void dot_product(int* a, int* b, int* res);
