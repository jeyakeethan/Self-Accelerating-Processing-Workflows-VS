#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void Vector_Addition(const int* dev_a, const int* dev_b, int* dev_c);
