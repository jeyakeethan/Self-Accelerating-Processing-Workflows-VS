#include <kernels.h>
#include <Constants.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void Vector_Addition(const int* dev_a, const int* dev_b, int* dev_c)
{
    //Get the id of thread within a block
    unsigned short tid = blockIdx.x * blockDim.x + threadIdx.x;

        dev_c[tid] = dev_a[tid] + dev_b[tid];
}