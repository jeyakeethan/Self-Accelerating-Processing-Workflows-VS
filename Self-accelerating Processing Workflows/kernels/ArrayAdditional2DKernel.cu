#include <kernels.h>
#include <Constants.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void Vector_Addition2D(const int* dev_a, const int* dev_b, int* dev_c)
{
    //Get the id of thread within a block
    //unsigned short tid = threadIdx.x;
    int i = blockIdx.x * gridDim.x + threadIdx.x;

    //if (tid < THREADS_PER_BLOCK) // check the boundry condition for the threads
    dev_c[i] = dev_a[i] + dev_b[i];
}