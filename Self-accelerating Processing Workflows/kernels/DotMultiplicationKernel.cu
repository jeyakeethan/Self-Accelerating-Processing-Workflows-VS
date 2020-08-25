#include <kernels.h>
#include <Constants.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void dot_product(int* a, int* b, int* res)
{
	__shared__ int products[THREADS_PER_BLOCK];

	int id = blockDim.x * blockIdx.x + threadIdx.x;
	products[threadIdx.x] = a[id] * b[id];
	__syncthreads();
	if (threadIdx.x == 0)
	{
		int sum_of_products = 0;
		for (int i = 0; i < THREADS_PER_BLOCK; i++)
		{
			sum_of_products = sum_of_products + products[i];
		}
		atomicAdd(res, sum_of_products);
	}
}