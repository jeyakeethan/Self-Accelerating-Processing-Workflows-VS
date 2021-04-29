#include <models/ComplexModel.h>
#include <kernels.h>
#include <omp.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Constants.h>
#include <iostream>

#ifndef _COMPLEX_MODEL_CPP_
#define _COMPLEX_MODEL_CPP_

using namespace std;

template <class T>
ComplexModel<T>::ComplexModel(int CPUCores) :ComputationalModel(CPUCores, "complex-model") {
	//super(CPUCores);
}

template <class T>
ComplexModel<T>::~ComplexModel() {}


template <class T>
void ComplexModel<T>::CPUImplementation() {
	// log mode to see the flow of execution
	CPUGPULOG << 0;
	int x = localMD->x, y = localMD->y, z = localMD->z;
	int length_b = y * z;
// ADD Y to the B first
#pragma omp parallel num_threads(CPUCores)
	{
#pragma omp for
		for (int x = 0; x < length_b; x++) {
			localB[x] = localB[x] + localY[x];
		}
#pragma omp barrier
	}

// Multiply B and A then add X
#pragma omp parallel num_threads(CPUCores)
	{
#pragma omp for
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < z; j++) {
				T sum = 0;
				for (int k = 0; k < y; k++) {
					sum += localA[y * i + k] * localB[j + z * k];
				}
				int index = z * i + j;
				localC[index] = sum + localX[index];
			}
		}
	}
}

template <class T>
void ComplexModel<T>::GPUImplementation() {
	// log mode to see the flow of execution
	CPUGPULOG << 1;

	//Device array
	numericalType1* dev_a, * dev_b, * dev_y, * dev_out, * dev_x;
	int x = localMD->x, y = localMD->y, z = localMD->z;
	int l1 = x * y * sizeof(numericalType1);
	int l2 = y * z * sizeof(numericalType1);
	int l3 = x * z * sizeof(numericalType1);

	//Allocate the memory on the GPU
	cudaMalloc((void**)&dev_a, l1);
	cudaMalloc((void**)&dev_b, l2);
	cudaMalloc((void**)&dev_y, l2);
	cudaMalloc((void**)&dev_out, l3);
	cudaMalloc((void**)&dev_x, l3);

	//Copy Host array to Device array
	cudaMemcpy(dev_a, localA, l1, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, localB, l2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, localX, l2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, localX, l3, cudaMemcpyHostToDevice);

	// Execute the kernel
	// Array addition kernel
	dim3 blockDims(THREADS_PER_BLOCK, 1, 1);
	dim3 gridDims((unsigned int)ceil((double)(l2 / blockDims.x)), 1, 1);
	Vector_Addition << < blockDims, gridDims >> > (dev_y, dev_b, dev_b);

	cudaDeviceSynchronize();		//sychronize

	// Complex Kernel
	dim3 dimGrid(32, 1024), dimBlock(32);
	complex_model_kernel << < dimGrid, dimBlock >> > (dev_a, dev_b, dev_x, dev_out, y, z);

	//Copy back to Host array from Device array
	cudaMemcpy(localC, dev_out, l3, cudaMemcpyDeviceToHost);

	//Free the Device array memory
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_out);

	//sychronize to confirm that results have been computed and copied back
	cudaDeviceSynchronize();
}


// retrive attributes
template <class T>
vector<float>* ComplexModel<T>::getAttributes() {
	return attr;
}

template <class T>
vector<float>* ComplexModel<T>::getAttributesBatch() {
	return attr;
}
#endif // _COMPLEX_MODEL_CPP_