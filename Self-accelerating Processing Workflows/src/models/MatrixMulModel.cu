#include <models/MatrixMulModel.h>
#include <kernels.h>
#include <omp.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Constants.h>
#include <iostream>

#ifndef _MATRIXMULTIPLICATIONMODEL_CPP_
#define _MATRIXMULTIPLICATIONMODEL_CPP_

using namespace std;

template <class T>
MatrixMultiplicationModel<T>::MatrixMultiplicationModel(int CPUCores):ComputationalModel(CPUCores, "matrix-multiplication") { }

template <class T>
MatrixMultiplicationModel<T>::~MatrixMultiplicationModel() {}


template <class T>
void MatrixMultiplicationModel<T>::CPUImplementation() {
	// log mode to see the flow of execution
	CPUGPULOG << 0;

	int x = localMD->x, y = localMD->y, z = localMD->z;
	
	//implement using multi threads
#pragma omp parallel num_threads(CPUCores)
		{
#pragma omp for
			for (int i = 0; i < x; i++) {
				for (int j = 0; j < z; j++) {
					T sum = 0;
					for (int k = 0; k < y; k++) {
						sum += localA[y * i + k] * localB[j + z * k];
					}
					localC[z * i + j] = sum;
				}
			}
		}
#pragma omp barrier
}

template <class T>
void MatrixMultiplicationModel<T>::GPUImplementation() {
	// log mode to see the flow of execution
	CPUGPULOG << 1;

	//Device array
	numericalType1 *dev_a, *dev_b, *dev_c;

	int l1 = localMD->x * localMD->y * sizeof(numericalType1);
	int l2 = localMD->y * localMD->z * sizeof(numericalType1);
	int l3 = localMD->x * localMD->z * sizeof(numericalType1);

	//Allocate the memory on the GPU
	cudaMalloc((void**)&dev_a, l1);
	cudaMalloc((void**)&dev_b, l2);
	cudaMalloc((void**)&dev_c, l3);

	//Copy Host array to Device array
	cudaMemcpy(dev_a, localA, l1, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, localB, l2, cudaMemcpyHostToDevice);
	// Execute the kernel
	// define grid and thread block sizes

	dim3 dimGrid(32, 1024), dimBlock(32);
	matrix_multiplication << < dimGrid, dimBlock >> > (dev_a, dev_b, dev_c, localMD->y, localMD->z);

	//Copy back to Host array from Device array
	cudaMemcpy(localC, dev_c, l3, cudaMemcpyDeviceToHost);

	//Free the Device array memory
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	//sychronize to confirm that results have been computed and copied back
	cudaDeviceSynchronize();
}


// retrive attributes
template <class T>
vector<float> MatrixMultiplicationModel<T>::getAttributes() {
	return *attr;
}

template <class T>
vector<float> MatrixMultiplicationModel<T>::getAttributesBatch() {
	return *attr;
}
#endif // _MATRIXMULTIPLICATIONMODEL_CPP_