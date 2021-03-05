#include <BlurModel.h>
#include <kernels.h>
#include <omp.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Constants.h>
#include <iostream>

#ifndef _BLURMODEL_CPP_
#define _BLURMODEL_CPP_

using namespace std;

template <class T>
BlurModel<T>::BlurModel(int CPUCores):ComputationalModel(CPUCores) {
	//super(CPUCores);
}

template <class T>
BlurModel<T>::~BlurModel() {}


// retrive influenced attributes
template <class T>
vector<float>* BlurModel<T>::getAttributes() {
	return attr;
}

template <class T>
void BlurModel<T>::CPUImplementation() {
	// log mode to see the flow of execution
	CPUGPULOG << 0;

	/*	
	//implement using multi threads
#pragma omp parallel num_threads(CPUCores)
		{
#pragma omp for
			for (int i = 0; i < localMD->x; i++) {
				for (int j = 0; j < localMD->z; j++) {
					T sum = 0;
					for (int k = 0; k < localMD->y; k++) {
						sum += localA[localMD->y * i + k] * localB[j + localMD->z * k];
					}
					localC[localMD->z * i + j] = sum;
				}
			}
#pragma omp barrier
		}
		*/
}

template <class T>
void BlurModel<T>::GPUImplementation() {
	// log mode to see the flow of execution
	CPUGPULOG << 1;

	unsigned char* dev_input;
	unsigned char* dev_output;
	int size = width * height * 3;
	getError(cudaMalloc((void**)&dev_input, size * sizeof(unsigned char)));
	getError(cudaMemcpy(dev_input, input_image, size * sizeof(unsigned char), cudaMemcpyHostToDevice));

	getError(cudaMalloc((void**)&dev_output, size * sizeof(unsigned char)));

	dim3 blockDims(512, 1, 1);
	dim3 gridDims((unsigned int)ceil((double)(size / blockDims.x)), 1, 1);

	blur_image << <gridDims, blockDims >> > (dev_input, dev_output, width, height);


	getError(cudaMemcpy(output_image, dev_output, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	getError(cudaFree(dev_input));
	getError(cudaFree(dev_output));
}

template <class T>
void BlurModel<T>::getError(cudaError_t err) {
    if (err != cudaSuccess) {
        cout << "Error " << cudaGetErrorString(err) << endl;
    }
}
#endif // _BLURMODEL_CPP_