#include <models/ArrayAddModel.h>
#include <kernels.h>
#include <omp.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Constants.h>
#include <iostream>

#ifndef _ARRAYADDMODEL_CPP_
#define _ARRAYADDMODEL_CPP_

using namespace std;

template <class T>
ArrayAdditionModel<T>::ArrayAdditionModel(int CPUCores): ComputationalModel(CPUCores, "Array-Addition") {
}

template <class T>
ArrayAdditionModel<T>::~ArrayAdditionModel() {}

template <class T>
void ArrayAdditionModel<T>::CPUImplementation(){
#pragma omp parallel num_threads(CPUCores)
    {
#pragma omp for
    for(int x = 0; x < localL; x++){
        //cout << localA[x] << "," << localB[x] << ",";
        localC[x] = localA[x] + localB[x];
    }
#pragma omp barrier
    }
}

template <class T>
void ArrayAdditionModel<T>::GPUImplementation(){
    //Device array
    int *dev_a , *dev_b, *dev_c;
    //Allocate the memory on the GPU
    cudaMalloc((void **)&dev_a , localL *sizeof(int));
    cudaMalloc((void **)&dev_b , localL *sizeof(int));
    cudaMalloc((void **)&dev_c , localL *sizeof(int));
    //Copy Host array to Device array
    cudaMemcpy (dev_a , localA , localL *sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy (dev_b , localB , localL *sizeof(int) , cudaMemcpyHostToDevice);
    // Execute the kernel

    dim3 blockDims(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDims((unsigned int)ceil((double)(localL / blockDims.x)), 1, 1);
    Vector_Addition << < blockDims, gridDims >> > (dev_a, dev_b, dev_c);
    //Copy back to Host array from Device array
    cudaMemcpy(localC , dev_c , localL *sizeof(int) , cudaMemcpyDeviceToHost);
    //Free the Device array memory
    cudaFree (dev_a);
    cudaFree (dev_b);
    cudaFree (dev_c);
}

template <class T>
vector<float>* ArrayAdditionModel<T>::getAttributes(){
    return new vector<float>{ 1, float(localL) };
}

#endif //ARRAYADDMODEL_CPP