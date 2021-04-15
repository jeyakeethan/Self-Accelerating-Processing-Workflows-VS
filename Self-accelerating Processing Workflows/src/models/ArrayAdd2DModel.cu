#include <models/ArrayAdd2DModel.h>
#include <kernels.h>
#include <omp.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Constants.h>
#include <iostream>

#ifndef _ARRAYADD2DMODEL_CPP_
#define _ARRAYADD2DMODEL_CPP_

using namespace std;

template <class T>
ArrayAddition2DModel<T>::ArrayAddition2DModel(int CPUCores): ComputationalModel(CPUCores, "Array-Addition2D") {
}

template <class T>
ArrayAddition2DModel<T>::~ArrayAddition2DModel() {}

template <class T>
void ArrayAddition2DModel<T>::CPUImplementation() {
#pragma omp parallel num_threads(CPUCores)
    {
#pragma omp for
        for (int x = 0; x < localRow; x++) {
            for (int y = 0; x < localCol; y++) {
                //cout << localA[x] << "," << localB[x] << ",";
                int m = x * y;
                localC[m] = localA[m] + localB[m];
            }
#pragma omp barrier
        }
    }
}

template <class T>
void ArrayAddition2DModel<T>::GPUImplementation(){
    //Device array
    int *dev_a , *dev_b, *dev_c;
    int localSize = localRow * localCol;
    //Allocate the memory on the GPU
    cudaMalloc((void **)&dev_a , localSize *sizeof(int));
    cudaMalloc((void **)&dev_b , localSize *sizeof(int));
    cudaMalloc((void **)&dev_c , localSize *sizeof(int));
    //Copy Host array to Device array
    cudaMemcpy (dev_a , localA , localSize *sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy (dev_b , localB , localSize *sizeof(int) , cudaMemcpyHostToDevice);
    // Execute the kernel

    Vector_Addition2D <<< (localSize / THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (dev_a, dev_b, dev_c);
    //Copy back to Host array from Device array
    cudaMemcpy(localC , dev_c , localSize *sizeof(int) , cudaMemcpyDeviceToHost);
    //Free the Device array memory
    cudaFree (dev_a);
    cudaFree (dev_b);
    cudaFree (dev_c);
}

template <class T>
vector<float>* ArrayAddition2DModel<T>::getAttributes(){
    return new vector<float>{ (float)localRow, (float)localCol};
}

#endif //ARRAYADD2DMODEL_CPP