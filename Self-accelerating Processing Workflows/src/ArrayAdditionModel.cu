#include <ArrayAdditionModel.h>
#include <kernels.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Constants.h>
#include <iostream>

#ifndef _ARRAYADDITIONMODEL_CPP_
#define _ARRAYADDITIONMODEL_CPP_

using namespace std;

template <class T>
ArrayAdditionModel<T>::ArrayAdditionModel(){}

template <class T>
ArrayAdditionModel<T>::ArrayAdditionModel(T *in1, T *in2, T *out, int length): localA(in1), localB(in2), localC(out), localL(length) { }

template <class T>
ArrayAdditionModel<T>::~ArrayAdditionModel() {}

template <class T>
void ArrayAdditionModel<T>::CPUImplementation(){
    for(int x = 0; x < localL; x++){
        //cout << localA[x] << "," << localB[x] << ",";
        localC[x] = localA[x] + localB[x];
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

    Vector_Addition <<< localL / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (dev_a, dev_b, dev_c);
    //Copy back to Host array from Device array
    cudaMemcpy(localC , dev_c , localL *sizeof(int) , cudaMemcpyDeviceToHost);
    //Free the Device array memory
    cudaFree (dev_a);
    cudaFree (dev_b);
    cudaFree (dev_c);
}

#endif // ARRAYADDITIONMODEL_CPP