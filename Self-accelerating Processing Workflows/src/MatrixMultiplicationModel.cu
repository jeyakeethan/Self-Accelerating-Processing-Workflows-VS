#include <MatrixMultiplicationModel.h>
#include <kernels.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Constants.h>
#include <iostream>

#ifndef _MATRIXMULTIPLICATIONMODEL_CPP_
#define _MATRIXMULTIPLICATIONMODEL_CPP_

using namespace std;

template <class T>
MatrixMultiplicationModel<T>::MatrixMultiplicationModel(){}

template <class T>
MatrixMultiplicationModel<T>::MatrixMultiplicationModel(T *mat1, T *mat2, T *out, dim3 *matricesDim): localA(mat1), localB(mat2), localC(out), localMD(matricesDim) { }

template <class T>
MatrixMultiplicationModel<T>::~MatrixMultiplicationModel() {}

template <class T>
void MatrixMultiplicationModel<T>::CPUImplementation(){
    // implement using multi threads
}

template <class T>
void MatrixMultiplicationModel<T>::GPUImplementation(){
    //Device array
    int *dev_a , *dev_b, *dev_c;

    int l1 = localMD->x * localMD->y * sizeof(numericalType1));
    int l2 = localMD->y * localMD->z * sizeof(numericalType1));
    int l3 = localMD->x * localMD->z * sizeof(numericalType1));

    //Allocate the memory on the GPU
    cudaMalloc((void **)&dev_a , l1);
    cudaMalloc((void **)&dev_b , l2);
    cudaMalloc((void **)&dev_c , l3);

    //Copy Host array to Device array
    cudaMemcpy (dev_a , localA , l1, cudaMemcpyHostToDevice);
    cudaMemcpy (dev_b , localB , l2, cudaMemcpyHostToDevice);
    // Execute the kernel

    matrix_multiplication <<< ..... >>> (dev_a, dev_b, dev_c, localMD->x);
    //Copy back to Host array from Device array
    cudaMemcpy(localC , dev_c , l3, cudaMemcpyDeviceToHost);
    //Free the Device array memory
    cudaFree (dev_a);
    cudaFree (dev_b);
    cudaFree (dev_c);
}

#endif // _MATRIXMULTIPLICATIONMODEL_CPP_