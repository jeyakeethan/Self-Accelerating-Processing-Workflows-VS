#include <MatrixMultiplicationModel.h>
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
MatrixMultiplicationModel<T>::MatrixMultiplicationModel(int CPUCores){
    super(CPUCores);
}

template <class T>
MatrixMultiplicationModel<T>::~MatrixMultiplicationModel() {}

template <class T>
void MatrixMultiplicationModel<T>::CPUImplementation(){
    // implement using multi threads
    int no_rows_per_thread = localMD->y / CPUCores;
#pragma omp parallel num_threads(CPUCores)
    threadMatMult(localA, localB, localC, localMD, no_rows_per_thread);
}

template <class T>
void threadMatMult(T *a, T *b, T *out, myDim3 *matD, int no_rows) {
    long my_rank = omp_get_thread_num();
    // long no_threads = omp_get_num_threads();
    // long no_rows = mat->y / no_threads;

    int my_first_row = my_rank * no_rows;
    int my_last_row = (my_rank+1) * no_rows-1;

    int i, j, k;
    for (i = my_first_row; i < my_last_row; i++) {
        for (j = 0; j < matD->y; j++) {
            out[matD->y * i + j] = 0;
            for (k = 0; k < matD->y; k++)
                out[matD->y*i+j] += a[matD->y * i+k] * b[j+ matD->z * k];
        }
    }
    return;
}

template <class T>
void MatrixMultiplicationModel<T>::GPUImplementation(){
    //Device array
    int *dev_a , *dev_b, *dev_c;

    int l1 = localMD->x * localMD->y * sizeof(numericalType1);
    int l2 = localMD->y * localMD->z * sizeof(numericalType1);
    int l3 = localMD->x * localMD->z * sizeof(numericalType1);

    //Allocate the memory on the GPU
    cudaMalloc((void **)&dev_a , l1);
    cudaMalloc((void **)&dev_b , l2);
    cudaMalloc((void **)&dev_c , l3);

    //Copy Host array to Device array
    cudaMemcpy (dev_a , localA , l1, cudaMemcpyHostToDevice);
    cudaMemcpy (dev_b , localB , l2, cudaMemcpyHostToDevice);
    // Execute the kernel
    // define grid and thread block sizes

    dim3 dimGrid((l3/1024/32+1), 1024), dimBlock(32);
    matrix_multiplication <<<dimGrid, dimBlock>>> (dev_a, dev_b, dev_c, localMD->x);
    //Copy back to Host array from Device array
    cudaMemcpy(localC , dev_c , l3, cudaMemcpyDeviceToHost);
    //Free the Device array memory
    cudaFree (dev_a);
    cudaFree (dev_b);
    cudaFree (dev_c);
}

#endif // _MATRIXMULTIPLICATIONMODEL_CPP_