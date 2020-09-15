#include <MatrixMultiplicationModel.h>
#include <kernels.h>
#include <pthread.h>

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
    pthread_t* thread_handles;
    malloc(thread_handles, CPUCores * sizeof(pthread_t));
    for (int i = 0; i < CPUCores; i++) {
        int[2] args_ = {i, no_rows_per_thread };
        pthread_create(&thread_handles[i], NULL, threadMatMult, (void*)args_);
    }
    for (int i = 0; i < CPUCores; i++) {
        pthread_join(thread_handles[i], NULL);
    }
}

void* threadMatMult(void* args) {
    long my_rank = (long)args[0];
    long no_rows = (long)args[1];
    int my_first_row = my_rank * no_rows;
    int my_last_row = (my_rank+1) * no_rows-1;

    int i, j, k;
    for (i = my_first_row; i < my_last_row; i++) {
        for (j = 0; j < localMD->y; j++) {
            out[localMD->y * i + j] = 0;
            for (k = 0; k < localMD->y; k++)
                out[localMD->y*i+j] += localA[localMD->y * i+k] * localB[j+ localMD->z * k];
        }
    }
    return NULL;
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