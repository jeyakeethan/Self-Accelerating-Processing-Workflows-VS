#include <Constants.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include <ArrayAdditionModel.h>
#include <kernels.h>

ArrayAdditionModel::ArrayAdditionModel() {}
ArrayAdditionModel::ArrayAdditionModel(int *in1, int *in2, int *out, int length): localA(in1), localB(in2), localC(out), localL(length) { }
ArrayAdditionModel::~ArrayAdditionModel() {}

void ArrayAdditionModel::CPUImplementation(){
    printf("Hello CPU IMPL \n");
    for(int x = 0; x < localL; x++){
        localC[x] = localA[x] + localB[x];
    }
}
void ArrayAdditionModel::GPUImplementation(){
    printf("Hello GPU IMPL \n");
    //Device array
    int *dev_a , *dev_b, *dev_c;
    //Allocate the memory on the GPU
    cudaMalloc((void **)&dev_a , N*sizeof(int));
    cudaMalloc((void **)&dev_b , N*sizeof(int));
    cudaMalloc((void **)&dev_c , N*sizeof(int));
    //Copy Host array to Device array
    cudaMemcpy (dev_a , localA , N*sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy (dev_b , localB , N*sizeof(int) , cudaMemcpyHostToDevice);
    // Execute the kernel

    Vector_Addition <<< N / THREATS_PER_BLOCK, THREATS_PER_BLOCK >>> (dev_a, dev_b, dev_c);
    //Copy back to Host array from Device array
    cudaMemcpy(localC , dev_c , N*sizeof(int) , cudaMemcpyDeviceToHost);
    //Free the Device array memory
    cudaFree (dev_a);
    cudaFree (dev_b);
    cudaFree (dev_c);
}
