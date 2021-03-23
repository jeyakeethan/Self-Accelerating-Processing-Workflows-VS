#include <models/BlurModel.h>
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

template <class T>
void BlurModel<T>::CPUImplementation() {
	// log mode to see the flow of execution
	CPUGPULOG << 0;

    int fsize = 5; // Filter size

    int size = width * height;
    //implement using multi threads
#pragma omp parallel num_threads(CPUCores)
    {
#pragma omp for
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int offset = y * width + x;
                if (offset < size) {
                    float output_red = 0;
                    float output_green = 0;
                    float output_blue = 0;
                    int hits = 0;
                    for (int ox = -fsize; ox < fsize + 1; ++ox) {
                        for (int oy = -fsize; oy < fsize + 1; ++oy) {
                            if ((x + ox) > -1 && (x + ox) < width && (y + oy) > -1 && (y + oy) < height) {
                                const int currentoffset = (offset + ox + oy * width) * 3;
                                output_red += input_image[currentoffset];
                                output_green += input_image[currentoffset + 1];
                                output_blue += input_image[currentoffset + 2];
                                hits++;
                            }
                        }
                    }
                    output_image[offset * 3] = output_red / hits;
                    output_image[offset * 3 + 1] = output_green / hits;
                    output_image[offset * 3 + 2] = output_blue / hits;
                }
            }
        }
#pragma omp barrier
    }

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


// retrive influenced attributes
template <class T>
vector<float>* BlurModel<T>::getAttributes() {
    return attr;
}

#endif // _BLURMODEL_CPP_