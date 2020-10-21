#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

// measure time
#include <windows.h>
#include <time.h>

#include <Constants.h>
#include <ComputationalModel.h>
#include <ArrayAdditionModel.h>
#include <DotMultiplicationModel.h>
#include <MatrixMultiplicationModel.h>
#include <random>
#include <string>
#include <thread>
#include <future>

using namespace std;
int main()
{
	ComputationalModel::setOperationalMode(true);
	LARGE_INTEGER start, stop, clockFreq;
	ofstream outfile;
	QueryPerformanceFrequency(&clockFreq);
	double delay;
	int elapsedTime; int i;

	MatrixMultiplicationModel<numericalType1> matmulmodel(1);
	numericalType1 mat1[6] = { 1, 3, 7,8,4,3 };
	numericalType1 mat2[6] = { 1, 3, 7,8,3,2 };
	numericalType1 outCPU[4] = {0,0,0,0};
	matmulmodel.setData(mat1, mat2, outCPU, new myDim3(2, 3, 2));

	QueryPerformanceCounter(&start);
	matmulmodel.execute(2);
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << endl << "CPU Time: " << elapsedTime << " ms" << endl;
	for (i = 0; i < 4; i++) {
		cout << outCPU[i] << endl;
	}

	numericalType1 outGPU[4] = { 0,0,0,0 };
	matmulmodel.setData(mat1, mat2, outGPU, new myDim3(2, 3, 2));
	QueryPerformanceCounter(&start);
	matmulmodel.execute(2);
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << endl << "GPU Time: " << elapsedTime << " ms" << endl;
	for (i = 0; i < 4; i++)
		cout << outGPU[i] << endl;
	return 0;
}
