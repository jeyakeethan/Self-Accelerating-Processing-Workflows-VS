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

static bool compareResults(numericalType1* arr1, numericalType1* arr2, int len) {
	for (int x = 0; x < len; x++)
		if (arr1[x] != arr2[x])
			return false;
	return true;
}

int main()
{
	/*LARGE_INTEGER start, stop, clockFreq;
	ofstream outfile;
	QueryPerformanceFrequency(&clockFreq);
	double delay;
	int elapsedTime;

	MatrixMultiplicationModel<numericalType1> matmulmodel(4);
	numericalType1 mat1[6] = { 1, 3, 7,8,4,3 };
	numericalType1 mat2[6] = { 1, 3, 7,8,3,2 };
	numericalType1 out[4];
	matmulmodel.setData(mat1, mat2, out, new myDim3(2, 3, 2));

	QueryPerformanceCounter(&start);
	matmulmodel.execute(1);
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << endl << "CPU Time: " << elapsedTime << " ms" << endl;
	for (int t = 0; t < 4; t++) {
		cout << out[t] << endl;
		out[t] = 0;
	}

	QueryPerformanceCounter(&start);
	matmulmodel.execute(2);
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << endl << "GPU Time: " << elapsedTime << " ms" << endl;
	for (int t = 0; t < 4; t++)
		cout << out[t] << endl;

	*/

	// ComputationalModel::setOperationalMode(true);
	LARGE_INTEGER start, stop, clockFreq;
	ofstream outfile;
	QueryPerformanceFrequency(&clockFreq);
	double delay;
	int elapsedTimeCPU, elapsedTimeGPU;

	MatrixMultiplicationModel<numericalType1> matmulmodel(4);
	matmulmodel.clearLogs();		// empty the performance matrix log file

	int step = 32;
	int levels = 1;
	int lengthX, lengthY, lengthZ;
	numericalType1 *mat1, *mat2, *matOut1, *matOut2;
	cout << "Status\t" << "CPU\t" << "GPU (ms)" << endl << endl;		// print header
	for (int l = step; l <= levels * step; l += step) {
		for (int m = step; m <= levels * step; m += step) {
			for (int n = step; n <= levels * step; n += step) {
				lengthX = l * m;
				lengthY = m * n;
				lengthZ = l * n;
				mat1 = new numericalType1[lengthX];
				mat2 = new numericalType1[lengthY];
				matOut1 = new numericalType1[lengthZ];
				matOut2 = new numericalType1[lengthZ];

				for (int a = 0; a < lengthX; a++)
					mat1[a] = rand() % RANGE_OF_INT_VALUES;
				for (int b = 0; b < lengthY; b++)
					mat2[b] = rand() % RANGE_OF_INT_VALUES;

				matmulmodel.setData(mat1, mat2, matOut1, new myDim3(l, m, n));
				QueryPerformanceCounter(&start);
				matmulmodel.executeAndLogging(1);
				QueryPerformanceCounter(&stop);
				delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
				elapsedTimeCPU = int(delay * 1000);

				matmulmodel.setData(mat1, mat2, matOut2, new myDim3(l, m, n));
				QueryPerformanceCounter(&start);
				matmulmodel.executeAndLogging(2);
				QueryPerformanceCounter(&stop);
				delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
				elapsedTimeGPU = int(delay * 1000);

				string status = "Differ";
				if (compareResults(matOut1, matOut2, lengthZ))
					status = "Same";
				cout << status << "\t" << elapsedTimeCPU << "\t" << elapsedTimeGPU << endl;

				for (int t = 0; t < lengthZ; t++)
					cout << matOut1[t] << ", " << matOut2[t] << endl;

				free(mat1);
				free(matOut1);
				free(mat2);
				free(matOut2);
			}
		}
	}
	return 0;
}
