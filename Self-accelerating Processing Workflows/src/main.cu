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
#include <models/MatrixMulModel.h>
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
	// ComputationalModel::setOperationalMode(true);
	LARGE_INTEGER start, stop, clockFreq;
	ofstream outfile("ML_train_data.csv");
	QueryPerformanceFrequency(&clockFreq);
	double delay;
	double elapsedTimeCPU, elapsedTimeGPU;

	const int experiment_count = 5;

	MatrixMultiplicationModel<numericalType1> matmulmodel(4);
	matmulmodel.clearLogs();		// empty the performance matrix log file

	int step = 32;
	int levels = 8;
	int lengthX, lengthY, lengthZ;
	numericalType1 *mat1, *mat2, *matOut1, *matOut2;
	cout << "Dim\t" << "Status\t" << "CPU\t" << "GPU (ms)" << endl << endl;		// print header
	outfile << "x,y,z,prediction" << endl;		// print header
	for (int m = step; m <= levels * step; m += step) {
		for (int l = step; l <= levels * step; l += step) {
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

				elapsedTimeCPU = 0;
				for (int l = 0; l < experiment_count; l++) {
					QueryPerformanceCounter(&start);
					matmulmodel.invoke(mat1, mat2, matOut1, new myDim3(l, m, n));
					matmulmodel.execute(1);
					QueryPerformanceCounter(&stop);
					elapsedTimeCPU += (stop.QuadPart - start.QuadPart);
				}


				elapsedTimeGPU = 0;
				for (int l = 0; l < experiment_count; l++) {
					QueryPerformanceCounter(&start);
					matmulmodel.invoke(mat1, mat2, matOut1, new myDim3(l, m, n));
					matmulmodel.execute(2);
					QueryPerformanceCounter(&stop);
					elapsedTimeGPU += (stop.QuadPart - start.QuadPart);
				}

				string status = "Differ";
				if (compareResults(matOut1, matOut2, lengthZ))
					status = "Same";
				cout << l << "," << m << "," << n << "\t" << status << "\t" << elapsedTimeCPU << "\t" << elapsedTimeGPU << endl;

				//print results
				// for (int t = 0; t < lengthZ; t++)
				//	cout << matOut1[t] << ", " << matOut2[t] << endl;

				outfile << l << "," << m << "," << n << "," << (elapsedTimeCPU < elapsedTimeGPU ? 0 : 1) << endl;

				free(mat1);
				free(mat2);
				free(matOut1);
				free(matOut2);
			}
		}
	}
	outfile.close();
	return 0;
}
