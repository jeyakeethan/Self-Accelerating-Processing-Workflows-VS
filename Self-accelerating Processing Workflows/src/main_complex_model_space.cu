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
#include <models/ComplexModel.h>
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
	LARGE_INTEGER start, stop, clockFreq;
	ofstream outfile("../ml-datasets/complex-model.csv");
	QueryPerformanceFrequency(&clockFreq);
	double delay;
	double elapsedTimeCPU, elapsedTimeGPU;

	const int experiment_count = 5;

	ComplexModel<numericalType1> complexModel(6);
	complexModel.clearLogs();		// empty the performance matrix log file

	int step = 32;
	int levels = 8;
	int lengthA, lengthB, lengthC;
	numericalType1* mat1, * mat2, *maty, *matx, * matOut1, * matOut2;
	cout << "Dim\t" << "Status\t" << "CPU\t" << "GPU (ms)" << endl << endl;		// print header
	for (int m = step; m <= levels * step; m += step) {
		for (int l = step; l <= levels * step; l += step) {
			for (int n = step; n <= levels * step; n += step) {
				lengthA = l * m;
				lengthB = m * n;
				lengthC = l * n;
				mat1 = new numericalType1[lengthA];
				mat2 = new numericalType1[lengthB];
				maty = new numericalType1[lengthB];
				matx = new numericalType1[lengthC];
				matOut1 = new numericalType1[lengthC];
				matOut2 = new numericalType1[lengthC];

				for (int a = 0; a < lengthA; a++)
					mat1[a] = rand() % RANGE_OF_INT_VALUES;
				for (int b = 0; b < lengthB; b++)
					mat2[b] = rand() % RANGE_OF_INT_VALUES;
				for (int b = 0; b < lengthB; b++)
					maty[b] = rand() % RANGE_OF_INT_VALUES;
				for (int c = 0; c < lengthC; c++)
					matx[c] = rand() % RANGE_OF_INT_VALUES;

				elapsedTimeCPU = 0;
				for (int k = 0; k < experiment_count; k++) {
					complexModel.SetData(mat1, mat2, matx,  maty, matOut1, new myDim3(l, m, n));
					QueryPerformanceCounter(&start);
					complexModel.execute(1);
					QueryPerformanceCounter(&stop);
					elapsedTimeCPU += (stop.QuadPart - start.QuadPart);
				}


				complexModel.SetData(mat1, mat2, matx, maty, matOut2, new myDim3(l, m, n));
				complexModel.execute(2);
				elapsedTimeGPU = 0;
				for (int k = 0; k < experiment_count; k++) {
					complexModel.SetData(mat1, mat2, matx, maty, matOut2, new myDim3(l, m, n));
					QueryPerformanceCounter(&start);
					complexModel.execute(2);
					QueryPerformanceCounter(&stop);
					elapsedTimeGPU += (stop.QuadPart - start.QuadPart);
				}

				/* print results
				string status = "Differ";
				if (compareResults(matOut1, matOut2, lengthC))
					status = "Same";
				cout << l << "," << m << "," << n << "\t" << status << "\t" << elapsedTimeCPU << "\t" << elapsedTimeGPU << endl;

				// for (int t = 0; t < lengthC; t++)
				//	cout << matOut1[t] << ", " << matOut2[t] << endl;
				*/

				outfile << l << "," << m << "," << n << "," << ((elapsedTimeGPU - elapsedTimeCPU > SPACE_TIME_MARGIN) ? 0 : 1) << endl;

				free(mat1);
				free(mat2);
				free(matx);
				free(maty);
				free(matOut1);
				free(matOut2);
			}
		}
	}
	outfile.close();
	return 0;
}
