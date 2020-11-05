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
#include<cmath>

using namespace std;
int main()
{
	LARGE_INTEGER start, stop, clockFreq;
	ofstream outfile;
	QueryPerformanceFrequency(&clockFreq);
	double delay;
	int elapsedTime;

	MatrixMultiplicationModel<numericalType1> matmulmodel(4);
	matmulmodel.clearLogs();		// empty the performance matrix log file

	const int step = 32;
	const int levels = 1;
	const int spaceLength = pow(levels, 3);
		cout << spaceLength << endl;
	myDim3 *matrixSpace = new myDim3[spaceLength];
	int lengthX, lengthY, lengthZ, counter = 0;
	for (int l = step; l <= levels * step; l += step) 
		for (int m = step; m <= levels * step; m += step) 
			for (int n = step; n <= levels * step; n += step) {
				lengthX = l * m;
				lengthY = m * n;
				lengthZ = l * n;
				matrixSpace[counter++] = myDim3(lengthX,lengthY,lengthZ);
			}
	for (int space = 0; space < spaceLength; space++) {
		cout << matrixSpace[space].x << ", " << matrixSpace[space].y << ", " << matrixSpace[space].z << endl;
	}
	return 0;
}
