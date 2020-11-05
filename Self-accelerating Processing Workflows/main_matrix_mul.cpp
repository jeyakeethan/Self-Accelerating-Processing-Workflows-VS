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

	numericalType1** arraySet1 = new numericalType1 * [EXPERIMENT_COUNT];
	numericalType1** arraySet2 = new numericalType1 * [EXPERIMENT_COUNT];
	int* arrayLength = new int[EXPERIMENT_COUNT];
	int x, k, length, fileNum;

	// ---Random Seed Value---
	srand(5);

	switch (INPUT_NATURE) {
	case 1:
		//  *********Generate Aligned Square Wave Input Stream*********
		int widthCount = 0, width = rand() % MAX_WIDTH_ALIGNED + 1;
		bool iSmall = true;
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			if (++widthCount > width) {
				//cout << "width: " << width << endl << endl;
				widthCount = 0;
				width = rand() % (MAX_WIDTH_ALIGNED - MIN_WIDTH_ALIGNED) + MIN_WIDTH_ALIGNED;
				iSmall = !iSmall;
			}
			//cout << "length: " << length << endl;
			arrayLength[x] = length;
			numericalType1* temp1 = new numericalType1[length];
			numericalType1* temp2 = new numericalType1[length];
			arraySet1[x] = temp1;
			arraySet2[x] = temp2;
			for (k = 0; k < length; k++) {
				temp1[k] = rand() % RANGE_OF_INT_VALUES;
				temp2[k] = rand() % RANGE_OF_INT_VALUES;
			}
		}
		break;
	case 2:
		/*********Generate Aligned Binary Input Stream*********/
		int widthCount = 0, width = rand() % MAX_WIDTH_ALIGNED + 1;
		bool iSmall = true;
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			if (++widthCount > width) {
				//cout << "width: " << width << endl << endl;
				widthCount = 0;
				width = rand() % (MAX_WIDTH_ALIGNED - MIN_WIDTH_ALIGNED) + MIN_WIDTH_ALIGNED;
				iSmall = !iSmall;
			}
			if (iSmall) length = rand() % SMALL_ARRAY_MAX_LENGTH + 1;
			else length = rand() % (ARRAY_MAX_LENGTH - SMALL_ARRAY_MAX_LENGTH) + SMALL_ARRAY_MAX_LENGTH + 1;
			//cout << "length: " << length << endl;
			arrayLength[x] = length;
			numericalType1* temp1 = new numericalType1[length];
			numericalType1* temp2 = new numericalType1[length];
			arraySet1[x] = temp1;
			arraySet2[x] = temp2;
			for (k = 0; k < length; k++) {
				temp1[k] = rand() % RANGE_OF_INT_VALUES;
				temp2[k] = rand() % RANGE_OF_INT_VALUES;
			}
		}
		break;
	case 3:
		/*********Generate Odd Input Stream*********/
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			length = rand() % ARRAY_MAX_LENGTH + 1;
			//cout << "length: " << length << endl;
			arrayLength[x] = length;
			numericalType1* temp1 = new numericalType1[length];
			numericalType1* temp2 = new numericalType1[length];
			arraySet1[x] = temp1;
			arraySet2[x] = temp2;
			for (k = 0; k < length; k++) {
				temp1[k] = rand() % RANGE_OF_INT_VALUES;
				temp2[k] = rand() % RANGE_OF_INT_VALUES;
			}
		}
		break;
	case 4:
		/*********Generate GPU Specific Input Stream*********/
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			length = rand() % (ARRAY_MAX_LENGTH - SMALL_ARRAY_MAX_LENGTH) + SMALL_ARRAY_MAX_LENGTH + 1;
			//cout << "length: " << length << endl;
			arrayLength[x] = length;
			numericalType1* temp1 = new numericalType1[length];
			numericalType1* temp2 = new numericalType1[length];
			arraySet1[x] = temp1;
			arraySet2[x] = temp2;
			for (k = 0; k < length; k++) {
				temp1[k] = rand() % RANGE_OF_INT_VALUES;
				temp2[k] = rand() % RANGE_OF_INT_VALUES;
			}
		}
		break;
	case 5:
		/*********Generate CPU Specific Input Stream*********/
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			length = rand() % SMALL_ARRAY_MAX_LENGTH + 1;
			//cout << "length: " << length << endl;
			arrayLength[x] = length;
			numericalType1* temp1 = new numericalType1[length];
			numericalType1* temp2 = new numericalType1[length];
			arraySet1[x] = temp1;
			arraySet2[x] = temp2;
			for (k = 0; k < length; k++) {
				temp1[k] = rand() % RANGE_OF_INT_VALUES;
				temp2[k] = rand() % RANGE_OF_INT_VALUES;
			}
		}
		break;
	}

	free(matrixSpace);
	return 0;
}
