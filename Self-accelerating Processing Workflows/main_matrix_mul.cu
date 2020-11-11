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

	// ---Random Seed Value---
	srand(5);

	MatrixMultiplicationModel<numericalType1> matmulmodel(4);
	matmulmodel.clearLogs();		// empty the performance matrix log file

	const int step = 32;
	const int levels = 3;
	const int spaceLength = pow(levels, 3);
	myDim3**matrixSpace = new myDim3*[spaceLength];

	int lengthX, lengthY, lengthZ, counter = 0;
	for (int l = step; l <= levels * step; l += step) 
		for (int m = step; m <= levels * step; m += step) 
			for (int n = step; n <= levels * step; n += step) {
				matrixSpace[counter++] = new myDim3(l,m,n);
			}
//	for (int space = 0; space < spaceLength; space++) {
//		cout << matrixSpace[space].x << ", " << matrixSpace[space].y << ", " << matrixSpace[space].z << endl;
//	}

	numericalType1** arraySet1 = new numericalType1 * [EXPERIMENT_COUNT];
	numericalType1** arraySet2 = new numericalType1 * [EXPERIMENT_COUNT];
	int x, k, fileNum, length, widthCount, width;
	int arrayLength[EXPERIMENT_COUNT];
	myDim3* dimension;
	numericalType1* mat1, * mat2;
	numericalType1*  matOut;
	bool iSmall;

	switch (INPUT_NATURE) {
	case 1:
		/*********Generate Aligned Binary Input Stream*********/
		widthCount = 0, width = rand() % MAX_WIDTH_ALIGNED + 1;
		iSmall = true;
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			if (++widthCount > width) {
				//cout << "width: " << width << endl << endl;
				widthCount = 0;
				width = rand() % (MAX_WIDTH_ALIGNED - MIN_WIDTH_ALIGNED) + MIN_WIDTH_ALIGNED;
				iSmall = !iSmall;
			}
			if (iSmall) dimension = matrixSpace[0];
			else dimension = matrixSpace[4];
			//cout << "length: " << length << endl;
			int l1 = matrixSpace[0]->x * matrixSpace[0]->y, l2 = matrixSpace[0]->z * matrixSpace[0]->y, l3 = matrixSpace[0]->x * matrixSpace[0]->z;
			numericalType1* temp1 = new numericalType1[l1];
			numericalType1* temp2 = new numericalType1[l2];
			matOut = new numericalType1[l3];
			arraySet1[x] = temp1;
			arraySet2[x] = temp2;
			for (k = 0; k < l1; k++) 
				temp1[k] = rand() % RANGE_OF_INT_VALUES;
			for (k = 0; k < l2; k++)
				temp2[k] = rand() % RANGE_OF_INT_VALUES;
			
		}
		break;
	case 2:
		//  *********Generate Aligned Square Wave Input Stream*********
		widthCount = 0, width = rand() % MAX_WIDTH_ALIGNED + 1;
		iSmall = true;
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

	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		matmulmodel.setData(arraySet1[x], arraySet2[x], matOut, dimension);
		matmulmodel.executeAndLogging(2);
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	int elapsedTimeGPU = int(delay * 1000);
	
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		matmulmodel.setData(arraySet1[x], arraySet2[x], matOut, dimension);
		matmulmodel.executeAndLogging(1);
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	int elapsedTimeCPU = int(delay * 1000);

	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		matmulmodel.setData(arraySet1[x], arraySet2[x], matOut, dimension);
		matmulmodel.execute();
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	int elapsedAutoTime = int(delay * 1000);

	cout << "CPU:\t" << elapsedTimeCPU << "\tGPU:\t" << elapsedTimeGPU  << "\tSelfFlow:\t" << elapsedAutoTime << endl;

	for (int ex = 0; ex < EXPERIMENT_COUNT; ex++) {
		free(arraySet1[ex]);
		free(arraySet2[ex]);
	}
	free(matrixSpace);
	free(matOut);
	return 0;
}
