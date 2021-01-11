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
#include <MatrixMulMLModel.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>
using namespace std;
int main()
{
	//MatrixMulMLModel::trainModelStatic();

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
	const int levels = 4;
	const int spaceLength = pow(levels, 3);
	myDim3**matrixSpace = new myDim3*[spaceLength];

	int lengthX, lengthY, lengthZ;
	int levels2 = levels * levels;
	for (int n = 0; n < spaceLength; n++) {
		matrixSpace[n] = new myDim3((n / levels2 + 1) * step, (n % levels2 / levels + 1) * step, (n % levels + 1) * step);
		// cout << matrixSpace[n]->x << ", " << matrixSpace[n]->y << ", " << matrixSpace[n]->z << endl;
	}
	numericalType1** arraySet1 = new numericalType1 * [EXPERIMENT_COUNT];
	numericalType1** arraySet2 = new numericalType1 * [EXPERIMENT_COUNT];
	numericalType1** matOut = new numericalType1 * [EXPERIMENT_COUNT];
	myDim3** correspondingMatrixSpace = new myDim3 * [EXPERIMENT_COUNT];
	int x, k, fileNum, length, widthCount, width;
	int l1, l2, l3;
	int arrayLength[EXPERIMENT_COUNT];
	myDim3* dimension; 
	myDim3* selectedMatDim;
	//numericalType1* mat1, * mat2;
	bool iSmall;
	switch (INPUT_NATURE) {
		case 1:
			/*********Generate Aligned Binary Input Stream*********/
			widthCount = 0, width = rand() % MAX_WIDTH_ALIGNED + 1;
			iSmall = true;
			for (x = 0; x < spaceLength; x++) {
				if (++widthCount > width) {
					//cout << "width: " << width << endl << endl;
					widthCount = 0;
					width = rand() % (MAX_WIDTH_ALIGNED - MIN_WIDTH_ALIGNED) + MIN_WIDTH_ALIGNED;
					iSmall = !iSmall;
				}
				if (iSmall) dimension = matrixSpace[0];
				else dimension = matrixSpace[4];
				//cout << "length: " << length << endl;
				l1 = matrixSpace[x]->x * matrixSpace[x]->y, l2 = matrixSpace[x]->z * matrixSpace[x]->y, l3 = matrixSpace[x]->x * matrixSpace[x]->z;
				numericalType1* temp1 = new numericalType1[l1];
				numericalType1* temp2 = new numericalType1[l2];
				arraySet1[x] = temp1;
				arraySet2[x] = temp2;
				matOut[x] = new numericalType1[l3];
				correspondingMatrixSpace[x] = matrixSpace[x];
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
			selectedMatDim = matrixSpace[rand() % spaceLength];
			l1 = selectedMatDim->x * selectedMatDim->y, l2 = selectedMatDim->z * selectedMatDim->y, l3 = selectedMatDim->x * selectedMatDim->z;
			for (x = 0; x < EXPERIMENT_COUNT; x++) {
				if (++widthCount > width) {
					//cout << "width: " << width << endl << endl;
					widthCount = 0;
					width = rand() % (MAX_WIDTH_ALIGNED - MIN_WIDTH_ALIGNED) + MIN_WIDTH_ALIGNED;
					// if(iSmall)
					//	selectedMatDim = matrixSpace[rand() % spaceLength];
					// else
					//	selectedMatDim = matrixSpace[rand() % spaceLength];
					selectedMatDim = matrixSpace[rand() % spaceLength];
					l1 = selectedMatDim->x * selectedMatDim->y, l2 = selectedMatDim->z * selectedMatDim->y, l3 = selectedMatDim->x * selectedMatDim->z;
					iSmall = !iSmall;
				}
				//cout << "length: " << length << endl;
			
				numericalType1* temp1 = new numericalType1[l1];
				numericalType1* temp2 = new numericalType1[l2];
				arraySet1[x] = temp1;
				arraySet2[x] = temp2;
				matOut[x] = new numericalType1[l3];
				correspondingMatrixSpace[x] = selectedMatDim;
				for (k = 0; k < l1; k++)
					temp1[k] = rand() % RANGE_OF_INT_VALUES;
				for (k = 0; k < l2; k++)
					temp2[k] = rand() % RANGE_OF_INT_VALUES;
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

	/*Mannual Execute only in GPU*/
	matmulmodel.setData(arraySet1[0], arraySet2[0], matOut[5], correspondingMatrixSpace[0]);	// to initialise GPU to avoid initialization overhead
	matmulmodel.executeAndLogging(2);											// to initialise GPU to avoid initialization overhead
	if (LOGGER_MODE_ON) {
		QueryPerformanceCounter(&start);
		for (x = 0; x < spaceLength; x++) {
			matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
			matmulmodel.executeAndLogging(2);
		}
		QueryPerformanceCounter(&stop);
	}
	else {
		QueryPerformanceCounter(&start);
		for (x = 0; x < spaceLength; x++) {
			matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
			matmulmodel.execute(2);
		}
		QueryPerformanceCounter(&stop);
	}
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	int elapsedTimeGPU = int(delay * 1000);
	matmulmodel.logExTime("\n\n"); // add new line in logging file
	

	/*Mannual Execute only in CPU*/
	if (LOGGER_MODE_ON) {
		QueryPerformanceCounter(&start);
		for (x = 0; x < spaceLength; x++) {
			matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
			matmulmodel.executeAndLogging(1);
		}
		QueryPerformanceCounter(&stop);
	}
	else {
		QueryPerformanceCounter(&start);
		for (x = 0; x < spaceLength; x++) {
			matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
			matmulmodel.execute(1);
		}
		QueryPerformanceCounter(&stop);
	}
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	int elapsedTimeCPU = int(delay * 1000);
	matmulmodel.logExTime("\n\n"); // add new line in logging file

	/*Automated Hybrid*/
	matmulmodel.setData(arraySet1[0], arraySet2[0], matOut[x], correspondingMatrixSpace[0]);	// to initialise GPU to avoid initialization overhead
	matmulmodel.executeAndLogging(2);											// to initialise GPU to avoid initialization overhead
	if (LOGGER_MODE_ON) {
		QueryPerformanceCounter(&start);
		for (x = 0; x < spaceLength; x++) {
			matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
			matmulmodel.executeAndLogging();
		}
		QueryPerformanceCounter(&stop);
	}
	else {
		QueryPerformanceCounter(&start);
		for (x = 0; x < spaceLength; x++) {
			matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
			matmulmodel.execute();
		}
		QueryPerformanceCounter(&stop);
	}
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	int elapsedAutoTime = int(delay * 1000);
	matmulmodel.logExTime("\n\n"); // add new line in logging file

	// Automated ML only
	matmulmodel.setData(arraySet1[0], arraySet2[0], matOut[x], correspondingMatrixSpace[0]);	// to initialise GPU to avoid initialization overhead
	matmulmodel.executeAndLogging(2);											// to initialise GPU to avoid initialization overhead
	if (LOGGER_MODE_ON) {
		QueryPerformanceCounter(&start);
		for (x = 0; x < spaceLength; x++) {
			matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
			matmulmodel.executeByML();
		}
		QueryPerformanceCounter(&stop);
	}
	else {
		QueryPerformanceCounter(&start);
		for (x = 0; x < spaceLength; x++) {
			matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
			matmulmodel.executeByML();
		}
		QueryPerformanceCounter(&stop);
	}
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	int elapsedML = int(delay * 1000);
	matmulmodel.logExTime("\n\n"); // add new line in logging file

	cout << "CPU:\t" << elapsedTimeCPU << "\tGPU:\t" << elapsedTimeGPU  << "\tSelfFlow:\t" << elapsedAutoTime<< "\tML Flow:\t" << elapsedML << endl;

	for (int ex = 0; ex < spaceLength; ex++) {
		free(arraySet1[ex]);
		free(arraySet2[ex]);
		free(matOut[ex]);
		free(matrixSpace[ex]);
		//free(correspondingMatrixSpace[ex]);
	}
	free(matrixSpace);
	free(matOut);
	return 0;
}
