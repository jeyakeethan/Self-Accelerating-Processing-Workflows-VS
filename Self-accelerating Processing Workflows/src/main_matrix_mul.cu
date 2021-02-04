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
#include <sstream>
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
	freopen("console log.txt", "w", stdout);

	LARGE_INTEGER start, stop, clockFreq;
	ofstream outfile;
	QueryPerformanceFrequency(&clockFreq);
	double delay;

	// ---Random Seed Value---
	srand(5);

	MatrixMultiplicationModel<numericalType1> matmulmodel(6);
	matmulmodel.clearLogs();		// empty the performance matrix log file

	const int step = 32;
	const int levels = 8;
	const int spaceLength = pow(levels, 3);
	int loop_length = EXPERIMENT_COUNT;
	myDim3**matrixSpace = new myDim3*[spaceLength];

	int levels2 = levels * levels;
	for (int n = 0; n < spaceLength; n++) {
		matrixSpace[n] = new myDim3((n / levels2 + 1) * step, (n % levels2 / levels + 1) * step, (n % levels + 1) * step);
		// cout << matrixSpace[n]->x << ", " << matrixSpace[n]->y << ", " << matrixSpace[n]->z << endl;
	}
	numericalType1** arraySet1 = new numericalType1 * [EXPERIMENT_COUNT];
	numericalType1** arraySet2 = new numericalType1 * [EXPERIMENT_COUNT];
	numericalType1** matOut = new numericalType1 * [EXPERIMENT_COUNT];
	myDim3** correspondingMatrixSpace = new myDim3 * [EXPERIMENT_COUNT];
	int x, k, widthCount, width;
	int l1, l2, l3;
	myDim3 *dimension, *CPUSpecificMatDim, *GPUSpecificMatDim;
	CPUSpecificMatDim = matrixSpace[0], GPUSpecificMatDim = matrixSpace[spaceLength-1];
	int lmn = CPUSpecificMatDim->x * CPUSpecificMatDim->x * CPUSpecificMatDim->x;
	int xyz = GPUSpecificMatDim->x * GPUSpecificMatDim->x * GPUSpecificMatDim->x;
	bool iSmall;
	stringstream results0;
	stringstream results;

	auto TestEachCase = [&]() {
		results0 << endl << "Execution in GPU only started" << endl;
		/*Mannual Execute only in GPU*/
		matmulmodel.setData(arraySet1[0], arraySet2[0], matOut[0], correspondingMatrixSpace[0]);	// to initialise GPU to avoid initialization overhead
		matmulmodel.execute(2);																		// to initialise GPU to avoid initialization overhead
		if (LOGGER_MODE_ON) {
			QueryPerformanceCounter(&start);
			for (x = 0; x < loop_length; x++) {
				matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
				matmulmodel.executeAndLogging(2);
			}
			QueryPerformanceCounter(&stop);
		}
		else {
			QueryPerformanceCounter(&start);
			for (x = 0; x < loop_length; x++) {
				matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
				matmulmodel.execute(2);
			}
			QueryPerformanceCounter(&stop);
		}
		delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
		int elapsedTimeGPU = int(delay * 1000);
		matmulmodel.logExTime("\n\n"); // add new line in logging file
		results0 << endl << "Code: " << matmulmodel.CPUGPULOG.str() << endl;
		matmulmodel.CPUGPULOG.clear();


		results0 << endl << "Execution in CPU only started" << endl;
		/*Mannual Execute only in CPU*/
		if (LOGGER_MODE_ON) {
			QueryPerformanceCounter(&start);
			for (x = 0; x < loop_length; x++) {
				matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
				matmulmodel.executeAndLogging(1);
			}
			QueryPerformanceCounter(&stop);
		}
		else {
			QueryPerformanceCounter(&start);
			for (x = 0; x < loop_length; x++) {
				matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
				matmulmodel.execute(1);
			}
			QueryPerformanceCounter(&stop);
		}
		delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
		int elapsedTimeCPU = int(delay * 1000);
		matmulmodel.logExTime("\n\n"); // add new line in logging file
		results0 << endl << "Code: " << matmulmodel.CPUGPULOG.str() << endl;
		matmulmodel.CPUGPULOG.clear();


		results0 << endl << "Automated Hybrid Execution started" << endl;
		/*Automated Hybrid*/
		matmulmodel.setData(arraySet1[0], arraySet2[0], matOut[0], correspondingMatrixSpace[0]);	// to initialise GPU to avoid initialization overhead
		matmulmodel.execute(2);																		// to initialise GPU to avoid initialization overhead
		if (LOGGER_MODE_ON) {
			QueryPerformanceCounter(&start);
			for (x = 0; x < loop_length; x++) {
				matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
				matmulmodel.executeAndLogging();
			}
			QueryPerformanceCounter(&stop);
		}
		else {
			QueryPerformanceCounter(&start);
			for (x = 0; x < loop_length; x++) {
				matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
				matmulmodel.execute();
			}
			QueryPerformanceCounter(&stop);
		}
		delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
		int elapsedAutoTime = int(delay * 1000);
		matmulmodel.logExTime("\n\n"); // add new line in logging file
		results0 << endl << "Code: " << matmulmodel.CPUGPULOG.str() << endl;
		matmulmodel.CPUGPULOG.clear();

		results0 << endl << "Automated ML only Execution started" << endl;
		// Automated ML only
		matmulmodel.setData(arraySet1[0], arraySet2[0], matOut[0], correspondingMatrixSpace[0]);	// to initialise GPU to avoid initialization overhead
		matmulmodel.execute(2);																		// to initialise GPU to avoid initialization overhead
		if (LOGGER_MODE_ON) {
			QueryPerformanceCounter(&start);
			for (x = 0; x < loop_length; x++) {
				matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
				matmulmodel.executeByML();
			}
			QueryPerformanceCounter(&stop);
		}
		else {
			QueryPerformanceCounter(&start);
			for (x = 0; x < loop_length; x++) {
				matmulmodel.setData(arraySet1[x], arraySet2[x], matOut[x], correspondingMatrixSpace[x]);
				matmulmodel.executeByML();
			}
			QueryPerformanceCounter(&stop);
		}
		delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
		int elapsedML = int(delay * 1000);
		matmulmodel.logExTime("\n\n");		// add new line in logging file
		results0 << endl << "Code: " << matmulmodel.CPUGPULOG.str() << endl;
		matmulmodel.CPUGPULOG.clear();

		results0 << endl << "CPU:\t" << elapsedTimeCPU << "\tGPU:\t" << elapsedTimeGPU << "\tSelfFlow:\t" << elapsedAutoTime << "\tML Flow:\t" << elapsedML << endl << endl << endl;
		results << endl << "CPU:\t" << elapsedTimeCPU << "\tGPU:\t" << elapsedTimeGPU << "\tSelfFlow:\t" << elapsedAutoTime << "\tML Flow:\t" << elapsedML << endl << endl << endl;

		for (int ex = 0; ex < loop_length; ex++) {
			free(arraySet1[ex]);
			free(arraySet2[ex]);
			free(matOut[ex]);
		}
	};

	switch (INPUT_NATURE) {
		case 1:
			results0 << "/*********Generate Binary Input Stream*********/" << endl;
			widthCount = 0, width = rand() % (MAX_WIDTH_ALIGNED-MIN_WIDTH_ALIGNED) + MIN_WIDTH_ALIGNED + 1;
			iSmall = true;
			results0 << CPUSpecificMatDim->x << "," << CPUSpecificMatDim->y << "," << CPUSpecificMatDim->z << "___" << GPUSpecificMatDim->x << "," << GPUSpecificMatDim->y << "," << GPUSpecificMatDim->z << "...." << endl;
			
			for (x = 0; x < EXPERIMENT_COUNT; x++) {
				if (++widthCount > width) {
					//cout << width << "__";
					widthCount = 0;
					width = rand() % (MAX_WIDTH_ALIGNED - MIN_WIDTH_ALIGNED) + MIN_WIDTH_ALIGNED;
					iSmall = !iSmall;
				}
				if (iSmall) {
					dimension = CPUSpecificMatDim; 
					results0 << lmn << ",";
				}
				else {
					dimension = GPUSpecificMatDim;
					results0 << xyz << ",";
				}
				l1 = dimension->x * dimension->y, l2 = dimension->z * dimension->y, l3 = dimension->x * dimension->z;
				numericalType1* temp1 = new numericalType1[l1];
				numericalType1* temp2 = new numericalType1[l2];
				arraySet1[x] = temp1;
				arraySet2[x] = temp2;
				matOut[x] = new numericalType1[l3];
				correspondingMatrixSpace[x] = dimension;
				for (k = 0; k < l1; k++)
					temp1[k] = rand() % RANGE_OF_INT_VALUES;
				for (k = 0; k < l2; k++)
					temp2[k] = rand() % RANGE_OF_INT_VALUES;
			}
			results0 << endl << endl;
			TestEachCase();
		case 2:
			results0 << "/*********Generate Square Wave Input Stream*********/" << endl;
			widthCount = 0, width = rand() % (MAX_WIDTH_ALIGNED - MIN_WIDTH_ALIGNED) + MIN_WIDTH_ALIGNED + 1;
			dimension = matrixSpace[rand() % spaceLength];
			l1 = dimension->x * dimension->y, l2 = dimension->z * dimension->y, l3 = dimension->x * dimension->z;
			lmn = dimension->x * dimension->y * dimension->z;
			for (x = 0; x < EXPERIMENT_COUNT; x++) {
				if (++widthCount > width) {
					// cout << width << "|" << dimension->x << "," << dimension->y << "," << dimension->z << " __ ";
					widthCount = 0;
					width = rand() % (MAX_WIDTH_ALIGNED - MIN_WIDTH_ALIGNED) + MIN_WIDTH_ALIGNED;
					dimension = matrixSpace[rand() % spaceLength];
					l1 = dimension->x * dimension->y, l2 = dimension->z * dimension->y, l3 = dimension->x * dimension->z;
					lmn = dimension->x * dimension->y * dimension->z;
				}
				results0 << lmn << ",";
				numericalType1* temp1 = new numericalType1[l1];
				numericalType1* temp2 = new numericalType1[l2];
				arraySet1[x] = temp1;
				arraySet2[x] = temp2;
				matOut[x] = new numericalType1[l3];
				correspondingMatrixSpace[x] = dimension;
				for (k = 0; k < l1; k++)
					temp1[k] = rand() % RANGE_OF_INT_VALUES;
				for (k = 0; k < l2; k++)
					temp2[k] = rand() % RANGE_OF_INT_VALUES;
			}
			results0 << endl << endl;
			TestEachCase();
		case 3:
			results0 << "/*********Generate GPU Specific Input Stream*********/" << endl;
			dimension = GPUSpecificMatDim;
			results0 << dimension->x << "," << dimension->y << "," << dimension->z << "_________....";
			l1 = dimension->x * dimension->y, l2 = dimension->z * dimension->y, l3 = dimension->x * dimension->z;
			for (x = 0; x < EXPERIMENT_COUNT; x++) {
				numericalType1* temp1 = new numericalType1[l1];
				numericalType1* temp2 = new numericalType1[l2];
				arraySet1[x] = temp1;
				arraySet2[x] = temp2;
				matOut[x] = new numericalType1[l3];
				correspondingMatrixSpace[x] = dimension;
				for (k = 0; k < l1; k++)
					temp1[k] = rand() % RANGE_OF_INT_VALUES;
				for (k = 0; k < l2; k++)
					temp2[k] = rand() % RANGE_OF_INT_VALUES;
			}
			results0 << endl << endl;
			TestEachCase();
		case 4:
			results0 << "/*********Generate CPU Specific Input Stream*********/" << endl;
			dimension = CPUSpecificMatDim;
			results0 << dimension->x << "," << dimension->y << "," << dimension->z << "_________....";
			l1 = dimension->x * dimension->y, l2 = dimension->z * dimension->y, l3 = dimension->x * dimension->z;
			for (x = 0; x < EXPERIMENT_COUNT; x++) {
				numericalType1* temp1 = new numericalType1[l1];
				numericalType1* temp2 = new numericalType1[l2];
				arraySet1[x] = temp1;
				arraySet2[x] = temp2;
				matOut[x] = new numericalType1[l3];
				correspondingMatrixSpace[x] = dimension;
				for (k = 0; k < l1; k++)
					temp1[k] = rand() % RANGE_OF_INT_VALUES;
				for (k = 0; k < l2; k++)
					temp2[k] = rand() % RANGE_OF_INT_VALUES;
			}
			results0 << endl << endl;
			TestEachCase();
		case 5:
			results0 << "/*********Generate Odd Input Stream*********/" << endl << "Dim Array: ";
			for (x = 0; x < EXPERIMENT_COUNT; x++) {
				dimension = matrixSpace[ rand() % spaceLength ];
				// cout << dimension->x << "," << dimension->y << "," << dimension->z << "_";
				results0 << (dimension->x * dimension->y * dimension->z) << ",";
				l1 = dimension->x * dimension->y, l2 = dimension->z * dimension->y, l3 = dimension->x * dimension->z;
				numericalType1* temp1 = new numericalType1[l1];
				numericalType1* temp2 = new numericalType1[l2];
				arraySet1[x] = temp1;
				arraySet2[x] = temp2;
				matOut[x] = new numericalType1[l3];
				correspondingMatrixSpace[x] = dimension;
				for (k = 0; k < l1; k++)
					temp1[k] = rand() % RANGE_OF_INT_VALUES;
				for (k = 0; k < l2; k++)
					temp2[k] = rand() % RANGE_OF_INT_VALUES;
			}
			results0 << endl << endl;
			TestEachCase();
			break;
	}
	cout << results0.str();
	cout << results.str();
	for (int ex = 0; ex < spaceLength; ex++) {
		free(matrixSpace[ex]);
	}
	free(matrixSpace);

	free(correspondingMatrixSpace);

	free(arraySet1);
	free(arraySet2);
	free(matOut);

	return 0;
}