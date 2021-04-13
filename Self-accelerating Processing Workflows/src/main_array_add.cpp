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
#include <models/ArrayAddModel.h>
#include <random>
#include <string>

using namespace std;
int main()
{
	LARGE_INTEGER start, stop, clockFreq;
	ofstream outfile;
	QueryPerformanceFrequency(&clockFreq);
	double delay;
	int elapsedTime;

	ArrayAdditionModel<numericalType1> arrayAdditionModel(1);

	numericalType1** arraySet1 = new numericalType1 * [EXPERIMENT_COUNT];
	numericalType1** arraySet2 = new numericalType1 * [EXPERIMENT_COUNT];
	int* arrayLength = new int[EXPERIMENT_COUNT];
	int x, k, length, fileNum;

	/*---Random Seed Value---*/
	srand(5);

	if (INPUT_NATURE == 1) {
		/*********Generate Aligned Square Wave Input Stream*********/
		int widthCount = 0, width = rand() % MAX_WIDTH_ALIGNED + 1, arrayMaxLength = ARRAY_MAX_LENGTH, smallArrayMaxLength = SMALL_ARRAY_MAX_LENGTH;
		bool iSmall = true;
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			if (++widthCount > width) {
				//cout << "width: " << width << endl << endl;
				widthCount = 0;
				width = rand() % (MAX_WIDTH_ALIGNED - MIN_WIDTH_ALIGNED) + MIN_WIDTH_ALIGNED;
				iSmall = !iSmall;
				if (iSmall) smallArrayMaxLength = rand() % (ARRAY_MAX_LENGTH - SMALL_ARRAY_MAX_LENGTH) + SMALL_ARRAY_MAX_LENGTH + 1;
				else arrayMaxLength = rand() % SMALL_ARRAY_MAX_LENGTH + 1;
			}
			if (iSmall) length = rand() % smallArrayMaxLength + 1;
			else length = rand() % (arrayMaxLength - SMALL_ARRAY_MAX_LENGTH) + SMALL_ARRAY_MAX_LENGTH + 1;
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
	}
	else if (INPUT_NATURE == 2) {
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
	}
	else if (INPUT_NATURE == 3) {
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
	}
	else if (INPUT_NATURE == 4) {
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
	}
	else if (INPUT_NATURE == 5) {
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
	}

	/************Write Input Nature into File************/
	string inputNatureFile = "Input Nature.csv"; fileNum = 0;
	while (FILE* file = fopen(inputNatureFile.c_str(), "r")) {
		fclose(file);
		inputNatureFile = "Input Nature_" + to_string(++fileNum) + ".csv";
	}
	outfile.open(inputNatureFile);

	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		int len = arrayLength[x];
		outfile << len << endl;
		//for (y = 0; y < len/100+1; y++)
		//    cout << "-";
		//cout << endl;
	}
	outfile.close();

	string timeLogFile = "Time.log"; fileNum = 0;
	while (FILE* file = fopen(timeLogFile.c_str(), "r")) {
		fclose(file);
		timeLogFile = "Time_" + to_string(++fileNum) + ".log";
	}
	outfile.open(timeLogFile);


	/**********Self flow experiment - ArrayAdditionModel**********/
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		int len = arrayLength[x];
		numericalType1* output = new numericalType1[len];
		arrayAdditionModel.invoke(arraySet1[x], arraySet2[x], output, len);
		arrayAdditionModel.execute();
		//for(int i=0; i<len; i++)
		//   cout << output[i] << ", ";
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "Self Flow Time: " << elapsedTime << " ms" << endl << endl;
	outfile << "Self Flow Time: " << elapsedTime << " ms" << endl << endl;

	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		int len = arrayLength[x];
		numericalType1* output = new numericalType1[len];
		arrayAdditionModel.invoke(arraySet1[x], arraySet2[x], output, len);
		arrayAdditionModel.execute(1);
		//for (int i = 0; i < len; i++)
		//    cout << output[i] << ", ";
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "CPU Only Time: " << elapsedTime << " ms" << endl << endl;
	outfile << "CPU Only Time: " << elapsedTime << " ms" << endl << endl;


	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		int len = arrayLength[x];
		numericalType1* output = new numericalType1[len];
		arrayAdditionModel.invoke(arraySet1[x], arraySet2[x], output, len);
		arrayAdditionModel.execute(2);
		//for (int i = 0; i < len; i++)
		//    cout << output[i] << ", ";
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "GPU Only Time: " << elapsedTime << " ms" << endl << endl;
	outfile << "GPU Only Time: " << elapsedTime << " ms" << endl << endl;
	outfile.close();

	/*************Free Host Memory**************/
	delete[] arraySet1;
	delete[] arraySet2;

	/*

	int inputA[N];
	int inputB[N];
	int output[N];

	ArrayAdditionModel<int> arrayAdditionModel;

	for (int exp = 0; exp < EXPERIMENT_COUNT; exp++) {
		for (int k = 0; k < N; k++) {
			inputA[k] = rand() % RANGE_OF_INT_VALUES;
			inputB[k] = rand() % RANGE_OF_INT_VALUES;
		}

		arrayAdditionModel.invoke(inputA, inputB, output, N);
		arrayAdditionModel.execute();
		// for(int i=0; i<N; i++)
		//    cout << output[i] << ", ";
	}
	QueryPerformanceCounter(&stop);
	double delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	int elapsedTime = int(delay * 1000);
	cout << "CPU Time: " << elapsedTime << " ms" << endl;


	int inputA[N];
	int inputB[N];

	DotMultiplicationModel dotMultiplicationModel;

	for (int exp = 0; exp < EXPERIMENT_COUNT; exp++) {
		int out = 0;
		for (int k = 0; k < N; k++) {
			inputA[k] = rand() % RANGE_OF_INT_VALUES;
			inputB[k] = rand() % RANGE_OF_INT_VALUES;
		}

		dotMultiplicationModel.invoke(inputA, inputB, &out, N);
		dotMultiplicationModel.execute(1);
		//cout << out << endl;
	}
	*/
	return 0;
}
