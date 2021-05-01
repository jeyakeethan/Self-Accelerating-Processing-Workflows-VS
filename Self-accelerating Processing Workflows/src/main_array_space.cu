#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "random_array_generator.cpp"

#include <stdio.h>
#include <iostream>
#include <fstream>

// measure time
#include <windows.h>
#include <time.h>

#include <Constants.h>
#include <ComputationalModel.h>
#include <models/ArrayAddModel.h>
#include <models/ArrayAdd2DModel.h>
#include <random>
#include <string>

using namespace std;
int main()
{	// write logs into file
	//string console_log_name = "../logs/array_addtion_" + CONSOLE_LOG_FILE_NAME;
	//freopen(console_log_name.c_str(), "w", stdout);

	srand(5);		// Random Seed Value

	LARGE_INTEGER start, stop, clockFreq;
	ofstream dataset_file;
	QueryPerformanceFrequency(&clockFreq);
	double delayCPU, delayGPU;
	int elapsedTime;
	int fileNum;
	short favor;
	const int experiment_count = 10;


	/*------- Write Input Nature into File -------*/
	string inputNatureFile = "../ml-datasets/Array-Addition.csv";
	dataset_file.open(inputNatureFile, ios_base::out);

	/*------------- Single dimension vector addition ------------*/
	cout << "One Dimension experiments started" << endl;

	ArrayAdditionModel<numericalType1> arrayAdditionModel(6);

	const int number_entries = 1000;
	numericalType1* arraySet1 [experiment_count];
	numericalType1* arraySet2 [experiment_count];
	numericalType1* outputs [experiment_count];
	int arrayLength[number_entries];
	int x, y, z, k, i, length;
	int step = 10000;
	for (i = 1; i < number_entries; i++) {
		length = step * i;
		arrayLength[i] = length;
		for (x = 0; x < experiment_count; x++) {
			arraySet1[x] = generate_1d_array(length);
			arraySet2[x] = generate_1d_array(length);
			outputs[x] = new numericalType1[length];
		}

		/*-------- CPU Time - ArrayAdditionModel --------*/
		QueryPerformanceCounter(&start);
		for (x = 0; x < experiment_count; x++) {
			arrayAdditionModel.SetData(arraySet1[x], arraySet2[x], outputs[x], length);
			arrayAdditionModel.execute(1);
		}
		QueryPerformanceCounter(&stop);
		delayCPU = (double)(stop.QuadPart - start.QuadPart);
		cout << "CPU Time: " << delayCPU << ", ";

		/*-------- GPU Time - ArrayAdditionModel --------*/
		QueryPerformanceCounter(&start);
		for (x = 0; x < experiment_count; x++) {
			arrayAdditionModel.SetData(arraySet1[x], arraySet2[x], outputs[x], length);
			arrayAdditionModel.execute(2);
		}
		QueryPerformanceCounter(&stop);
		delayGPU = (double)(stop.QuadPart - start.QuadPart);
		cout << "GPU Time: " << delayGPU << ", " << endl;

		dataset_file << length << "," << (delayGPU > delayCPU ? 0 : 1) << endl;

		/*************Free Host Memory**************/
		for (x = 0; x < experiment_count; x++) {
			delete[] arraySet1[x];
			delete[] arraySet2[x];
			delete[] outputs[x];
		}
	}
	dataset_file.close();
	return 0;
}
