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
{
	//string console_log_name = "../logs/Array_addtion_" + CONSOLE_LOG_FILE_NAME;
	//freopen(console_log_name.c_str(), "w", stdout);	// write logs into file

	srand(5);		// Random Seed Value

	LARGE_INTEGER start, stop, clockFreq;
	ofstream input_nature_file;
	ofstream time_log_file;
	QueryPerformanceFrequency(&clockFreq);
	double delay;
	int elapsedTime;
	int fileNum;
	short favor;


	/*------- Write Input Nature into File -------*/
	string inputNatureFile = "../logs/Array_addition_Input Nature.csv"; fileNum = 0;
	while (FILE* file = fopen(inputNatureFile.c_str(), "r")) {
		fclose(file);
		inputNatureFile = "../logs/Array_addition_Input Nature_" + to_string(++fileNum) + ".csv";
	}
	input_nature_file.open(inputNatureFile, ios_base::out);

	string timeLogFile = "../logs/Array_addition_Time.txt"; fileNum = 0;
	while (FILE* file = fopen(timeLogFile.c_str(), "r")) {
		fclose(file);
		timeLogFile = "../logs/Array_addition_Time_" + to_string(++fileNum) + ".txt";
	}
	time_log_file.open(timeLogFile);

	/*------------- Single dimension vector addition ------------*/
	cout << "One Dimension experiments started" << endl;
	input_nature_file << "One Dimension experiments started" << endl;
	time_log_file << "One Dimension experiments started" << endl;

	ArrayAdditionModel<numericalType1> arrayAdditionModel(6);

	numericalType1* arraySet1 [EXPERIMENT_COUNT];
	numericalType1* arraySet2 [EXPERIMENT_COUNT];
	int* arrayLength = new int[EXPERIMENT_COUNT];
	int x, y, z, k, i, length;

	for (i = 1; i < 101; i++) {
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			/*favor = rand() % 2;
			if (favor == 0) length = rand() % SMALL_ARRAY_MAX_LENGTH + 1;
			else length = rand() % (ARRAY_MAX_LENGTH - SMALL_ARRAY_MAX_LENGTH) + SMALL_ARRAY_MAX_LENGTH + 1;*/
			arrayLength[x] = 1000 * i;

			arraySet1[x] = generate_1d_array(1000 * i);
			arraySet2[x] = generate_1d_array(1000 * i);

			//input_nature_file << length << "," << endl;		// log input nature
		}


		numericalType1* output;

		/*-------- CPU Time - ArrayAdditionModel --------*/
		QueryPerformanceCounter(&start);
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			int len = arrayLength[x];
			output = new numericalType1[len];
			arrayAdditionModel.invoke(arraySet1[x], arraySet2[x], output, len);
			arrayAdditionModel.execute(1);
		}
		QueryPerformanceCounter(&stop);
		delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
		elapsedTime = int(delay * 1000);
		cout << "CPU Time: " << elapsedTime << " ms" << endl << endl;
		time_log_file << "CPU Time: " << elapsedTime << " ms" << endl << endl;

		/*-------- GPU Time - ArrayAdditionModel --------*/
		QueryPerformanceCounter(&start);
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			int len = arrayLength[x];
			output = new numericalType1[len];
			arrayAdditionModel.invoke(arraySet1[x], arraySet2[x], output, len);
			arrayAdditionModel.execute(2);
		}
		QueryPerformanceCounter(&stop);
		delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
		elapsedTime = int(delay * 1000);
		cout << "GPU Time: " << elapsedTime << " ms" << endl << endl;
		time_log_file << "GPU Time: " << elapsedTime << " ms" << endl << endl;

		/*************Free Host Memory**************/
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			delete[] arraySet1[x];
			delete[] arraySet2[x];
		}
		/*delete[] arraySet1;
		delete[] arraySet2;
		delete[] arrayLength;*/
	}

	return 0;
}
