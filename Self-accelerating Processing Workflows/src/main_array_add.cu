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
#include <random>
#include <string>

using namespace std;
int main()
{
	freopen(("Array_addtion_" + CONSOLE_LOG_FILE_NAME).c_str(), "w", stdout);	// write logs into file

	LARGE_INTEGER start, stop, clockFreq;
	ofstream input_nature_file;
	ofstream time_log_file;
	QueryPerformanceFrequency(&clockFreq);
	double delay;
	int elapsedTime;
	int fileNum;


	/*------- Write Input Nature into File -------*/
	string inputNatureFile = "../logs/Array_addition_Input Nature.csv"; fileNum = 0;
	while (FILE* file = fopen(inputNatureFile.c_str(), "r")) {
		fclose(file);
		inputNatureFile = "../logs/Array_addition_Input Nature_" + to_string(++fileNum) + ".csv";
	}
	input_nature_file.open(inputNatureFile);

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

	numericalType1** arraySet1 = new numericalType1 * [EXPERIMENT_COUNT];
	numericalType1** arraySet2 = new numericalType1 * [EXPERIMENT_COUNT];
	int* arrayLength = new int[EXPERIMENT_COUNT];
	int x, k, length;

	srand(5);		// Random Seed Value

	int widthCount = 0;
	bool iSmall = true;
	short int favor;
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		favor = rand() % 2;
		if (favor == 0) length = rand() % SMALL_ARRAY_MAX_LENGTH + 1;
		else length = rand() % (ARRAY_MAX_LENGTH - SMALL_ARRAY_MAX_LENGTH) + SMALL_ARRAY_MAX_LENGTH + 1;
		arrayLength[x] = length;
		arraySet1[x] = generate_1d_array(length);
		arraySet2[x] = generate_1d_array(length);

		input_nature_file << length << "," << endl;		// log input nature
	}

	numericalType1* output;
	/*-------- Framework - ArrayAdditionModel --------*/
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		int len = arrayLength[x];
		output = new numericalType1[len];
		arrayAdditionModel.invoke(arraySet1[x], arraySet2[x], output, len);
		arrayAdditionModel.execute();
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "\nAuto Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "Auto Time: " << elapsedTime << " ms" << endl << endl;

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
	delete[] arraySet1;
	delete[] arraySet2;


/*------------- Two dimension vector addition ------------*/
cout << "Two Dimension experiments started" << endl;
input_nature_file << "Two Dimension experiments started" << endl;
time_log_file << "Two Dimension experiments started" << endl;





	input_nature_file.close();
	time_log_file.close();

	return 0;
}
