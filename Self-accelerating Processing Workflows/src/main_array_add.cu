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

	numericalType1** arraySet1 = new numericalType1 * [EXPERIMENT_COUNT];
	numericalType1** arraySet2 = new numericalType1 * [EXPERIMENT_COUNT];
	int* arrayLength = new int[EXPERIMENT_COUNT];
	int x, y, z, k, length;

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
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		delete[] arraySet1[x];
		delete[] arraySet2[x];
	}
	delete[] arraySet1;
	delete[] arraySet2;
	delete[] arrayLength;


/*------------- Two dimension vector addition ------------*/
cout << "Two Dimension experiments started" << endl;
input_nature_file << "Two Dimension experiments started" << endl;
time_log_file << "Two Dimension experiments started" << endl;

	ArrayAddition2DModel<numericalType1> arrayAddition2DModel(6);

	numericalType1** arraySetB1 = new numericalType1 * [EXPERIMENT_COUNT];
	numericalType1** arraySetB2 = new numericalType1 * [EXPERIMENT_COUNT];
	myDim2* dim_space = new myDim2[EXPERIMENT_COUNT];	//todo
	myDim2* dimensions = new myDim2[EXPERIMENT_COUNT];
	numericalType1** outputB;
	int dim_index;
	myDim2 dimension;

	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		favor = rand() % 2;
		if (favor == 0) dim_index = rand() % SMALL_ARRAY_MAX_LENGTH + 1;
		else dim_index = rand() % (ARRAY_MAX_LENGTH - SMALL_ARRAY_MAX_LENGTH) + SMALL_ARRAY_MAX_LENGTH + 1;

		dimension = dim_space[dim_index];
		dimensions[x] = dimension;
		length = dimension.x * dimension.y;
		arraySetB1[x] = generate_1d_array(length);
		arraySetB2[x] = generate_1d_array(length);
		outputB[x] = new numericalType1[length];

		input_nature_file << "[" << dimension.x << "," << dimension.y << "]" << ", " << endl;		// log input nature
	}

	/*-------- Framework - ArrayAdditionModel --------*/
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		arrayAddition2DModel.invoke(arraySetB1[x], arraySetB2[x], outputB[x], dimensions[x].x, dimensions[x].y);
		arrayAddition2DModel.execute();
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "\nAuto Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "Auto Time: " << elapsedTime << " ms" << endl << endl;

	/*-------- CPU Time - ArrayAdditionModel --------*/
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		arrayAddition2DModel.invoke(arraySetB1[x], arraySetB2[x], outputB[x], dimensions[x].x, dimensions[x].y);
		arrayAddition2DModel.execute(1);
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "CPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "CPU Time: " << elapsedTime << " ms" << endl << endl;

	/*-------- GPU Time - ArrayAdditionModel --------*/
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		arrayAddition2DModel.invoke(arraySetB1[x], arraySetB2[x], outputB[x], dimensions[x].x, dimensions[x].y);
		arrayAddition2DModel.execute(2);
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "GPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "GPU Time: " << elapsedTime << " ms" << endl << endl;

	/*************Free Host Memory**************/
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		delete[] arraySetB1[x];
		delete[] arraySetB2[x];
		delete[] outputB[x];
	}
	delete[] arraySetB1;
	delete[] arraySetB2;
	delete[] outputB;
	delete[] dimensions;
	delete[] dim_space;


/*

//------------- THree dimension vector addition ------------
cout << "Three Dimension experiments started" << endl;
input_nature_file << "Three Dimension experiments started" << endl;
time_log_file << "Three Dimension experiments started" << endl;

	ArrayAdditionModel3D<numericalType1> arrayAdditionModel3D(6);

	numericalType1**** arraySetC1 = new numericalType1 ***[EXPERIMENT_COUNT];
	numericalType1**** arraySetC2 = new numericalType1 ***[EXPERIMENT_COUNT];
	myDim3* dim_space_3d = new myDim3[EXPERIMENT_COUNT];	//todo
	myDim3* dimensions_3d = new myDim3[EXPERIMENT_COUNT];
	myDim3 dimension_3d;

	short int favor;
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		favor = rand() % 2;
		if (favor == 0) dim_index = rand() % SMALL_ARRAY_MAX_LENGTH + 1;
		else dim_index = rand() % (ARRAY_MAX_LENGTH - SMALL_ARRAY_MAX_LENGTH) + SMALL_ARRAY_MAX_LENGTH + 1;

		dimension_3d = dim_space_3d[dim_index];
		dimensions_3d[x] = dimension_3d;
		arraySetC1[x] = generate_3d_array(dimension_3d.x, dimension_3d.y, dimension_3d.z);
		arraySetC2[x] = generate_3d_array(dimension_3d.x, dimension_3d.y, dimension_3d.z);

		input_nature_file << "[" << dimension_3d.x << "," << dimension_3d.y <<  "," << dimension_3d.z <<"]" << ", " << endl;		// log input nature
	}

	numericalType1**** outputC;
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		myDim3 dim_3d = dimensions_3d[x];
		outputC[x] = new numericalType1 **[dim_3d.x];
		for (y = 0; y < dim_3d.x; y++) {
			outputC[x][y] = new numericalType1 * [dim_3d.y];
			for (z = 0; z < dim_3d.y; z++)
				outputC[x][y][z] = new numericalType1[dim_3d.z];
		}
	}
	//-------- Framework - ArrayAdditionModel --------
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		arrayAdditionModel3D.invoke(arraySetC1[x], arraySetC2[x], outputC[x], dimensions_3d[x]);
		arrayAdditionModel3D.execute();
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "\nAuto Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "Auto Time: " << elapsedTime << " ms" << endl << endl;

	//-------- CPU Time - ArrayAdditionModel --------
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		arrayAdditionModel3D.invoke(arraySetC1[x], arraySetC2[x], outputC[x], dimensions_3d[x]);
		arrayAdditionModel3D.execute(1);
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "CPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "CPU Time: " << elapsedTime << " ms" << endl << endl;

	//-------- GPU Time - ArrayAdditionModel --------
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		arrayAdditionModel3D.invoke(arraySetC1[x], arraySetC2[x], outputC[x], dimensions_3d[x]);
		arrayAdditionModel3D.execute(2);
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "GPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "GPU Time: " << elapsedTime << " ms" << endl << endl;

	//*************Free Host Memory**************
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		for (y = 0; y < dimensions_3d[x].x; y++) {
			delete[] arraySetC1[x][y];
			delete[] arraySetC2[x][y];
			delete[] outputC[x][y];
		}
		delete[] arraySetC1[x];
		delete[] arraySetC2[x];
		delete[] outputC[x];
	}
	delete[] arraySetC1;
	delete[] arraySetC2;
	delete[] outputC;
	delete[] dimensions_3d;
	delete[] dim_space_3d;

	input_nature_file.close();
	time_log_file.close();
	*/
	return 0;
}
