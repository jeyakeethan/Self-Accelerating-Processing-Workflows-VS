#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "random_array_generator.cpp"

#include "pandas.h"
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

	const int BOUNDARY_POINT = 100000;
	numericalType1* arraySet1[EXPERIMENT_COUNT];
	numericalType1* arraySet2[EXPERIMENT_COUNT];
	numericalType1* outputs[EXPERIMENT_COUNT];
	int arrayLength[EXPERIMENT_COUNT];
	int x, y, z, k, length;

	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		favor = rand() % 2;
		if (favor == 0) length = rand() % BOUNDARY_POINT + 1;
		else length = rand() % BOUNDARY_POINT + BOUNDARY_POINT + 1;
		arrayLength[x] = length;
		arraySet1[x] = generate_1d_array(length);
		arraySet2[x] = generate_1d_array(length);
		outputs[x] = new numericalType1[length];

		input_nature_file << length << ",";		// log input nature
	}

	// -------- Framework - ArrayAdditionModel --------
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		int len = arrayLength[x];
		arrayAdditionModel.invoke(arraySet1[x], arraySet2[x], outputs[x], len);
		arrayAdditionModel.execute();
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "\nAuto Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "Auto Time: " << elapsedTime << " ms" << endl << endl;

	// -------- CPU Time - ArrayAdditionModel --------
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		int len = arrayLength[x];
		arrayAdditionModel.invoke(arraySet1[x], arraySet2[x], outputs[x], len);
		arrayAdditionModel.execute(1);
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "CPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "CPU Time: " << elapsedTime << " ms" << endl << endl;

	// -------- GPU Time - ArrayAdditionModel --------
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		int len = arrayLength[x];
		arrayAdditionModel.invoke(arraySet1[x], arraySet2[x], outputs[x], len);
		arrayAdditionModel.execute(2);
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "GPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "GPU Time: " << elapsedTime << " ms" << endl << endl;

	// *************Free Host Memory**************
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		delete[] arraySet1[x];
		delete[] arraySet2[x];
		delete[] outputs[x];
	}

/*------------- Two dimension vector addition ------------*/
cout << "Two Dimension experiments started" << endl;
input_nature_file << "Two Dimension experiments started" << endl;
time_log_file << "Two Dimension experiments started" << endl;

	ArrayAddition2DModel<numericalType1> arrayAddition2DModel(6);

	numericalType1* arraySetB1[EXPERIMENT_COUNT];
	numericalType1* arraySetB2[EXPERIMENT_COUNT];

	// load related dimesion spaces
	const int dim_space_len_2d = 10;
	
	int index_g;

	myDim2 cpu_dim_space_2d[dim_space_len_2d];
	myDim2 gpu_dim_space_2d[dim_space_len_2d];
	pandas::Dataset dataset = pandas::ReadCSV("../ml-datasets/Array-Addition2D.csv", ',', -1, 1000);
	for (x = 0; x < dim_space_len_2d; x++) {
		len_dataset = dataset.labels.size();

		cpu_dim_space_2d[x].x = dataset.features.at(x).at(0);
		cpu_dim_space_2d[x].y = dataset.features.at(x).at(1);

		index_g = len_dataset - dim_space_len_2d + len_dataset
		gpu_dim_space_2d[x].x = dataset.features.at(index_g).at(0);
		gpu_dim_space_2d[x].y = dataset.features.at(index_g).at(1);
	}
	myDim2 dimensions[EXPERIMENT_COUNT];
	numericalType1** outputB;
	int dim_index;
	myDim2 dimension;

	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		favor = rand() % 2;
		dim_index = rand() % dim_space_len_2d;
		if (favor == 0) dimension = cpu_dim_space_2d[dim_index];
		else dimension = gpu_dim_space_2d[dim_index];

		dimensions[x] = dimension;
		length = dimension.x * dimension.y;
		arraySetB1[x] = generate_1d_array(length);
		arraySetB2[x] = generate_1d_array(length);
		outputB[x] = new numericalType1[length];

		input_nature_file << "[" << dimension.x << "," << dimension.y << "]" << ", " << endl;		// log input nature
	}

	// -------- Framework - ArrayAdditionModel --------
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

	// -------- CPU Time - ArrayAdditionModel --------
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

	// -------- GPU Time - ArrayAdditionModel --------
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

	// ************Free Host Memory**************
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		delete[] arraySetB1[x];
		delete[] arraySetB2[x];
		delete[] outputB[x];
	}
	delete[] arraySetB1;
	delete[] arraySetB2;
	delete[] outputB;
	delete[] dimensions;
	delete[] cpu_dim_space_2d;
	delete[] gpu_dim_space_2d;


/*------------- THree dimension vector addition ------------
cout << "Three Dimension experiments started" << endl;
input_nature_file << "Three Dimension experiments started" << endl;
time_log_file << "Three Dimension experiments started" << endl;

	ArrayAddition3DModel<numericalType1> arrayAddition3DModel(6);

	numericalType1* arraySetC1[EXPERIMENT_COUNT];
	numericalType1* arraySetC2 [EXPERIMENT_COUNT];

	// load related dimesion spaces
	const int dim_space_len_3d = 10;

	myDim3 cpu_dim_space_3d[dim_space_len_3d];
	myDim3 gpu_dim_space_3d[dim_space_len_3d];
	//TO DO

	myDim3 dimensions_3d[EXPERIMENT_COUNT];
	myDim3 dimension_3d;


	numericalType1** outputB;

	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		favor = rand() % 2;
		dim_index = rand() % dim_space_len_3d;
		if (favor == 0) dimension_3d = cpu_dim_space_3d[dim_index];
		else dimension_3d = gpu_dim_space_3d[dim_index];
		dimensions_3d[x] = dimension_3d;

		length = dimension_3d.x * dimension_3d.y * dimension_3d.z;
		arraySetC1[x] = generate_1d_array(length);
		arraySetC2[x] = generate_1d_array(length);

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
		arrayAddition3DModel.invoke(arraySetC1[x], arraySetC2[x], outputC[x], dimensions_3d[x]);
		arrayAddition3DModel.execute();
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "\nAuto Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "Auto Time: " << elapsedTime << " ms" << endl << endl;

	//-------- CPU Time - ArrayAdditionModel --------
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		arrayAddition3DModel.invoke(arraySetC1[x], arraySetC2[x], outputC[x], dimensions_3d[x]);
		arrayAddition3DModel.execute(1);
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "CPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "CPU Time: " << elapsedTime << " ms" << endl << endl;

	//-------- GPU Time - ArrayAdditionModel --------
	QueryPerformanceCounter(&start);
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		arrayAddition3DModel.invoke(arraySetC1[x], arraySetC2[x], outputC[x], dimensions_3d[x]);
		arrayAddition3DModel.execute(2);
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "GPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "GPU Time: " << elapsedTime << " ms" << endl << endl;

	//*************Free Host Memory**************
	for (x = 0; x < EXPERIMENT_COUNT; x++) {
		delete[] arraySetC1[x];
		delete[] arraySetC2[x];
		delete[] outputC[x];
	}
	delete[] arraySetC1;
	delete[] arraySetC2;
	delete[] outputC;
	delete[] dimensions_3d;
	delete[] cpu_dim_space_3d;
	delete[] gpu_dim_space_3d;
*/



/*------------- Size based experiments started ------------*/
cout << "Size based experiments started" << endl;
input_nature_file << "Size based experiments started" << endl;
time_log_file << "Size based experiments started" << endl;

	const int start_len = 10000, len_step = 10000;	// BOUNDARY_POINT = 100000; above
	for (length = 0; length < BOUNDARY_POINT * 2; length += len_step) {
		input_nature_file << length << endl;		// log input nature

		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			arraySet1[x] = generate_1d_array(length);
			arraySet2[x] = generate_1d_array(length);
			outputs[x] = new numericalType1(length);
		}

		// -------- Framework - ArrayAdditionModel --------
		QueryPerformanceCounter(&start);
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			arrayAdditionModel.invoke(arraySet1[x], arraySet2[x], outputs[x], length);
			arrayAdditionModel.execute();
		}
		QueryPerformanceCounter(&stop);
		delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
		elapsedTime = int(delay * 1000);
		cout << "\nAuto Time: " << elapsedTime << " ms" << endl << endl;
		time_log_file << "Auto Time: " << elapsedTime << " ms" << endl << endl;

		// -------- CPU Time - ArrayAdditionModel --------
		QueryPerformanceCounter(&start);
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			arrayAdditionModel.invoke(arraySet1[x], arraySet2[x], outputs[x], length);
			arrayAdditionModel.execute(1);
		}
		QueryPerformanceCounter(&stop);
		delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
		elapsedTime = int(delay * 1000);
		cout << "CPU Time: " << elapsedTime << " ms" << endl << endl;
		time_log_file << "CPU Time: " << elapsedTime << " ms" << endl << endl;

		// -------- GPU Time - ArrayAdditionModel --------
		QueryPerformanceCounter(&start);
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			arrayAdditionModel.invoke(arraySet1[x], arraySet2[x], outputs[x], length);
			arrayAdditionModel.execute(2);
		}
		QueryPerformanceCounter(&stop);
		delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
		elapsedTime = int(delay * 1000);
		cout << "GPU Time: " << elapsedTime << " ms" << endl << endl;
		time_log_file << "GPU Time: " << elapsedTime << " ms" << endl << endl;

		//************Free Host Memory**************
		for (x = 0; x < EXPERIMENT_COUNT; x++) {
			delete[] arraySet1[x];
			delete[] arraySet2[x];
			delete[] outputs[x];
		}
	}


	input_nature_file.close();
	time_log_file.close();


	return 0;
}
