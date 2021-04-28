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
#include <models/MatrixMulModel.h>
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



	//------------- THree dimension vector addition ------------
	cout << "Three Dimension experiments started" << endl;
	input_nature_file << "Three Dimension experiments started" << endl;
	time_log_file << "Three Dimension experiments started" << endl;

	ArrayAdditionModel<numericalType1> arrayAdditionModel(6);
	MatrixMultiplicationModel<numericalType1> matrixMultiplicationModel(6);

	numericalType1* arrayAddSet1[EXPERIMENT_COUNT];
	numericalType1* arrayAddSet2[EXPERIMENT_COUNT];
	numericalType1* arrayAddSet3[EXPERIMENT_COUNT];
	numericalType1* arrayAddSet4[EXPERIMENT_COUNT];
	numericalType1* outputAdd1[EXPERIMENT_COUNT];
	numericalType1* outputAdd2[EXPERIMENT_COUNT];
	numericalType1* outputMul[EXPERIMENT_COUNT];
	int arrayLength1[EXPERIMENT_COUNT];
	int arrayLength2[EXPERIMENT_COUNT];
	myDim3 dimensions[EXPERIMENT_COUNT];
	myDim3 dimension;

	int length1, length2, length3, index_g, dim_index, len_dataset;

	// load related dimesion spaces
	const int dim_space_len_3d = 20;
	const int value_range = 32;

	myDim3 cpu_dim_space_3d[dim_space_len_3d];
	myDim3 gpu_dim_space_3d[dim_space_len_3d];
	pandas::Dataset dataset = pandas::ReadCSV("../ml-datasets/matrix-multiplication.csv", ',', -1, 1000);
	len_dataset = dataset.labels.size();
	if (len_dataset > 20)
		for (int x = 0; x < dim_space_len_3d; x++) {
			/*cpu_dim_space_3d[x].x = dataset.features.at(x).at(0);
			cpu_dim_space_3d[x].y = dataset.features.at(x).at(1);
			cpu_dim_space_3d[x].z = dataset.features.at(x).at(2);*/
			//cout << "[" << cpu_dim_space_3d[x].x << "," << cpu_dim_space_3d[x].y << cpu_dim_space_3d[x].z << "]" << " = " << dataset.labels.at(x) << ", " << matrixMultiplicationModel.mlModel->predict(new vector<float>{ (float)cpu_dim_space_3d[x].x, (float)cpu_dim_space_3d[x].y, (float)cpu_dim_space_3d[x].z }) << endl;

			//cout << "[" << cpu_dim_space_3d[x].x << "," << cpu_dim_space_3d[x].y << cpu_dim_space_3d[x].z << "]" << " = " << dataset.labels.at(x) << ", " << matrixMultiplicationModel.mlModel->predict(new vector<float>{ (float)cpu_dim_space_3d[x].x, (float)cpu_dim_space_3d[x].y, (float)cpu_dim_space_3d[x].z }) << endl;

			index_g = len_dataset - dim_space_len_3d + x;
			gpu_dim_space_3d[x].x = dataset.features.at(index_g).at(0);
			gpu_dim_space_3d[x].y = dataset.features.at(index_g).at(1);
			gpu_dim_space_3d[x].z = dataset.features.at(index_g).at(2);
			//cout << "[" << gpu_dim_space_3d[x].x << "," << gpu_dim_space_3d[x].y << gpu_dim_space_3d[x].z << "]" << " = " << dataset.labels.at(x) << ", " << matrixMultiplicationModel.mlModel->predict(new vector<float>{ (float)gpu_dim_space_3d[x].x, (float)gpu_dim_space_3d[x].y, (float)gpu_dim_space_3d[x].z }) << endl;

			cout << "[" << gpu_dim_space_3d[x].x << "," << gpu_dim_space_3d[x].y << "," << gpu_dim_space_3d[x].z << "]" << " = " << dataset.labels.at(index_g) << ", " << matrixMultiplicationModel.mlModel->predict(new vector<float>{ (float)gpu_dim_space_3d[x].x, (float)gpu_dim_space_3d[x].y, (float)gpu_dim_space_3d[x].z }) << endl;
		}

	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		/*favor = rand() % 2;
		dim_index = rand() % dim_space_len_3d;
		if (favor == 0) dimension = cpu_dim_space_3d[dim_index];
		else dimension = gpu_dim_space_3d[dim_index];*/

		dim_index = rand() % dim_space_len_3d;
		dimension = gpu_dim_space_3d[dim_index];

		dimensions[x] = dimension;

		length1 = dimension.x * dimension.y;
		length2 = dimension.y * dimension.z;
		length3 = dimension.x * dimension.z;
		arrayLength1[x] = length1;
		arrayAddSet1[x] = generate_1d_array(length1);
		arrayAddSet2[x] = generate_1d_array(length1);
		arrayLength2[x] = length2;
		arrayAddSet3[x] = generate_1d_array(length2);
		arrayAddSet4[x] = generate_1d_array(length2);
		outputAdd1[x] = new numericalType1[length1];
		outputAdd2[x] = new numericalType1[length2];
		outputMul[x] = new numericalType1[length3];

		//input_nature_file << "[" << dimension.x << "," << dimension.y << "]" << ", " << endl;		// log input nature
	}

	// -------- Framework --------
	QueryPerformanceCounter(&start);
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		int len1 = arrayLength1[x];
		int len2 = arrayLength2[x];
		arrayAdditionModel.SetData(arrayAddSet1[x], arrayAddSet2[x], outputAdd1[x], len1);
		arrayAdditionModel.SetData(arrayAddSet3[x], arrayAddSet4[x], outputAdd2[x], len2);
		arrayAdditionModel.execute();
		matrixMultiplicationModel.SetData(outputAdd1[x], outputAdd2[x], outputMul[x], &dimensions[x]);
		matrixMultiplicationModel.execute();

	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "\nAuto Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "Auto Time: " << elapsedTime << " ms" << endl << endl;

	// -------- CPU Time --------
	QueryPerformanceCounter(&start);
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		int len1 = arrayLength1[x];
		int len2 = arrayLength2[x];
		arrayAdditionModel.SetData(arrayAddSet1[x], arrayAddSet2[x], outputAdd1[x], len1);
		arrayAdditionModel.SetData(arrayAddSet3[x], arrayAddSet4[x], outputAdd2[x], len2);
		arrayAdditionModel.execute(1);
		matrixMultiplicationModel.SetData(outputAdd1[x], outputAdd2[x], outputMul[x], &dimensions[x]);
		matrixMultiplicationModel.execute(1);
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "CPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "CPU Time: " << elapsedTime << " ms" << endl << endl;

	// -------- GPU Time --------
	QueryPerformanceCounter(&start);
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		int len1 = arrayLength1[x];
		int len2 = arrayLength2[x];
		arrayAdditionModel.SetData(arrayAddSet1[x], arrayAddSet2[x], outputAdd1[x], len1);
		arrayAdditionModel.SetData(arrayAddSet3[x], arrayAddSet4[x], outputAdd2[x], len2);
		arrayAdditionModel.execute(2);
		matrixMultiplicationModel.SetData(outputAdd1[x], outputAdd2[x], outputMul[x], &dimensions[x]);
		matrixMultiplicationModel.execute(2);
	}
	QueryPerformanceCounter(&stop);
	delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	elapsedTime = int(delay * 1000);
	cout << "GPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "GPU Time: " << elapsedTime << " ms" << endl << endl;

	// ************Free Host Memory**************
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		delete[] arrayAddSet1[x];
		delete[] arrayAddSet2[x];
		delete[] arrayAddSet3[x];
		delete[] arrayAddSet4[x];
		delete[] outputMul[x];
	}

	input_nature_file.close();
	time_log_file.close();


	return 0;
}
