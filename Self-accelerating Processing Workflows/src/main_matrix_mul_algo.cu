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
	// write logs into file
	//string console_log_name = "../logs/Array_addtion_" + CONSOLE_LOG_FILE_NAME;
	//freopen(console_log_name.c_str(), "w", stdout);


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
	string inputNatureFile = "../logs/Matrix_multiplication_Input Nature.csv"; fileNum = 0;
	while (FILE* file = fopen(inputNatureFile.c_str(), "r")) {
		fclose(file);
		inputNatureFile = "../logs/Matrix_multiplication_Input Nature_" + to_string(++fileNum) + ".csv";
	}
	input_nature_file.open(inputNatureFile, ios_base::out);

	string timeLogFile = "../logs/Matrix_multiplication_Time.txt"; fileNum = 0;
	while (FILE* file = fopen(timeLogFile.c_str(), "r")) {
		fclose(file);
		timeLogFile = "../logs/Matrix_multiplication_Time_" + to_string(++fileNum) + ".txt";
	}
	time_log_file.open(timeLogFile);

	

	//------------- THree dimension vector addition ------------
	cout << endl << "Matrix multiplication experiments started" << endl;
	input_nature_file << "Matrix multiplication experiments started" << endl;
	time_log_file << "Matrix multiplication experiments started" << endl;

	MatrixMultiplicationModel<numericalType1> matrixMultiplicationModel(6);

	numericalType1* arraySet1[EXPERIMENT_COUNT];
	numericalType1* arraySet2[EXPERIMENT_COUNT];
	numericalType1* outputs1[EXPERIMENT_COUNT];
	numericalType1* outputs2[EXPERIMENT_COUNT];
	numericalType1* outputs3[EXPERIMENT_COUNT];
	myDim3 dimensions[EXPERIMENT_COUNT];
	myDim3 dimension;

	int length1, length2, length3, index_g, dim_index, len_dataset, accuracyCount = 0, falseNegative = 0;

	// load related dimesion spaces
	const int dim_space_len_3d = 100;
	const int value_range = 32;

	myDim3 cpu_dim_space_3d[dim_space_len_3d];
	myDim3 gpu_dim_space_3d[dim_space_len_3d];
	pandas::Dataset dataset = pandas::ReadCSV("../ml-datasets/experiment-matrix-multiplication-sorted.csv", ',', -1, 100000);
	len_dataset = dataset.labels.size();
	if (len_dataset > 20)
		for (int x = 0; x < dim_space_len_3d; x++) {
			cpu_dim_space_3d[x].x = dataset.features.at(x).at(0);
			cpu_dim_space_3d[x].y = dataset.features.at(x).at(1);
			cpu_dim_space_3d[x].z = dataset.features.at(x).at(2);
			vector<float> cpu{ (float)cpu_dim_space_3d[x].x, (float)cpu_dim_space_3d[x].y, (float)cpu_dim_space_3d[x].z };
			bool pre_cpu = matrixMultiplicationModel.mlModel->predict_logic(cpu);
			cout << "[" << cpu_dim_space_3d[x].x << "," << cpu_dim_space_3d[x].y << "," << cpu_dim_space_3d[x].z << "]" << " =\t" << dataset.labels.at(x) << ",\t" << (pre_cpu ? 1 : 0) << endl;
			if (dataset.labels.at(x) == (pre_cpu ? 1 : 0)) {
				// cout << "same" << endl;
				accuracyCount += 1;
			}
			if (dataset.labels.at(x) == 1 && !pre_cpu) {
				falseNegative++;
			}
			index_g = len_dataset - dim_space_len_3d + x;
			gpu_dim_space_3d[x].x = dataset.features.at(index_g).at(0);
			gpu_dim_space_3d[x].y = dataset.features.at(index_g).at(1);
			gpu_dim_space_3d[x].z = dataset.features.at(index_g).at(2);
			vector<float> gpu{ (float)gpu_dim_space_3d[x].x, (float)gpu_dim_space_3d[x].y, (float)gpu_dim_space_3d[x].z };
			bool pre_gpu = matrixMultiplicationModel.mlModel->predict_logic(gpu);
			cout << "[" << gpu_dim_space_3d[x].x << "," << gpu_dim_space_3d[x].y << "," << gpu_dim_space_3d[x].z << "]" << " =\t" << dataset.labels.at(index_g) << ",\t" << (pre_gpu ? 1 : 0) << endl;
			if (dataset.labels.at(index_g) == (pre_gpu ? 1 : 0)) {
				// cout << "same" << endl;
				accuracyCount += 1;
			}
			if (dataset.labels.at(index_g) == 1 && !pre_gpu) {
				falseNegative++;
			}
		}
	cout << "Accuracy: " << accuracyCount << endl;
	cout << "FalseNegative: " << falseNegative << "/" << dim_space_len_3d*2 << endl;

	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		favor = rand() % 2;
		dim_index = rand() % dim_space_len_3d;
		if (favor == 0) dimension = gpu_dim_space_3d[dim_index];
		else dimension = cpu_dim_space_3d[dim_index];

		dimensions[x] = dimension;

		length1 = dimension.x * dimension.y;
		length2 = dimension.y * dimension.z;
		length3 = dimension.x * dimension.z;
		arraySet1[x] = generate_1d_array(length1);
		arraySet2[x] = generate_1d_array(length2);

		outputs1[x] = new numericalType1[length3];
		outputs2[x] = new numericalType1[length3];
		outputs3[x] = new numericalType1[length3];

		input_nature_file << "[" << dimension.x << "," << dimension.y << "," << dimension.z << "]" << ", " << endl;		// log input nature
	}

	/*matrixMultiplicationModel.SetData(arraySet1[0], arraySet2[0], output2[0], &dimensions[0]);
	matrixMultiplicationModel.execute(2);*/

	// -------- GPU Time --------
	delay = 0;
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		QueryPerformanceCounter(&start);
		matrixMultiplicationModel.SetData(arraySet1[x], arraySet2[x], outputs1[x], &dimensions[x]);

		matrixMultiplicationModel.execute(2);
		QueryPerformanceCounter(&stop);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	elapsedTime = int(delay * 1000);
	cout << "GPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "GPU Time: " << elapsedTime << " ms" << endl << endl;

	// -------- Framework --------
	delay = 0;
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		QueryPerformanceCounter(&start);
		matrixMultiplicationModel.SetData(arraySet1[x], arraySet2[x], outputs2[x], &dimensions[x]);

		matrixMultiplicationModel.execute();
		QueryPerformanceCounter(&stop);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	elapsedTime = int(delay * 1000);
	cout << "Auto Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "Auto Time: " << elapsedTime << " ms" << endl << endl;

	// -------- CPU Time --------
	delay = 0;
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		QueryPerformanceCounter(&start);
		matrixMultiplicationModel.SetData(arraySet1[x], arraySet2[x], outputs3[x], &dimensions[x]);
		matrixMultiplicationModel.execute(1);
		QueryPerformanceCounter(&stop);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	elapsedTime = int(delay * 1000);
	cout << "CPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "CPU Time: " << elapsedTime << " ms" << endl << endl;

	// ************Free Host Memory**************
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		delete[] arraySet1[x];
		delete[] arraySet2[x];

		delete[] outputs1[x];
		delete[] outputs2[x];
		delete[] outputs3[x];
	}

	input_nature_file.close();
	time_log_file.close();

	return 0;
}
