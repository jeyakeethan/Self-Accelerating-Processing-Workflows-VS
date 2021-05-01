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
#include <models/BlurModel.h>
#include <random>
#include <string>

using namespace std;
int main()
{
	//string console_log_name = "../logs/Blur_" + CONSOLE_LOG_FILE_NAME;
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
	string inputNatureFile = "../logs/Blur_Input Nature.csv"; fileNum = 0;
	while (FILE* file = fopen(inputNatureFile.c_str(), "r")) {
		fclose(file);
		inputNatureFile = "../logs/Blur_Input Nature_" + to_string(++fileNum) + ".csv";
	}
	input_nature_file.open(inputNatureFile, ios_base::out);

	string timeLogFile = "../logs/Blur_Time.txt"; fileNum = 0;
	while (FILE* file = fopen(timeLogFile.c_str(), "r")) {
		fclose(file);
		timeLogFile = "../logs/Blur_Time_" + to_string(++fileNum) + ".txt";
	}
	time_log_file.open(timeLogFile);

/*------------- Two dimension vector addition ------------*/
cout << "Blur experiments" << endl;
input_nature_file << "Blur experiments" << endl;
time_log_file << "Blur experiments" << endl;

	BlurModel<numericalType1> blurModel(6);
	
	unsigned char* arraySet1[EXPERIMENT_COUNT];
	unsigned char* outputs[EXPERIMENT_COUNT];
	myDim2 dimensions[EXPERIMENT_COUNT];
	myDim2 dimension;

	int length, index_g, dim_index, len_dataset, accuracyCount = 0;

	// load related dimesion spaces
	const int dim_space_len_2d = 100;
	const int value_range = 256;

	myDim2 cpu_dim_space_2d[dim_space_len_2d];
	myDim2 gpu_dim_space_2d[dim_space_len_2d];
	pandas::Dataset dataset = pandas::ReadCSV("../ml-datasets/blur.csv", ',', -1, 1000);
	len_dataset = dataset.labels.size();
	if (len_dataset > 20) {
		for (int x = 0; x < dim_space_len_2d; x++) {
			cpu_dim_space_2d[x].x = dataset.features.at(x).at(0);
			cpu_dim_space_2d[x].y = dataset.features.at(x).at(1);
			vector<float> cpu{ (float)cpu_dim_space_2d[x].x, (float)cpu_dim_space_2d[x].y };
			bool pre_cpu = blurModel.mlModel->predict_logic(&cpu);
			cout << "[" << cpu_dim_space_2d[x].x << "," << cpu_dim_space_2d[x].y << "]" << " =\t" << dataset.labels.at(x) << ",\t" << (pre_cpu ? 1 : 0) << endl;
			if (dataset.labels.at(x) == (pre_cpu ? 1 : 0)) {
				cout << "same" << endl;
				accuracyCount += 1;
			}

			index_g = len_dataset - dim_space_len_2d + x;
			gpu_dim_space_2d[x].x = dataset.features.at(index_g).at(0);
			gpu_dim_space_2d[x].y = dataset.features.at(index_g).at(1);
			vector<float> gpu{ (float)gpu_dim_space_2d[x].x, (float)gpu_dim_space_2d[x].y };
			bool pre_gpu = blurModel.mlModel->predict_logic(&gpu);
			cout << "[" << gpu_dim_space_2d[x].x << "," << gpu_dim_space_2d[x].y << "]" << " =\t" << dataset.labels.at(index_g) << ",\t" << (pre_gpu ? 1 : 0) << endl;
			if (dataset.labels.at(index_g) == (pre_gpu ? 1 : 0)) {
				cout << "same" << endl;
				accuracyCount += 1;
			}
		}
	}
	cout << "Accuracy" << accuracyCount << endl;
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		favor = rand() % 2;
		dim_index = rand() % dim_space_len_2d;
		if (favor == 0) dimension = cpu_dim_space_2d[dim_index];
		else dimension = gpu_dim_space_2d[dim_index];

		dimensions[x] = dimension;

		length = dimension.x * dimension.y * 3;
		arraySet1[x] = generate_1d_array_char(length, value_range);
		outputs[x] = new unsigned char[length];

		input_nature_file << "[" << dimension.x << "," << dimension.y << "]" << ", " << endl;		// log input nature
	}

	// -------- GPU Time --------
	delay = 0;
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		blurModel.SetData(arraySet1[x], outputs[x], dimensions[x].x, dimensions[x].y);
		QueryPerformanceCounter(&start);
		blurModel.execute(2);
		QueryPerformanceCounter(&stop);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	elapsedTime = int(delay * 1000);
	cout << "GPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "GPU Time: " << elapsedTime << " ms" << endl << endl;

	// -------- Framework --------
	delay = 0;
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		blurModel.SetData(arraySet1[x], outputs[x], dimensions[x].x, dimensions[x].y);
		QueryPerformanceCounter(&start);
		blurModel.execute();
		QueryPerformanceCounter(&stop);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	elapsedTime = int(delay * 1000);
	cout << "\nAuto Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "Auto Time: " << elapsedTime << " ms" << endl << endl;

	// -------- CPU Time --------
	delay = 0;
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		blurModel.SetData(arraySet1[x], outputs[x], dimensions[x].x, dimensions[x].y);
		QueryPerformanceCounter(&start);
		blurModel.execute(1);
		QueryPerformanceCounter(&stop);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	elapsedTime = int(delay * 1000);
	cout << "CPU Time: " << elapsedTime << " ms" << endl << endl;
	time_log_file << "CPU Time: " << elapsedTime << " ms" << endl << endl;

	
	// ************Free Host Memory**************
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		delete[] arraySet1[x];
		delete[] outputs[x];
	}

	input_nature_file.close();
	time_log_file.close();


	return 0;
}
