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
#include <models/ComplexModel.h>
#include <random>
#include <string>

using namespace std;
int main()
{	
	// write logs into file
	//string console_log_name = "../logs/Array_addtion_" + CONSOLE_LOG_FILE_NAME;
	//freopen(console_log_name.c_str(), "w", stdout);	// write logs into file

	srand(5);		// Random Seed Value

	LARGE_INTEGER start, stop, clockFreq;
	ofstream input_nature_file;
	QueryPerformanceFrequency(&clockFreq);
	double delay;
	int elapsedTime;
	int fileNum;
	short favor;


	/*------- Write Input Nature into File -------*/
	string inputNatureFile = "../logs/Complex_Model_Input Nature.csv"; fileNum = 0;
	while (FILE* file = fopen(inputNatureFile.c_str(), "r")) {
		fclose(file);
		inputNatureFile = "../logs/Complex_Model_Input Nature_" + to_string(++fileNum) + ".csv";
	}
	input_nature_file.open(inputNatureFile, ios_base::out);


	//------------- THree dimension vector addition ------------
	cout << "Complex model experiments started" << endl;

	ComplexModel<numericalType1> complexModel(6);

	numericalType1* arraySet1[EXPERIMENT_COUNT];
	numericalType1* arraySet2[EXPERIMENT_COUNT];
	numericalType1* arraySetx[EXPERIMENT_COUNT];
	numericalType1* arraySety[EXPERIMENT_COUNT];
	numericalType1* outputs1[EXPERIMENT_COUNT];
	numericalType1* outputs2[EXPERIMENT_COUNT];
	numericalType1* outputs3[EXPERIMENT_COUNT];
	myDim3 dimensions[EXPERIMENT_COUNT];
	myDim3 dimension;

	int length1, length2, length3, index_g, dim_index, len_dataset;

	// load related dimesion spaces
	const int dim_space_len_3d = 10;
	const int value_range = 32;

	myDim3 cpu_dim_space_3d[dim_space_len_3d];
	myDim3 gpu_dim_space_3d[dim_space_len_3d];
	pandas::Dataset dataset = pandas::ReadCSV("../ml-datasets/complex-model.csv", ',', -1, 1000);
	len_dataset = dataset.labels.size();
	if (len_dataset > 20)
		for (int x = 0; x < dim_space_len_3d; x++) {
			cpu_dim_space_3d[x].x = dataset.features.at(x).at(0);
			cpu_dim_space_3d[x].y = dataset.features.at(x).at(1);
			cpu_dim_space_3d[x].z = dataset.features.at(x).at(2);
			vector<float> cpu{ (float)cpu_dim_space_3d[x].x, (float)cpu_dim_space_3d[x].y, (float)cpu_dim_space_3d[x].z };
			bool pre_cpu = complexModel.mlModel->predict_logic(&cpu);
			cout << "[" << cpu_dim_space_3d[x].x << "," << cpu_dim_space_3d[x].y << "," << cpu_dim_space_3d[x].z << "]" << " =\t" << dataset.labels.at(x) << ",\t" << (pre_cpu ? 1 : 0) << endl;

			index_g = len_dataset - dim_space_len_3d + x;
			gpu_dim_space_3d[x].x = dataset.features.at(index_g).at(0);
			gpu_dim_space_3d[x].y = dataset.features.at(index_g).at(1);
			gpu_dim_space_3d[x].z = dataset.features.at(index_g).at(2);
			vector<float> gpu{ (float)gpu_dim_space_3d[x].x, (float)gpu_dim_space_3d[x].y, (float)gpu_dim_space_3d[x].z };
			bool pre_gpu = complexModel.mlModel->predict_logic(&gpu);
			cout << "[" << gpu_dim_space_3d[x].x << "," << gpu_dim_space_3d[x].y << "," << gpu_dim_space_3d[x].z << "]" << " =\t" << dataset.labels.at(index_g) << ",\t" << (pre_gpu ? 1 : 0) << endl;
		}

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
		arraySety[x] = generate_1d_array(length2);

		arraySetx[x] = generate_1d_array(length3);
		outputs1[x] = new numericalType1[length3];
		outputs2[x] = new numericalType1[length3];
		outputs3[x] = new numericalType1[length3];
		input_nature_file << "[" << dimension.x << "," << dimension.y << "," << dimension.z << "]" << ", " << endl;		// log input nature
	}

	// -------- GPU Time --------
	delay = 0;
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		QueryPerformanceCounter(&start);
		complexModel.SetData(arraySet1[x], arraySet2[x], arraySetx[x], arraySety[x], outputs1[x], &dimensions[x]);
		complexModel.execute(2);
		QueryPerformanceCounter(&stop);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	elapsedTime = int(delay * 1000);
	cout << endl << "GPU Time: " << elapsedTime << " ms" << endl << endl;

	// -------- Framework --------
	delay = 0;
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		QueryPerformanceCounter(&start);
		complexModel.SetData(arraySet1[x], arraySet2[x], arraySetx[x], arraySety[x], outputs2[x], &dimensions[x]);
		complexModel.execute();
		QueryPerformanceCounter(&stop);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	elapsedTime = int(delay * 1000);
	cout << "Auto Time: " << elapsedTime << " ms" << endl << endl;

	// -------- CPU Time --------
	delay = 0;
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		QueryPerformanceCounter(&start);
		complexModel.SetData(arraySet1[x], arraySet2[x], arraySetx[x], arraySety[x], outputs3[x], &dimensions[x]);
		complexModel.execute(1);
		QueryPerformanceCounter(&stop);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	elapsedTime = int(delay * 1000);
	cout << "CPU Time: " << elapsedTime << " ms" << endl << endl;

	// ************Free Host Memory**************
	for (int x = 0; x < EXPERIMENT_COUNT; x++) {
		delete[] arraySet1[x];
		delete[] arraySet2[x];
		delete[] arraySetx[x];
		delete[] arraySety[x];
		delete[] outputs1[x];
		delete[] outputs2[x];
		delete[] outputs3[x];
	}

	input_nature_file.close();

	return 0;
}
