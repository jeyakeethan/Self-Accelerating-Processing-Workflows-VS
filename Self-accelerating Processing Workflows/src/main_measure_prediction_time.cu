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
#include <models/MatrixMulModel.h>
#include <models/BlurModel.h>
#include <random>
#include <string>

using namespace std;

void measure_prediction_time_matrix_mul() {
	LARGE_INTEGER start, stop, clockFreq;
	QueryPerformanceFrequency(&clockFreq);
	double delay = 0;

	MatrixMultiplicationModel<numericalType1> matrixMultiplicationModel(6);
	vector<float>* vec;
	for (int i = 0; i < EXPERIMENT_COUNT; i++) {
		vec = new vector<float>{ float(rand() % 256),float(rand() % 256),float(rand() % 256) };
		QueryPerformanceCounter(&start);
		bool pred = matrixMultiplicationModel.mlModel->predict_logic(*vec);
		QueryPerformanceCounter(&stop);
		// cout << (pred ? 1 : 0);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	cout << endl << "Matrix Multiplication Model: " << endl;
	cout << "Total time: " << delay << "\tAvg Prediction time : " << delay / EXPERIMENT_COUNT << endl;
}

void measure_prediction_time_blur_model() {
	LARGE_INTEGER start, stop, clockFreq;
	QueryPerformanceFrequency(&clockFreq);
	double delay = 0;

	BlurModel<numericalType1> blurModel(6);
	vector<float>* vec;
	for (int i = 0; i < EXPERIMENT_COUNT; i++) {
		vec = new vector<float>{ float(rand() % 256),float(rand() % 256) };
		QueryPerformanceCounter(&start);
		bool pred = blurModel.mlModel->predict_logic(*vec);
		QueryPerformanceCounter(&stop);
		// cout << (pred ? 1 : 0);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	cout << endl << "Blur Model: " << endl;
	cout << "Total time: " << delay << "\tAvg Prediction time : " << delay / EXPERIMENT_COUNT << endl;
}


void measure_prediction_time_complex_model() {
	LARGE_INTEGER start, stop, clockFreq;
	QueryPerformanceFrequency(&clockFreq);
	double delay = 0;

	ComplexModel<numericalType1> complexModel(6);
	vector<float>* vec;
	for (int i = 0; i < EXPERIMENT_COUNT; i++) {
		vec = new vector<float>{ float(rand() % 256),float(rand() % 256),float(rand() % 256) };
		QueryPerformanceCounter(&start);
		bool pred = complexModel.mlModel->predict_logic(*vec);
		QueryPerformanceCounter(&stop);
		// cout << (pred ? 1 : 0);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	cout << endl << "Complex Model: " << endl;
	cout << "Total time: " << delay << "\tAvg Prediction time : " << delay / EXPERIMENT_COUNT << endl;
}

void measure_prediction_time_matrix_mul_() {
	LARGE_INTEGER start, stop, clockFreq;
	QueryPerformanceFrequency(&clockFreq);
	double delay = 0;

	MatrixMultiplicationModel<numericalType1> matrixMultiplicationModel(6);
	
	const int dim_space_len_3d = 100;
	const int value_range = 32;

	int  index_g, dim_index, len_dataset, favor, accuracyCount = 0;
	myDim3 cpu_dim_space_3d[dim_space_len_3d];
	myDim3 gpu_dim_space_3d[dim_space_len_3d];
	myDim3 dimension;
	pandas::Dataset dataset = pandas::ReadCSV("../ml-datasets/experiment-matrix-multiplication-sorted.csv", ',', -1, 1000);
	len_dataset = dataset.labels.size();
	if (len_dataset > 20)
		for (int x = 0; x < dim_space_len_3d; x++) {
			cpu_dim_space_3d[x].x = dataset.features.at(x).at(0);
			cpu_dim_space_3d[x].y = dataset.features.at(x).at(1);
			cpu_dim_space_3d[x].z = dataset.features.at(x).at(2);
			vector<float> cpu{ (float)cpu_dim_space_3d[x].x, (float)cpu_dim_space_3d[x].y, (float)cpu_dim_space_3d[x].z };
			bool pre_cpu = matrixMultiplicationModel.mlModel->predict_logic(cpu);
			if (dataset.labels.at(x) == (pre_cpu ? 1 : 0)) {
				// cout << "same" << endl;
				accuracyCount += 1;
			}

			index_g = len_dataset - dim_space_len_3d + x;
			gpu_dim_space_3d[x].x = dataset.features.at(index_g).at(0);
			gpu_dim_space_3d[x].y = dataset.features.at(index_g).at(1);
			gpu_dim_space_3d[x].z = dataset.features.at(index_g).at(2);
			vector<float> gpu{ (float)gpu_dim_space_3d[x].x, (float)gpu_dim_space_3d[x].y, (float)gpu_dim_space_3d[x].z };
			bool pre_gpu = matrixMultiplicationModel.mlModel->predict_logic(gpu);
			if (dataset.labels.at(index_g) == (pre_gpu ? 1 : 0)) {
				// cout << "same" << endl;
				accuracyCount += 1;
			}
		}
	cout << "Accuracy: " << accuracyCount << endl;

	vector<float>* vec;
	for (int i = 0; i < EXPERIMENT_COUNT; i++) {
		favor = rand() % 2;
		dim_index = rand() % dim_space_len_3d;
		if (favor == 0) dimension = gpu_dim_space_3d[dim_index];
		else dimension = cpu_dim_space_3d[dim_index];
		vec = new vector<float>{ float(dimension.x),float(dimension.y),float(dimension.z)};
		QueryPerformanceCounter(&start);
		bool pred = matrixMultiplicationModel.mlModel->predict_logic(*vec);
		QueryPerformanceCounter(&stop);
		// cout << (pred ? 1 : 0);
		delay += (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
	}
	cout << endl << "Matrix Multiplication Model: " << endl;
	cout << "Total time: " << delay << "\tAvg Prediction time : " << delay / EXPERIMENT_COUNT << endl;
}

int main()
{
	measure_prediction_time_matrix_mul_();
	measure_prediction_time_blur_model();
	measure_prediction_time_complex_model();
	return 0;
}