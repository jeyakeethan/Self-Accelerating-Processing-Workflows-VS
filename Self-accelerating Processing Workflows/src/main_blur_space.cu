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
#include <models/BlurModel.h>
#include "lodepng.h"
#include <random>
#include <string>

using namespace std;
int main()
{
	//string console_log_name = "../logs/blur_" + CONSOLE_LOG_FILE_NAME;
	//freopen(console_log_name.c_str(), "w", stdout);	// write logs into file

	srand(5);		// Random Seed Value

	LARGE_INTEGER start, stop, clockFreq;
	QueryPerformanceFrequency(&clockFreq);
	ofstream dataset_file;
	ofstream time_log_file;
	double delayCPU, delayGPU;
	int elapsedTime;
	int fileNum;
	short favor;
	unsigned error;

	/*------- Write Input Nature into File -------*/
	string inputNatureFile = "../ml-datasets/blur.csv";
	dataset_file.open(inputNatureFile, ios_base::out);

	string timeLogFile = "../logs/Blur_dataset_Time.txt"; fileNum = 0;
	while (FILE* file = fopen(timeLogFile.c_str(), "r")) {
		fclose(file);
		timeLogFile = "../logs/Blur_dataset_Time" + to_string(++fileNum) + ".txt";
	}
	time_log_file.open(timeLogFile);

	/*------------- Single dimension vector addition ------------*/
	cout << "BlurModel experiments started" << endl;
	time_log_file << "BlurModel experiments started" << endl;

	BlurModel <unsigned char> blurModel(6);

	const int experiment_count = 5;
	const int levels = 12;
	const int number_entries = levels * levels;
	const int value_range = 256;
	const int step = 10;
	int i_2, i_3;
	unsigned char* arraySet1[experiment_count];
	unsigned char* outputs[experiment_count];
	int arrayLength[number_entries];
	int x, y, z, k, i = 0, length;
	for (int height = step; height < levels * step + 1; height += step) {
		for (int width = step; width < levels * step + 1; width += step, i++) {
			length = height * width * 3;
			arrayLength[i] = length;
			for (x = 0; x < experiment_count; x++) {
				arraySet1[x] = generate_1d_array_char(length, value_range);
				outputs[x] = new unsigned char[length];
			}

			/*-------- CPU Time - ArrayAdditionModel --------*/
			QueryPerformanceCounter(&start);
			for (x = 0; x < experiment_count; x++) {
				blurModel.SetData(arraySet1[x], outputs[x], width, height);
				blurModel.execute(1);
			}
			QueryPerformanceCounter(&stop);
			delayCPU = (double)(stop.QuadPart - start.QuadPart);
			//cout << "CPU Time: " << delayCPU << ", ";
			//time_log_file << "CPU Time: " << delayCPU << ", ";

			blurModel.SetData(arraySet1[0], outputs[0], width, height);
			blurModel.execute(2);
			/*-------- GPU Time - ArrayAdditionModel --------*/
			QueryPerformanceCounter(&start);
			for (x = 0; x < experiment_count; x++) {
				blurModel.SetData(arraySet1[x], outputs[x], width, height);
				blurModel.execute(2);
			}
			QueryPerformanceCounter(&stop);
			delayGPU = (double)(stop.QuadPart - start.QuadPart);
			//cout << "GPU Time: " << delayGPU << ", " << endl;
			//time_log_file << "GPU Time: " << delayGPU << ", " << endl;

			dataset_file << width << "," << height << "," << (delayGPU > delayCPU ? 0 : 1) << endl;


			/*----------- Save experimented images ------------
			for (x = 0; x < experiment_count; x++) {
				// Prepare data for output
				vector<unsigned char> out_image;
				for (int i = 0; i < length; ++i) {
					out_image.push_back(outputs[x][i]);
					if ((i + 1) % 3 == 0) {
						out_image.push_back(255);
					}
				}
				// save image
				char output_file[60];
				sprintf(output_file, "../output/dataset-experiments/output_%i_%i_%i.png", width, height, x);
				error = lodepng::encode(output_file, out_image, width, height);

				//if there's an error, display it
				if (error) cout << "encoder error " << error << ": " << lodepng_error_text(error) << endl;
			}
			*/

			/*************Free Host Memory**************/
			for (x = 0; x < experiment_count; x++) {
				delete[] arraySet1[x];
				delete[] outputs[x];
			}
		}
	}
	dataset_file.close();
	time_log_file.close();
	return 0;
}

