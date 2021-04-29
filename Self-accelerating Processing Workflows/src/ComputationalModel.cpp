#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <ComputationalModel.h>
#include <MLModel.h>
#include <Logger.h>
//for time measure
#include <windows.h>
#include <time.h>
#include <stdio.h>

//for async function
#include <thread>
#include <future>
#include "CurrentCPULoad.cpp"

using namespace std;

__int64 currentTimeMillis();

ComputationalModel::ComputationalModel(int CPUCores_, string model_name):CPUCores(CPUCores_) {
	obj_id = obj_id_counter();
	resetFlow();
	name = model_name;

	// ml related codes
	mlModel = new MLModel(model_name);

	/* // Auto train model periodically
	mlTrainer = thread([this] {checkMLModel();});
	mlTrainer.detach();
	*/
}

ComputationalModel::~ComputationalModel() {
	// mlTrainer.~thread();

	logExTime(CPUGPULOG.str());
	Logger::close();

	delete mlModel;

	//TO DO; log present values for using next boot
}

inline void ComputationalModel::resetFlow() {
	lastProcessor = processor;
	prediction_empty_slot = 0;
	outlier_count = 0;
}


// Mannual mode execution
void ComputationalModel::execute(int mode) {
	switch (mode) {
	case 1:
		// cout << "Hello CPU" << endl;
		CPUImplementation();
		break;
	case 2:
		// cout << "Hello GPU" << endl;
		GPUImplementation();
		break;
	}
}

void ComputationalModel::executeAndLogging(int mode)
{
	// first class name, object id, data and finally the execution time
	LARGE_INTEGER start_cover;
	LARGE_INTEGER stop_cover;
	QueryPerformanceCounter(&start_cover);

	switch (mode) {
	case 1:
		// cout << "Hello CPU" << endl;
		CPUImplementation();
		break;
	case 2:
		// cout << "Hello GPU" << endl;
		GPUImplementation();
		break;
	}

	QueryPerformanceCounter(&stop_cover);

	duration = stop_cover.QuadPart - start_cover.QuadPart;
	stringstream s;
	s << typeid(*this).name() << ",";
	s << attributeToString(getAttributes());
	if (mode == 1)
		s << 0 << ",";
	else
		s << 1 << ",";
	s << duration << endl;
	logExTime(s.str());
}

void ComputationalModel::execute() {
	if (mlModel->predict_logic(getAttributes())) {
		GPUImplementation();
	}
	else {
		CPUImplementation();
	}
}

void ComputationalModel::executeAndLogging() {
	stringstream s;
	s << typeid(*this).name() << ",";
	s << attributeToString(getAttributes());
	s << 0 << ",";
	if (mlModel->predict_logic(getAttributes()) == 1) {
		QueryPerformanceCounter(&start);
		CPUImplementation();
		QueryPerformanceCounter(&stop);
	}
	else {
		QueryPerformanceCounter(&start);
		GPUImplementation();
		QueryPerformanceCounter(&stop);
	}
	duration = stop.QuadPart - start.QuadPart;
	s << duration << endl;
	logExTime(s.str());
}

void ComputationalModel::checkMLModel() {
	while (true) {
		string file_ml = "ml_Trainer.cache";
		ifstream mlTrainerReadFile(file_ml);
		int lastUpdatedTime = 0;
		mlTrainerReadFile >> lastUpdatedTime;
		mlTrainerReadFile.close();
		if (lastUpdatedTime != 0 && currentTimeMillis() - lastUpdatedTime > MONTH) {
			trainMLModel();
		}
		ofstream mlTrainerWriteFile(file_ml);
		mlTrainerWriteFile << currentTimeMillis() << endl;
		mlTrainerWriteFile.close();
		this_thread::sleep_for(chrono::seconds(ML_TRAIN_CHECK_PERIOD));
	}
}

void ComputationalModel::trainMLModel() {
	mlModel->trainModel();
}

void ComputationalModel::logExTime(string str) {
	if (!Logger::isOpen()) {
		Logger::open("../logs/framework/" + name + ".txt");
	}
	Logger::write(str);
}

void ComputationalModel::clearLogs() {
	Logger::clearLogs("../logs/framework/" + name + ".txt");
}

string ComputationalModel::attributeToString(vector<float>* attr) {
	stringstream s;
	for (int i = 1; i <= (*attr)[0]; i++) {
		s << (*attr)[i] << ",";
	}
	return s.str();
}

__int64 currentTimeMillis() {
	FILETIME f;
	GetSystemTimeAsFileTime(&f);
	(long long)f.dwHighDateTime;
	__int64 nano = ((__int64)f.dwHighDateTime << 32LL) + (__int64)f.dwLowDateTime;
	return (nano - 116444736000000000LL) / 10000;
}