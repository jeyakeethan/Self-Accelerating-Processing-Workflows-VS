#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <ComputationalModel.h>
#include <MatrixMulMLModel.h>
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

bool ComputationalModel::operationalMode = false;
ComputationalModel::ComputationalModel(int CPUCores_) :CPUCores(CPUCores_) {
	obj_id = obj_id_counter();
	resetFlow();
	resetOperator = thread(&ComputationalModel::resetOverPeriodIfBurst, this);
	resetOperator.detach();

	// ml related codes
	mlModel = new MatrixMulMLModel();
	mlTrainer = thread([this] {checkMLModel();});
	mlTrainer.detach();
}

inline void ComputationalModel::resetFlow() {
	countS = 1;
	countL = 1;
	reviseCount = REVISE_COUNT_MIN;
	alignedCount = -1;
	lastProcessor = processor;
	processor = -1;
	revisePeriod = REVISE_PERIOD;
	clocks = { 0, 0 };
	prediction_empty_slot = 0;
	sampleMode = 2;
	outlier_count = 0;
	// id_ = int(&*this);
}

ComputationalModel::~ComputationalModel() {
	resetOperator.~thread();
	mlTrainer.~thread();
	Logger::close();
	//TO DO; log present values for using next boot
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

// Auto mode execution
void ComputationalModel::execute() {
	switch (processor) {
		case 1:
			// catch ouliers and send them to the GPU to avoid severe letency 
			if (catchOutlier(getAttributes())) {
				GPUImplementation();
				return;
			}
			//cout << "Hello CPU" << endl;
			CPUImplementation();
			countL++;
			break;
		case 2:
			//cout << "Hello GPU" << endl;
			GPUImplementation();
			countL++;
			break;
		case -1:
			// catch ouliers and send them to the GPU to avoid severe letency 
			if (catchOutlier(getAttributes())) {
				GPUImplementation();
				return;
			}
			// cout << "Hello CPU" << endl;
			QueryPerformanceCounter(&start);
			CPUImplementation();
			QueryPerformanceCounter(&stop);
			clocks.CPU += stop.QuadPart - start.QuadPart;
			// cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
			if (++countS > SAMPLE_COUNT) {
				if (--sampleMode == 0) {
					if (clocks.CPU > clocks.GPU) {
						processor = 2;
						reviseCount += REVISE_COUNT_STEP * ++alignedCount;
					}
					else {
						processor = 1;
						reviseCount = REVISE_COUNT_MIN;
						alignedCount = 0;
					}
					lastRevisedClock.QuadPart = stop.QuadPart;
				}
				else {
					processor = -2; // processor = (processor - 1) % 3;
					countS = 1;
				}
			}
			return;
		case -2:
			// cout << "Hello GPU" << endl;
			QueryPerformanceCounter(&start);
			GPUImplementation();
			QueryPerformanceCounter(&stop);
			clocks.GPU += stop.QuadPart - start.QuadPart;
			// cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
			if (++countS > SAMPLE_COUNT) {
				if (--sampleMode == 0) {
					if (clocks.CPU > clocks.GPU) {
						processor = 2;
						reviseCount = REVISE_COUNT_MIN;
						alignedCount = 0;
					}
					else {
						processor = 1;
						reviseCount += REVISE_COUNT_STEP * ++alignedCount;
					}
					lastRevisedClock.QuadPart = stop.QuadPart;
				}
				else {
					processor = -1; // processor = (processor - 1) % 3;
					countS = 1;
				}
			}
			return;
		default:
			sampleMode = 2;
			processor = -1;
	}
	if (countL > reviseCount) {
		sampleMode = 2;
		countS = 1;
		countL = 1;
		processor = -processor;
		clocks = { 0, 0 };
		//            cout << endl;
	}
}

void ComputationalModel::executeAndLogging()
{
	LARGE_INTEGER start_cover;
	LARGE_INTEGER stop_cover;
	QueryPerformanceCounter(&start_cover);
	switch (processor) {
		case 1:
			// catch ouliers and send them to the GPU to avoid severe letency 
			if (catchOutlier(getAttributes())) {
				GPUImplementation();
				return;
			}
			// cout << "Hello CPU" << endl;
			CPUImplementation();
			countL++;
			break;
		case 2:
			// cout << "Hello GPU" << endl;
			GPUImplementation();
			countL++;
			break;
		case -1:
			// catch ouliers and send them to the GPU to avoid severe letency 
			if (catchOutlier(getAttributes())) {
				GPUImplementation();
				return;
			}
			// cout << "Hello CPU" << endl;
			QueryPerformanceCounter(&start);
			CPUImplementation();
			QueryPerformanceCounter(&stop);
			clocks.CPU += stop.QuadPart - start.QuadPart;;
			// cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
			if (++countS > SAMPLE_COUNT) {
				if (--sampleMode == 0) {
					if (clocks.CPU > clocks.GPU) {
						cout << "Switched to GPU" << endl;
						processor = 2;
						reviseCount += REVISE_COUNT_STEP * ++alignedCount;
					}
					else {
						cout << "CPU continue processing" << endl;
						processor = 1;
						reviseCount = REVISE_COUNT_MIN;
						alignedCount = 0;
					}
					lastRevisedClock.QuadPart = stop.QuadPart;
					//                    cout << "REVISE_COUNT: " << reviseCount << endl;
	//                    cout << alignedCount << "," << clocks.CPU << "," << clocks.GPU << endl << endl;
				}
				else {
					processor = -2; // processor = (processor - 1) % 3;
					countS = 1;
				}
			}
			break;
		case -2:
			// cout << "Hello GPU" << endl;
			QueryPerformanceCounter(&start);
			GPUImplementation();
			QueryPerformanceCounter(&stop);
			clocks.GPU += stop.QuadPart - start.QuadPart;;
			// cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
			if (++countS > SAMPLE_COUNT) {
				if (--sampleMode == 0) {
					if (clocks.CPU > clocks.GPU){
						cout << "GPU continue processing" << endl;
						processor = 2;
						reviseCount = REVISE_COUNT_MIN;
						alignedCount = 0;
					}
					else {
						cout << "Switched to CPU" << endl;
						processor = 1;
						reviseCount += REVISE_COUNT_STEP * ++alignedCount;
					}
					lastRevisedClock.QuadPart = stop.QuadPart;
				}
				else {
					processor = -1; // processor = (processor - 1) % 3;
					countS = 1;
				}
			}
			break;
		default:
			sampleMode = 2;
			processor = -1;
	}

	QueryPerformanceCounter(&stop_cover);

	duration = stop_cover.QuadPart - start_cover.QuadPart;

	stringstream s;
	s << typeid(*this).name() << ",";
	s << attributeToString(getAttributes());

	if (processor == 1 || processor == -1) {
		s << 0 << ",";
	}
	else {
		s << 1 << ",";
	}
	s << duration << endl;
	logExTime(s.str());

	if (processor == -2 || processor == -1)
		return;
	if (countL > reviseCount) {
		sampleMode = 2;
		countS = 1;
		countL = 1;
		processor = -processor;
		clocks = { 0, 0 };
	}
}

void ComputationalModel::executeByML() {
	if (mlModel->predict(getAttributes()) == 1) {
		CPUImplementation();
	}
	else {
		GPUImplementation();
	}
}

void ComputationalModel::executeByMLAndLogging() {
	stringstream s;
	s << typeid(*this).name() << ",";
	s << attributeToString(getAttributes());
	if (mlModel->predict(getAttributes()) == 1) {
		s << 0 << ",";
		QueryPerformanceCounter(&start);
		CPUImplementation();
		QueryPerformanceCounter(&stop);
	}
	else {
		s << 1 << ",";
		QueryPerformanceCounter(&start);
		GPUImplementation();
		QueryPerformanceCounter(&stop);
	}
	duration = stop.QuadPart - start.QuadPart;
	s << duration << endl;
	logExTime(s.str());
}

void ComputationalModel::setProcessor(int p) {
	processor = p;
}

/* static method run by a thread to reset the flow if the input stream is burst and sparsed */
void ComputationalModel::resetOverPeriodIfBurst(ComputationalModel* cm)
{
	LARGE_INTEGER now, frequency, reviseBoundary;
	QueryPerformanceFrequency(&frequency);
	reviseBoundary.QuadPart = frequency.QuadPart * cm->revisePeriod;
	while (true) {
		this_thread::sleep_for(chrono::seconds(cm->revisePeriod));
		QueryPerformanceCounter(&now);
		if (now.QuadPart - cm->lastRevisedClock.QuadPart > reviseBoundary.QuadPart) {
			cm->resetFlow();    // reset the flow
		}
	}
}

void ComputationalModel::checkMLModel() {
	while (true) {
		string file_ml = "ml_Trainer.cache";
		ifstream mlTrainerReadFile(file_ml);
		int lastUpdatedTime = 0;
		mlTrainerReadFile >> lastUpdatedTime;
		mlTrainerReadFile.close();
		if (lastUpdatedTime != 0 && currentTimeMillis() - lastUpdatedTime > MONTH) {
			trainML();
		}
		ofstream mlTrainerWriteFile(file_ml);
		mlTrainerWriteFile << currentTimeMillis() << endl;
		mlTrainerWriteFile.close();
		this_thread::sleep_for(chrono::seconds(ML_TRAIN_CHECK_PERIOD));
	}
}

bool ComputationalModel::catchOutlier(vector<float>* attr) {
	for (int i = 0; i < prediction_empty_slot; i++) {
		if (std::equal(std::begin(*cached_predictions[i]), std::end(*cached_predictions[i]),
			std::begin(*attr), std::end(*attr),
			[](int a, int b)->bool {return a >= b; }))
			return false;
	}
	if (mlModel->predict(attr) == 0) {
		CPUGPULOG << "OC";// cout << "An outlier caught " << attributeToString(attr) << endl;
		if (++outlier_count > MAX_OUTLIERS_LIMIT) {
			outlier_count = 0;
			lastProcessor = processor;
			processor = 2;
			clocks = { 0, 0 };
			countS = 1;
			countL = 1;
			alignedCount = -1;
			reviseCount = REVISE_COUNT_MIN;
			revisePeriod = REVISE_PERIOD;
			sampleMode = 0;

			CPUGPULOG << "SM";// cout << "switched due to max caught" << endl;
			return true;
		}
		return true;
	}
	else {
		outlier_count = 0;
		for (int i = 0; i < prediction_empty_slot; i++) {
			if (std::equal(std::begin(*cached_predictions[i]), std::end(*cached_predictions[i]),
				std::begin(*attr), std::end(*attr),
				[](int a, int b)->bool {return a < b; })) {
				*cached_predictions[i] = *attr;
				return false;
			}
		}
		if (prediction_empty_slot < NUMBER_OF_PREDICTIONS_TO_BE_CACHED - 1)
			cached_predictions[prediction_empty_slot++] = attr;
		return false;
	}
	return false;
}

void ComputationalModel::trainML() {
	mlModel->trainModel();
}

void ComputationalModel::setOperationalMode(bool om) {
	operationalMode = om;
}

void ComputationalModel::logExTime(string str) {
	if (!Logger::isOpen()) {
		Logger::open(LOG_FILE_NAME);
	}
	Logger::write(str);
}

void ComputationalModel::clearLogs() {
	Logger::clearLogs(LOG_FILE_NAME);
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