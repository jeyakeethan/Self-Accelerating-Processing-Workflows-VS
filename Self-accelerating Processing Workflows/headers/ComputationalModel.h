#ifndef COMPUTATIONALMODEL_H
#define COMPUTATIONALMODEL_H
#include <time.h>
#include <windows.h>

#include <Constants.h>
#include <MLModel.h>
#include <fstream>
#include <sstream>
#include <vector>

//for async function
#include <thread>
#include <future>
using namespace std;

struct Clock { LONGLONG CPU, GPU;};
class ComputationalModel
{
public:
	static bool operationalMode;
	stringstream CPUGPULOG;
	int processor, lastProcessor = -1, revisePeriod;
	string name;
	int CPUCores;
	int model_id, obj_id, prediction_empty_slot = 0;
	long long duration;
	int outlier_count = 0;
	MLModel * mlModel;
	LARGE_INTEGER start, stop;
	vector<float> *cached_predictions[NUMBER_OF_PREDICTIONS_TO_BE_CACHED];
	vector<float> cached_prediction_last;
	ComputationalModel(int CPUCores, string model_name);
	virtual ~ComputationalModel();
	static int m_id_counter() { static int m_id = 0; return m_id++; }
	static int obj_id_counter() { static int obj_id = 0; return obj_id++; }
	string static attributeToString(vector<float> attr);
	void checkMLModel();
	void resetFlow();
	void execute();
	void executeBatch();
	void execute(int mode);
	void executeAndLogging();
	void executeAndLogging(int mode);
	void clearLogs();
	void logExTime(string str);
	void trainMLModel();
protected:
private:
	thread resetOperator, mlTrainer;
	virtual void CPUImplementation() = 0;
	virtual void GPUImplementation() = 0;

	/**
	* developer has to return a array of values that have impact in the workload of the task that has been set
	* the first value of the pointer would be the length of the array and the attributes are followed
	* every subclass implementing this class must have to implement this virtual method
	**/
	virtual vector<float> getAttributes() = 0;
	virtual vector<float> getAttributesBatch() = 0;

};

#endif // COMPUTATIONALMODEL_H