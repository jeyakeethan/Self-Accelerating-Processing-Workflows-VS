#ifndef COMPUTATIONALMODEL_H
#define COMPUTATIONALMODEL_H
#include <time.h>
#include <windows.h>

#include <Constants.h>
#include <MatrixMulMLModel.h>
#include <fstream>
#include <sstream>

//for async function
#include <thread>
#include <future>
using namespace std;

struct Clock { LONGLONG CPU, GPU; float CPUmean, GPUmean; };
class ComputationalModel
{
public:
	static bool operationalMode;
	int countS, countL, reviseCount, alignedCount, processor, lastProcessor, revisePeriod;
	// stringstream s;
	Clock clocks;
	int CPUCores;
	int sampleMode, model_id, obj_id;
	long long duration;
	int* predictionBoundary;
	MatrixMulMLModel * mlModel;
	LARGE_INTEGER start, stop, lastRevisedClock;

	ComputationalModel(int CPUCores);
	virtual ~ComputationalModel();
	static int m_id_counter() { static int m_id = 0; return m_id++; }
	static int obj_id_counter() { static int obj_id = 0; return obj_id++; }
	static void setOperationalMode(bool om);
	static void resetOverPeriodIfBurst(ComputationalModel* cm);
	static void checkMLModel(ComputationalModel* cm);
	static void trainML(ComputationalModel* cm);
	void resetFlow();
	void execute();
	void execute(int mode);
	void executeAndLogging();
	void executeAndLogging(int mode);
	void executeByML();
	void executeByMLAndLogging();
	void prepareLogging();
	void setProcessor(int p);
	void clearLogs();
	void logExTime(string str);
	bool catchOutlier(int * attr);
protected:
private:
	thread resetOperator, mlTrainer;
	//void logExTime(string str);
	virtual void CPUImplementation() = 0;
	virtual void GPUImplementation() = 0;

	/**
	* developer has to return a array of values that have impact in the workload of the task that has been set
	* the first value of the pointer would be the length of the array and the attributes are followed
	* every subclass implementing this class must have to implement this virtual method
	**/
	virtual int* getAttributes() = 0;
	inline bool isBoundedTask() {
		int* attr = getAttributes();
		for (int i = 1; i <= attr[0]; i++)
			if (attr[i] > predictionBoundary[i])
				return false;
		return true;
	}
};

#endif // COMPUTATIONALMODEL_H