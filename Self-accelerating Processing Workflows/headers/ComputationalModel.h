#ifndef COMPUTATIONALMODEL_H
#define COMPUTATIONALMODEL_H
#include <time.h>
#include <windows.h>

#include <Constants.h>
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
        LARGE_INTEGER start, stop, lastRevisedClock;

        ComputationalModel(int CPUCores);
        virtual ~ComputationalModel();
        static int m_id_counter() { static int m_id = 0; return m_id++; }
        static int obj_id_counter() { static int obj_id = 0; return obj_id++; }
        static void setOperationalMode(bool om);
        static void resetOverPeriodIfBurst(ComputationalModel *cm);
        void resetFlow();
        void execute();
        void execute(int mode);
        void executeAndLogging();
        void executeAndLogging(int mode);
        void prepareLogging();
        void setProcessor(int p);
        void clearLogs();
    protected:
    private:
        thread revisor;
        void logExTime(string str);
        virtual void CPUImplementation() = 0;
        virtual void GPUImplementation() = 0;
        virtual int* getAttributes() = 0;
};

#endif // COMPUTATIONALMODEL_H