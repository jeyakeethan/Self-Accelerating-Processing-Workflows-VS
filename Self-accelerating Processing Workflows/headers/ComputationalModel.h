#ifndef COMPUTATIONALMODEL_H
#define COMPUTATIONALMODEL_H
#include <time.h>
#include <windows.h>

//for async function
#include <thread>
#include <future>
using namespace std;

struct Clock { LONGLONG CPU, GPU; float CPUmean, GPUmean; };
class ComputationalModel
{
    public:
        int countS, countL, reviseCount, alignedCount, processor, lastProcessor, revisePeriod;
        Clock clocks;
        int CPUCores;
        int sampleMode, id_;
        LARGE_INTEGER start, stop, lastRevisedClock;

        ComputationalModel(int CPUCores);
        virtual ~ComputationalModel();
        void resetFlow();
        void execute();
        void execute(int mode);
        void setProcessor(int p);
        static void resetOverPeriodIfBurst(ComputationalModel *cm);
    protected:
    private:
        thread revisor;
        virtual void CPUImplementation() = 0;
        virtual void GPUImplementation() = 0;
};

#endif // COMPUTATIONALMODEL_H