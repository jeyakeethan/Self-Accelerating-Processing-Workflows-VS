#ifndef COMPUTATIONALMODEL_H
#define COMPUTATIONALMODEL_H
#include <time.h>
#include <windows.h>

struct Clock { LONGLONG CPU, GPU; float CPUmean, GPUmean; };
class ComputationalModel
{
    public:
        ComputationalModel();
        virtual ~ComputationalModel();
        void resetFlow();
        void execute();
        void execute(int mode);
        void setProcessor(int p);
        int countS, countL, reviseCount, alignedCount, processor, lastProcessor, revisePeriod;
        Clock clocks;
        int sampleMode;
        int id_;
    protected:
    private:
        virtual void CPUImplementation() = 0;
        virtual void GPUImplementation() = 0;
};

#endif // COMPUTATIONALMODEL_H