#ifndef COMPUTATIONALMODEL_H
#define COMPUTATIONALMODEL_H
#include <time.h>
#include <windows.h>

struct Clock { LONGLONG CPU, GPU; float CPUmean, GPUmean; };
class ComputationalModel
{
    public:
        ComputationalModel();
        //virtual ~ComputationalModel();
        void execute();
        void execute(int mode);
        void setProcessor(int p);
        int countS, countL, processor, revisePeriod;
        Clock clocks;
        int sampleMode;
    protected:
    private:
        int _id;
        virtual void CPUImplementation() = 0;
        virtual void GPUImplementation() = 0;
};

#endif // COMPUTATIONALMODEL_H
