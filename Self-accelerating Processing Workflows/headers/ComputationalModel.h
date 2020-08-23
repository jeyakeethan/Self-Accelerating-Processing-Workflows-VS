#ifndef COMPUTATIONALMODEL_H
#define COMPUTATIONALMODEL_H
#include <time.h>
#include <windows.h>
#include <Constants.h>

struct Clock { LONGLONG CPU, GPU; float CPUmean, GPUmean; };
class ComputationalModel
{
    public:
        ComputationalModel();
        //virtual ~ComputationalModel();
        void execute(int mode = -1);
        void setProcessor(int p);
        int countCPU;
        int countGPU;
        int CPUclocks [LAST_N_TIME];
        int GPUclocks [LAST_N_TIME];
        int count;
        int processor;
    protected:
    private:
        int _id;
        virtual void CPUImplementation() = 0;
        virtual void GPUImplementation() = 0;
};

#endif // COMPUTATIONALMODEL_H
