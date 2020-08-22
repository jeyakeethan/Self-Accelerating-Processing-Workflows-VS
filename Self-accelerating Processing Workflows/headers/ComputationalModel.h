#ifndef COMPUTATIONALMODEL_H
#define COMPUTATIONALMODEL_H
#include <time.h>

struct Clock { clock_t CPU, GPU; float CPUmean, GPUmean; };
class ComputationalModel
{
    public:
        ComputationalModel();
        //virtual ~ComputationalModel();
        void execute(int mode = -1);
        void setProcessor(int p);
        int counts;
        Clock clocks;
    protected:
    private:
        int _id;
        int processor;
        virtual void CPUImplementation() = 0;
        virtual void GPUImplementation() = 0;
};

#endif // COMPUTATIONALMODEL_H
