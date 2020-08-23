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
        void execute(int mode = -1);
        void setProcessor(int p);
        int countS;
        int countL;
        Clock clocks;
        int processor;
    protected:
    private:
        int _id;
        virtual void CPUImplementation() = 0;
        virtual void GPUImplementation() = 0;
};

#endif // COMPUTATIONALMODEL_H
