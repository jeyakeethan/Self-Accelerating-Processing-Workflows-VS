#ifndef COMPUTATIONALMODEL_H
#define COMPUTATIONALMODEL_H
#include <time.h>

class ComputationalModel
{
    public:
        ComputationalModel();
        //virtual ~ComputationalModel();
        void execute(int mode = -1);

    protected:
    private:
        int _id;
        int processor;
        virtual void CPUImplementation() = 0;
        virtual void GPUImplementation() = 0;
};

#endif // COMPUTATIONALMODEL_H
