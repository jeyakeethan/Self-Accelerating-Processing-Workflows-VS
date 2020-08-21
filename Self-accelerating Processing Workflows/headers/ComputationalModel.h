#ifndef COMPUTATIONALMODEL_H
#define COMPUTATIONALMODEL_H
#include <time.h>

class ComputationalModel
{
    public:
        //ComputationalModel();
        //virtual ~ComputationalModel();
        static void updateResults(clock_t start, clock_t stop, int processor);
        void execute(int mode = -1);

    protected:

    private:
        int processor = 0;
        virtual void CPUImplementation() = 0;
        virtual void GPUImplementation() = 0;
};

#endif // COMPUTATIONALMODEL_H
