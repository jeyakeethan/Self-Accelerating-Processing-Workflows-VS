#ifndef ARRAYADDITIONMODEL_H
#define ARRAYADDITIONMODEL_H

#include <ComputationalModel.h>


class ArrayAdditionModel : public ComputationalModel
{
    public:
        int *localA, *localB, *localC;
        int localL;
        ArrayAdditionModel();
        ArrayAdditionModel(int *in1, int *in2, int *out, int length);
        virtual ~ArrayAdditionModel();
        inline void setData(int *in1, int *in2, int *out, int length){
            localA = in1; localB = in2; localC = out; localL = length;
            return;
        }
    protected:

    private:
        virtual void CPUImplementation();
        virtual void GPUImplementation();
};

#endif // ARRAYADDITIONMODEL_H
