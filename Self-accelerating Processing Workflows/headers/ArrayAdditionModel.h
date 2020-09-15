#ifndef ARRAYADDITIONMODEL_H
#define ARRAYADDITIONMODEL_H

#include <ComputationalModel.h>

template <class T>
class ArrayAdditionModel : public ComputationalModel
{
    public:
        T *localA, *localB, *localC;
        int localL;
        ArrayAdditionModel(int CPUCores);
        ~ArrayAdditionModel();
        inline void setData(T *in1, T *in2, T *out, int length){
            localA = in1; localB = in2; localC = out; localL = length;
            return;
        }
    protected:

    private:
        virtual void CPUImplementation();
        virtual void GPUImplementation();
};

#include "ArrayAdditionModel.cu"

#endif // ARRAYADDITIONMODEL_H