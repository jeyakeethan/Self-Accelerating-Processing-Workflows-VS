#ifndef ARRAYADDITION2DMODEL_H
#define ARRAYADDITION2DMODEL_H

#include <ComputationalModel.h>

template <class T>
class ArrayAddition2DModel : public ComputationalModel
{
public:
    T* localA, * localB, * localC;
    int localRow, localCol;
    ArrayAddition2DModel(int CPUCores);
    ~ArrayAddition2DModel();
    inline void invoke(T* in1, T* in2, T* out, int lengthx, int lengthy) {
        localA = in1; localB = in2; localC = out; localRow = lengthx, localCol = lengthy;
        return;
    }
protected:

private:
    virtual void CPUImplementation();
    virtual void GPUImplementation();
    virtual vector<float>* getAttributes();
};

#include "../src/models/ArrayAddModel.cu"

#endif // ARRAYADDITIONMODEL_H