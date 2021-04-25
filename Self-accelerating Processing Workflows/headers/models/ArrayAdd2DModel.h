#ifndef ARRAYADDITION2DMODEL_H
#define ARRAYADDITION2DMODEL_H

#include <ComputationalModel.h>
#include <Constants.h>

template <class T>
class ArrayAddition2DModel : public ComputationalModel
{
public:
    T* localA, * localB, * localC;
    int localRow, localCol;
    ArrayAddition2DModel(int CPUCores);
    ~ArrayAddition2DModel();
    inline void invoke(T* in1, T* in2, T* out, int row, int col) {
        localA = in1; localB = in2; localC = out; localRow = row; localCol = col;
        return;
    }
protected:

private:
    virtual void CPUImplementation();
    virtual void GPUImplementation();
    virtual vector<float>* getAttributes();
    virtual vector<float>* getAttributesBatch();
};

#include "../src/models/ArrayAdd2DModel.cu"

#endif // ARRAYADDITIONMODEL_H