#pragma once
#ifndef COMPLEX_MODEL_H
#define COMPLEX_MODEL_H

#include <Constants.h>
#include <ComputationalModel.h>
#include <vector>

using namespace std;

template <class T>
class ComplexModel : public ComputationalModel {
public:
    T* localA, * localB, * localC, * localX, * localY;
    myDim3* localMD;
    vector<float>* attr;
    ComplexModel(int CPUCores);
    ~ComplexModel();
    // out = A * (B + Y) + X;
    inline void SetData(T* mat1, T* mat2, T* matx, T* maty, T* out, myDim3* matricesDim) {
        localA = mat1; localB = mat2; localX = matx; localY = maty; localC = out; localMD = matricesDim;
        attr = new vector<float>{ (float)matricesDim->x, (float)matricesDim->y, (float)matricesDim->z };
        return;
    }
protected:

private:
    virtual void CPUImplementation();
    virtual void GPUImplementation();
    virtual vector<float>* getAttributes();
    virtual vector<float>* getAttributesBatch();
};


#include "../src/models/ComplexModel.cu"

#endif