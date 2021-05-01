#ifndef ARRAYADDITIONMODEL_H
#define ARRAYADDITIONMODEL_H

#include <ComputationalModel.h>

template <class T>
class ArrayAdditionModel : public ComputationalModel
{
    public:
        T *localA, *localB, *localC;
        int localL;
        vector<float> *attr;

        ArrayAdditionModel(int CPUCores);
        ~ArrayAdditionModel();

        inline void SetData(T *in1, T *in2, T *out, int length){
            localA = in1; localB = in2; localC = out; localL = length;
            attr = new vector<float>{ (float)length };
            return;
        }

    protected:

    private:
        virtual void CPUImplementation();
        virtual void GPUImplementation();
        virtual vector<float> getAttributes();
        virtual vector<float> getAttributesBatch();
};

#include "../src/models/ArrayAddModel.cu"

#endif // ARRAYADDITIONMODEL_H