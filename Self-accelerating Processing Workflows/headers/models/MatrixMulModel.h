#ifndef MATRIXMULTIPLICATIONMODEL_H
#define MATRIXMULTIPLICATIONMODEL_H

#include <ComputationalModel.h>
#include <Constants.h>
#include <vector>
using namespace std;

template <class T>
class MatrixMultiplicationModel: public ComputationalModel
{
    public:
        T *localA, *localB, *localC;
        myDim3 *localMD;
        vector<float> * attr;
        MatrixMultiplicationModel(int CPUCores);
        ~MatrixMultiplicationModel();
        inline void setData(T *mat1, T *mat2, T *out, myDim3 *matricesDim){
            localA = mat1; localB = mat2; localC = out; localMD = matricesDim;
            attr = new vector<float>{3, (float)matricesDim->x, (float)matricesDim->y, (float)matricesDim->z};
            return;
        }
    protected:

    private:
        virtual void CPUImplementation();
        virtual void GPUImplementation();
        virtual vector<float>* getAttributes();
};

#include "../src/models/MatrixMulModel.cu"

#endif // MATRIXMULTIPLICATIONMODEL_H