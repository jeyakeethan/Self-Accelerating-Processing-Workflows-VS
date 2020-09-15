#ifndef MATRIXMULTIPLICATIONMODEL_H
#define MATRIXMULTIPLICATIONMODEL_H

#include <ComputationalModel.h>
#include <Constants.h>

template <class T>
class MatrixMultiplicationModel : public ComputationalModel
{
    public:
        T *localA, *localB, *localC;
        dim3 *localMD;
        MatrixMultiplicationModel();
        MatrixMultiplicationModel(T *mat1, T *mat2, T *out, dim3 *matricesDim);
        ~MatrixMultiplicationModel();
        inline void setData(T *mat1, T *mat2, T *out, dim3 *matricesDim){
            localA = mat1; localB = mat2; localC = out; localMD = matricesDim;
            return;
        }
    protected:

    private:
        virtual void CPUImplementation();
        virtual void GPUImplementation();
};

#include "MatrixMultiplicationModel.cu"

#endif // MATRIXMULTIPLICATIONMODEL_H