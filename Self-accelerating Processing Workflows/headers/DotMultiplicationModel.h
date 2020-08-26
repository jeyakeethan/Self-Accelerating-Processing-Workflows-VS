#ifndef DOTMULTIPLICATIONMODEL_H
#define DOTMULTIPLICATIONMODEL_H

#include <ComputationalModel.h>


class DotMultiplicationModel : public ComputationalModel
{
    public:
        int *localA, *localB;
        long long *localC;
        int localL;
        DotMultiplicationModel();
        DotMultiplicationModel(int *in1, int *in2, long long *out, int length);
        virtual ~DotMultiplicationModel();
        inline void setData(int *in1, int *in2, long long *out, int length){
            localA = in1; localB = in2; localC = out; localL = length;
            return;
        }
    protected:

    private:
        virtual void CPUImplementation();
        virtual void GPUImplementation();
};

#endif // DOTMULTIPLICATIONMODEL_H
