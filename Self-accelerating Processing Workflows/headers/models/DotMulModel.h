#ifndef DOTMULTIPLICATIONMODEL_H
#define DOTMULTIPLICATIONMODEL_H

#include <ComputationalModel.h>


class DotMultiplicationModel : public ComputationalModel
{
    public:
        int *localA, *localB;
        int *localC;
        int localL;
        DotMultiplicationModel(int CPUCores);
        virtual ~DotMultiplicationModel();
        inline void invoke(int *in1, int *in2, int *out, int length){
            localA = in1; localB = in2; localC = out; localL = length;
            return;
        }
    protected:

    private:
        virtual void CPUImplementation();
        virtual void GPUImplementation();
};

#endif // DOTMULTIPLICATIONMODEL_H
