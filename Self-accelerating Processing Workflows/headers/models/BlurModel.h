#ifndef BLURMODEL_H
#define BLURMODEL_H

#include <ComputationalModel.h>
#include <Constants.h>
#include <vector>

using namespace std;

template <class T>
class BlurModel: public ComputationalModel
{
    public:
        unsigned char *input_image, *output_image;
        int width, height;
        vector<float> * attr;
        BlurModel(int CPUCores);
        ~BlurModel();
        inline void SetData(unsigned char *input, unsigned char *output, int widthT, int heightT){
            input_image = input;
            output_image = output;
            width = widthT;
            height = heightT;
            attr = new vector<float>{(float)widthT, (float)heightT};
            return;
        }
    protected:

    private:
        virtual void CPUImplementation();
        virtual void GPUImplementation();
        virtual vector<float>* getAttributes();
        virtual vector<float>* getAttributesBatch();
        void getError(cudaError_t err);
};

#include "../src/models/BlurModel.cu"

#endif // BLURMODEL_H