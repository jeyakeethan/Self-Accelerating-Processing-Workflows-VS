#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <Constants.h>
#include <ComputationalModel.h>
#include <ArrayAdditionModel.h>
#include <DotMultiplicationModel.h>
#include <random>


using namespace std;
int main()
{

    int inputA[N];
    int inputB[N];
    int output[N];

    ArrayAdditionModel arrayAdditionModel;

    for (int exp = 0; exp < EXPERIMENT_COUNT; exp++) {
        for (int k = 0; k < N; k++) {
            inputA[k] = rand() % RANGE_OF_INT_VALUES;
            inputB[k] = rand() % RANGE_OF_INT_VALUES;
        }

        arrayAdditionModel.setData(inputA, inputB, output, N);
        arrayAdditionModel.execute();
        // for(int i=0; i<N; i++)
        //    cout << output[i] << ", ";
        
    }

    /*
    int inputA[N];
    int inputB[N];

    DotMultiplicationModel dotMultiplicationModel;

    for (int exp = 0; exp < EXPERIMENT_COUNT; exp++) {
        int out = 0;
        for (int k = 0; k < N; k++) {
            inputA[k] = rand() % RANGE_OF_INT_VALUES;
            inputB[k] = rand() % RANGE_OF_INT_VALUES;
        }

        dotMultiplicationModel.setData(inputA, inputB, &out, N);
        dotMultiplicationModel.execute(1);
        printf("%d", out);
    }
    return 0;
    */
}
