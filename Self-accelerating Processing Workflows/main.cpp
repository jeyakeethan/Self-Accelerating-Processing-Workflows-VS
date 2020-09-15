#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

// measure time
#include <windows.h>
#include <time.h>

#include <Constants.h>
#include <ComputationalModel.h>
#include <ArrayAdditionModel.h>
#include <DotMultiplicationModel.h>
#include <MatrixMultiplicationModel.h>
#include <random>
#include <string>

using namespace std;
int main()
{
    LARGE_INTEGER start, stop, clockFreq;
    ofstream outfile;
    QueryPerformanceFrequency(&clockFreq);
    double delay;
    int elapsedTime;

    MatrixMultiplicationModel<numericalType1> matmulmodel(4);
    numericalType1 mat1 [6]= { 1, 3, 7,8,4,3 };
    numericalType1 mat2 [6]= { 1, 3, 7,8,3,2 };
    numericalType1 out[4];
    matmulmodel.setData(mat1, mat2, out, new myDim3(2, 3, 2));

    QueryPerformanceCounter(&start);
    matmulmodel.execute(1);
    QueryPerformanceCounter(&stop);
    delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
    elapsedTime = int(delay * 1000);
    cout << endl << "CPU Time: " << elapsedTime << " ms" << endl;
    for (int t = 0; t < 4; t++) {
        cout << out[t] << endl;
        out[t] = 0;
    }

    QueryPerformanceCounter(&start);
    matmulmodel.execute(2);
    QueryPerformanceCounter(&stop);
    delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
    elapsedTime = int(delay * 1000);
    cout << endl << "GPU Time: " << elapsedTime << " ms" << endl;
    for (int t = 0; t < 4; t++)
        cout << out[t] << endl;

    return 0;
}
