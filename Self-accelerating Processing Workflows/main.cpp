#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

// measure time
#include <windows.h>
#include <time.h>

#include <Constants.h>
#include <ComputationalModel.h>
#include <ArrayAdditionModel.h>
#include <DotMultiplicationModel.h>
#include <random>


using namespace std;
int main()
{
    LARGE_INTEGER start, stop, clockFreq;
    QueryPerformanceFrequency(&clockFreq);
    double delay;
    int elapsedTime;
    ArrayAdditionModel arrayAdditionModel;

    /*********Generate Input Stream*********/
    int ** arraySet1 = new int* [EXPERIMENT_COUNT];
    int ** arraySet2 = new int* [EXPERIMENT_COUNT];
    int* arrayLength = new int [EXPERIMENT_COUNT];
    int k, x, length, widthCount = 0, width = rand() % MAX_WIDTH_ALIGNED + 1;
    bool iSmall = true;
    for (x = 0; x < EXPERIMENT_COUNT; x++) {
        if (++widthCount > width) {
            //cout << "width: " << width << endl << endl;
            widthCount = 0;
            width = rand() % (MAX_WIDTH_ALIGNED - MIN_WIDTH_ALIGNED) + MIN_WIDTH_ALIGNED;
            iSmall = !iSmall;
        }
        if (iSmall) length = rand() % SMALL_ARRAY_MAX_LENGTH + 1;
        else length = rand() % (ARRAY_MAX_LENGTH - SMALL_ARRAY_MAX_LENGTH) + SMALL_ARRAY_MAX_LENGTH + 1;
        //cout << "length: " << length << endl;
        arrayLength[x] = length;
        int* temp1 = new int[length];
        int* temp2 = new int[length];
        arraySet1[x] = temp1;
        arraySet2[x] = temp2;
        for (k = 0; k < length; k++) {
            temp1[k] = rand() % RANGE_OF_INT_VALUES;
            temp2[k] = rand() % RANGE_OF_INT_VALUES;
        }
    }

    QueryPerformanceCounter(&start);
    for (x = 0; x < EXPERIMENT_COUNT; x++) {
        int len = arrayLength[x];
        int* output = new int[len];
        arrayAdditionModel.setData(arraySet1[x], arraySet2[x], output, len);
        arrayAdditionModel.execute();
        //for(int i=0; i<len; i++)
        //   cout << output[i] << ", ";
    }
    QueryPerformanceCounter(&stop);
    delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
    elapsedTime = int(delay * 1000);
    cout << "Self Flow Time: " << elapsedTime << " ms" << endl << endl;
    
    QueryPerformanceCounter(&start);
    for (x = 0; x < EXPERIMENT_COUNT; x++) {
        int len = arrayLength[x];
        int* output = new int[len];
        arrayAdditionModel.setData(arraySet1[x], arraySet2[x], output, len);
        arrayAdditionModel.execute(1);
        //for (int i = 0; i < len; i++)
        //    cout << output[i] << ", ";
    }
    QueryPerformanceCounter(&stop);
    delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
    elapsedTime = int(delay * 1000);
    cout << "CPU Only Time: " << elapsedTime << " ms" << endl << endl;
    
    
    QueryPerformanceCounter(&start);
    for (x = 0; x < EXPERIMENT_COUNT; x++) {
        int len = arrayLength[x];
        int* output = new int[len];
        arrayAdditionModel.setData(arraySet1[x], arraySet2[x], output, len);
        arrayAdditionModel.execute(2);
        //for (int i = 0; i < len; i++)
        //    cout << output[i] << ", ";
    }
    QueryPerformanceCounter(&stop);
    delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
    elapsedTime = int(delay * 1000);
    cout << "GPU Only Time: " << elapsedTime << " ms" << endl << endl;

    /*

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
    QueryPerformanceCounter(&stop);
    double delay = (double)(stop.QuadPart - start.QuadPart) / (double)clockFreq.QuadPart;
    int elapsedTime = int(delay * 1000);
    cout << "CPU Time: " << elapsedTime << " ms" << endl;


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
        //cout << out << endl;
    }
    return 0;
    */
}
