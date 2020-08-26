#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <Constants.h>
#include <ComputationalModel.h>
#include <ArrayAdditionModel.h>
#include <DotMultiplicationModel.h>
#include <random>
#include <fstream>
#include <string>


using namespace std;
int main()
{
    fstream inputFile;
    inputFile.open("test.txt", ios::in);
    int n;
    vector<int> inputA, inputB, output;
    if (inputFile.is_open()) {

        string line, stringArray1, stringArray2;
        while (getline(inputFile, line)) {
            cout << line;
            int pos = line.find(' ');
            n = stoi(line.substr(0, pos));
            line.erase(0, pos + 1);

            int x = 0;
            while (x < n) {
                pos = line.find(',');
                inputA.push_back(stoi(line.substr(0, pos)));
                line.erase(0, pos + 1);
            }

            x = 0;
            while (x < n) {
                pos = line.find(',');
                inputB.push_back(stoi(line.substr(0, pos)));
                line.erase(0, pos + 1);
            }
            inputB.push_back(stoi(line));
        }
    }
    inputFile.close();
     for(int i=0; i<n; i++)
        cout << inputA[i] << ", ";


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
    }*/
    return 0;
}
