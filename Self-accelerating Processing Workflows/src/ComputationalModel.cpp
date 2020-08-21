#include <iostream>
#include <ComputationalModel.h>

//for time measure
#include <time.h>
#include <stdio.h>

//for async function
#include <thread>
#include <future>

using namespace std;

// ComputationalModel::ComputationalModel(){}

// ComputationalModel::~ComputationalModel(){}


static long tCPU = 0;
static long tGPU = 0;

void ComputationalModel::execute(int mode)
{
    clock_t start, stop;

    start = clock();
    if(processor == 0){
        CPUImplementation();
    }
    else {
        GPUImplementation();
    }

    stop = clock();
    async(std::launch::async, [&]() { ComputationalModel::updateResults(start, stop, processor); });
    //ComputationalModel::updateResults(start, stop, freq, tCPU, tGPU);
}

void ComputationalModel::updateResults(clock_t start, clock_t stop, int processor){
    // To a async function
    clock_t delay = stop - start;
    float time = (float)delay/CLOCKS_PER_SEC*1000000;
    if(processor==0){
        cout << "CPU Time: " << time << " ns" << endl;
    } else {
        cout << "GPU Time: " << time << " ns" << endl;
    }

    // cout << tCPU <<","<<tGPU << endl << endl;
}
