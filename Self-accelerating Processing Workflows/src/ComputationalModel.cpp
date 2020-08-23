#include <iostream>
#include <ComputationalModel.h>
#include <WorkflowController.h>
#include <Constants.h>

//for time measure
#include <windows.h>
#include <time.h>
#include <stdio.h>

//for async function
#include <thread>
#include <future>

using namespace std;

ComputationalModel::ComputationalModel(): processor(0){
    WorkflowController::registerModel(this);
    countCPU = 0;
    countGPU = 0;
    count = 0;
}

// ComputationalModel::~ComputationalModel(){}

void ComputationalModel::execute(int mode)
{
    LARGE_INTEGER start, stop;

    if(processor == 0){
        QueryPerformanceCounter(&start);
        CPUImplementation();
        QueryPerformanceCounter(&stop);
        CPUclocks[countCPU]= stop.QuadPart - start.QuadPart;
        cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
        // if (countCPU > REVISE_COUNT)
        //    async(std::launch::async, [&]() { WorkflowController::updateCPUTime(this); });
        countCPU = countCPU +1%LAST_N_TIME;
    }
    else {
        QueryPerformanceCounter(&start);
        GPUImplementation();
        QueryPerformanceCounter(&stop);
        GPUclocks[countGPU] = stop.QuadPart - start.QuadPart;
        cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
        // if (countGPU > REVISE_COUNT)
        //    async(std::launch::async, [&]() { WorkflowController::updateGPUTime(this); });
        countGPU = countGPU + 1 % LAST_N_TIME;
    }
    if (++count > RESET_COUNT)
        async(std::launch::async, [&]() { WorkflowController::changeProcessor(this); });
}

void ComputationalModel::setProcessor(int p) {
    processor = p;
}
