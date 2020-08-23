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
    clocks = { 0, 0, 0.0, 0.0};
    countS = 0;
    countL = 0;
}

// ComputationalModel::~ComputationalModel(){}

void ComputationalModel::execute(int mode)
{
    LARGE_INTEGER start, stop;

    if(processor == 0){
        QueryPerformanceCounter(&start);
        CPUImplementation();
        QueryPerformanceCounter(&stop);
        clocks.CPU += stop.QuadPart - start.QuadPart;
        cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
        if (countS > REVISE_COUNT)
            async(std::launch::async, [&]() { WorkflowController::updateCPUTime(this); });
    }
    else {
        QueryPerformanceCounter(&start);
        GPUImplementation();
        QueryPerformanceCounter(&stop);
        clocks.GPU += stop.QuadPart - start.QuadPart;
        cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
        if (countS > REVISE_COUNT)
            async(std::launch::async, [&]() { WorkflowController::updateGPUTime(this); });
    }
    countS++;
    countL++;

    //ComputationalModel::updateResults(start, stop, freq, tCPU, tGPU);
}

void ComputationalModel::setProcessor(int p) {
    processor = p;
}
