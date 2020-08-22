#include <iostream>
#include <ComputationalModel.h>
#include <WorkflowController.h>

//for time measure
#include <time.h>
#include <stdio.h>

//for async function
#include <thread>
#include <future>

using namespace std;
ComputationalModel::ComputationalModel(): processor(0), _id((int)&*this) {
    WorkflowController::registerModel(_id);
}

// ComputationalModel::~ComputationalModel(){}

void ComputationalModel::execute(int mode)
{
    clock_t start, stop;

    if(processor == 0){
        start = clock();
        CPUImplementation();
        stop = clock();
        async(std::launch::async, [&]() { WorkflowController::updateCPUTime(this, start, stop); });
    }
    else {
        start = clock();
        GPUImplementation();
        stop = clock();
        async(std::launch::async, [&]() { WorkflowController::updateGPUTime(this, start, stop); });
    }

    //ComputationalModel::updateResults(start, stop, freq, tCPU, tGPU);
}

void ComputationalModel::setProcessor(int p) {
    processor = p;
}
