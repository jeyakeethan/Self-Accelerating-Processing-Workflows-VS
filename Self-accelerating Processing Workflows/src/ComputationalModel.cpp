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
    cout << _id;
}

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
    async(std::launch::async, [&]() { WorkflowController::updateElapsedTime(_id, start, stop, processor); });
    //ComputationalModel::updateResults(start, stop, freq, tCPU, tGPU);
}

