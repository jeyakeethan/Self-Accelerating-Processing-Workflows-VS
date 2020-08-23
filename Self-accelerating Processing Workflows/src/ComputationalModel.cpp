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

ComputationalModel::ComputationalModel(): processor(-1){
    WorkflowController::registerModel(this);
    clocks = { 0, 0, 0.0, 0.0};
    countS = 1;
    countL = 1;
    revisePeriod = REVISE_PERIOD;
    sampleMode = 2;
}

// ComputationalModel::~ComputationalModel(){}

void ComputationalModel::execute(int mode)
{
    LARGE_INTEGER start, stop;
    switch(processor){
        case 1:
            CPUImplementation();
            countL++;
            break;
        case 2:
            GPUImplementation();
            countL++;
            break;
        case -1:
            QueryPerformanceCounter(&start);
            CPUImplementation();
            QueryPerformanceCounter(&stop);
            clocks.CPU += stop.QuadPart - start.QuadPart;
            // cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
            if (++countS > SAMPLE_COUNT) {
                if (--sampleMode == 0) {
                    if (clocks.CPU > clocks.GPU)
                        processor = 2;
                    else
                        processor = 1;
                    cout << clocks.CPU << "," << clocks.GPU << endl << endl;
                } else {
                    processor = -2; // processor = (processor - 1) % 3;
                    countS = 1;
                }
            }
            //    async(std::launch::async, [&]() { WorkflowController::updateCPUTime(this); });
            break;
        case -2:
            QueryPerformanceCounter(&start);
            GPUImplementation();
            QueryPerformanceCounter(&stop);
            clocks.GPU += stop.QuadPart - start.QuadPart;
            // cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
            if (++countS > SAMPLE_COUNT) {
                if (--sampleMode == 0) {
                    if (clocks.CPU > clocks.GPU)
                        processor = 2;
                    else
                        processor = 1;
                    cout << clocks.CPU << "," << clocks.GPU << endl << endl;
                }
                else {
                    processor = -1; // processor = (processor - 1) % 3;
                    countS = 1;
                }
            }
            break;
        default:
            sampleMode = 2;
            processor = -1;
    }
    if (countL > REVISE_COUNT) {
            sampleMode = 2;
            countS = 1;
            countL = 1;
            processor = -processor;
            clocks = { 0, 0, 0.0, 0.0 };
            cout << endl;
    }

    //ComputationalModel::updateResults(start, stop, freq, tCPU, tGPU);
}

void ComputationalModel::setProcessor(int p) {
    processor = p;
}
