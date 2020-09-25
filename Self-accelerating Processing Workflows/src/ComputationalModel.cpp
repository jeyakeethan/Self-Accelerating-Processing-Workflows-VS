#include <iostream>
#include <ComputationalModel.h>
#include <Constants.h>

//for time measure
#include <windows.h>
#include <time.h>
#include <stdio.h>

//for async function
#include <thread>
#include <future>

using namespace std;

ComputationalModel::ComputationalModel(int CPUCores_):CPUCores(CPUCores_) {
    resetFlow();
    ComputationalModel* temp = this;
    ComputationalModel::resetOverPeriodIfBurst(temp);
}

inline void ComputationalModel::resetFlow() {
    clocks = { 0, 0, 0.0, 0.0 };
    countS = 1;
    countL = 1;
    alignedCount = -1;
    reviseCount = REVISE_COUNT_MIN;
    revisePeriod = REVISE_PERIOD;
    sampleMode = 2;
    processor = -1;
    lastProcessor = -1;
    id_ = int(&*this);

}

ComputationalModel::~ComputationalModel(){
    //TO DO; log present values for using next boot
}

// Mannual mode execution
void ComputationalModel::execute(int mode)
{
    switch (mode) {
    case 1:
        // cout << "Hello CPU" << endl;
        CPUImplementation();
        break;
    case 2:
        // cout << "Hello GPU" << endl;
        GPUImplementation();
        break;
    }
}

// Auto mode execution
void ComputationalModel::execute()
{
    switch(processor){
        case 1:
            //cout << "Hello CPU" << endl;
            CPUImplementation();
            countL++;
            break;
        case 2:
            //cout << "Hello GPU" << endl;
            GPUImplementation();
            countL++;
            break;
        case -1:
            // cout << "Hello CPU" << endl;
            QueryPerformanceCounter(&start);
            CPUImplementation();
            QueryPerformanceCounter(&stop);
            clocks.CPU += stop.QuadPart - start.QuadPart;
            // cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
            if (++countS > SAMPLE_COUNT) {
                if (--sampleMode == 0) {
                    if (clocks.CPU > clocks.GPU) {
                        processor = 2;
                        reviseCount += REVISE_COUNT_STEP * ++alignedCount;
                    } else {
                        processor = 1;
                        reviseCount = REVISE_COUNT_MIN;
                        alignedCount = 0;
                    }
//                    cout << "REVISE_COUNT: " << reviseCount << endl;
//                    cout << alignedCount << "," << clocks.CPU << "," << clocks.GPU << endl << endl;
                } else {
                    processor = -2; // processor = (processor - 1) % 3;
                    countS = 1;
                }
            }
            return;
        case -2:
            // cout << "Hello GPU" << endl;
            QueryPerformanceCounter(&start);
            GPUImplementation();
            QueryPerformanceCounter(&stop);
            clocks.GPU += stop.QuadPart - start.QuadPart;
            // cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
            if (++countS > SAMPLE_COUNT) {
                if (--sampleMode == 0) {
                    if (clocks.CPU > clocks.GPU) {
                        processor = 2;
                        reviseCount = REVISE_COUNT_MIN;
                        alignedCount = 0;
                    } else {
                        processor = 1;
                        reviseCount += REVISE_COUNT_STEP * ++alignedCount;
                    }
//                    cout << "REVISE_COUNT: " << reviseCount << endl;
//                    cout << alignedCount << "," << clocks.CPU << "," << clocks.GPU << endl << endl;
                }
                else {
                    processor = -1; // processor = (processor - 1) % 3;
                    countS = 1;
                }
            }
            return;
        default:
            sampleMode = 2;
            processor = -1;
    }
    if (countL > reviseCount) {
            sampleMode = 2;
            countS = 1;
            countL = 1;
            processor = -processor;
            clocks = { 0, 0, 0.0, 0.0 };
//            cout << endl;
    }
}

void ComputationalModel::setProcessor(int p) {
    processor = p;
}

inline void ComputationalModel::resetOverPeriodIfBurst(ComputationalModel* cm)
{
    thread revisor([cm]() {
        LARGE_INTEGER now, frequency, reviseBoundary;
        QueryPerformanceFrequency(&frequency);
        reviseBoundary.QuadPart = frequency.QuadPart * cm->revisePeriod;
        while (true) {
            this_thread::sleep_for(chrono::seconds(cm->revisePeriod));
            QueryPerformanceCounter(&now);
            if (now.QuadPart - cm->stop.QuadPart > reviseBoundary.QuadPart) {
                cm->resetFlow();    // reset the flow if the input stream is burst and sparsed
            }
        }
    });
}
