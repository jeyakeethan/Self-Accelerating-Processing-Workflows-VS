#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <ComputationalModel.h>
#include <Logger.h>
//for time measure
#include <windows.h>
#include <time.h>
#include <stdio.h>

//for async function
#include <thread>
#include <future>

using namespace std;

bool ComputationalModel::operationalMode = false;
ComputationalModel::ComputationalModel(int CPUCores_):CPUCores(CPUCores_) {
    obj_id = obj_id_counter();
    resetFlow();
    revisor = thread(&ComputationalModel::resetOverPeriodIfBurst, this);
    revisor.detach();
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
    // id_ = int(&*this);
}

ComputationalModel::~ComputationalModel(){
    revisor.~thread();
    Logger::close();
    //TO DO; log present values for using next boot
}

// Mannual mode execution
void ComputationalModel::execute(int mode) {
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

void ComputationalModel::executeAndLogging(int mode)
{
    // first class name, object id, data and finally the execution time
    stringstream s;
    s << typeid(*this).name() << " ";
    s << obj_id << " ";
    LARGE_INTEGER start_cover;
    LARGE_INTEGER stop_cover;
    QueryPerformanceCounter(&start_cover);

    switch (mode) {
        case 1:
            // cout << "Hello CPU" << endl;
            CPUImplementation();
            QueryPerformanceCounter(&stop);
            duration = stop.QuadPart - start.QuadPart;
            s << "CPU ";
            break;
        case 2:
            // cout << "Hello GPU" << endl;
            QueryPerformanceCounter(&start);
            GPUImplementation();
            QueryPerformanceCounter(&stop);
            duration = stop.QuadPart - start.QuadPart;
            s << "GPU ";
            break;
    }

    QueryPerformanceCounter(&stop_cover);
    duration = stop_cover.QuadPart - start_cover.QuadPart;

    int* attr = getAttributes();
    for (int i = 0; i < 3; i++) {
        s << attr[i] << " ";
    }
    s << duration;
    logExTime(s.str());
}

// Auto mode execution
void ComputationalModel::execute() {
    switch (processor) {
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
                }
                else {
                    processor = 1;
                    reviseCount = REVISE_COUNT_MIN;
                    alignedCount = 0;
                }
                lastRevisedClock.QuadPart = stop.QuadPart;
            }
            else {
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
                }
                else {
                    processor = 1;
                    reviseCount += REVISE_COUNT_STEP * ++alignedCount;
                }
                lastRevisedClock.QuadPart = stop.QuadPart;
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

void ComputationalModel::executeAndLogging()
{
    s.clear();
    s << typeid(*this).name() << " ";
    s << obj_id << " ";
    LARGE_INTEGER start_cover;
    LARGE_INTEGER stop_cover;
    QueryPerformanceCounter(&start_cover);
    switch (processor) {
    case 1:
        // cout << "Hello CPU" << endl;
        s << "CPU ";
        CPUImplementation();
        break;
    case 2:
        // cout << "Hello GPU" << endl;
        s << "GPU ";
        GPUImplementation();
        break;
    case -1:
        // cout << "Hello CPU" << endl;
        s << "CPU ";
        QueryPerformanceCounter(&start);
        CPUImplementation();
        QueryPerformanceCounter(&stop);
        clocks.CPU += stop.QuadPart - start.QuadPart;;
        // cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
        if (++countS > SAMPLE_COUNT) {
            if (--sampleMode == 0) {
                if (clocks.CPU > clocks.GPU) {
                    processor = 2;
                    reviseCount += REVISE_COUNT_STEP * ++alignedCount;
                }
                else {
                    processor = 1;
                    reviseCount = REVISE_COUNT_MIN;
                    alignedCount = 0;
                }
                lastRevisedClock.QuadPart = stop.QuadPart;
                //                    cout << "REVISE_COUNT: " << reviseCount << endl;
//                    cout << alignedCount << "," << clocks.CPU << "," << clocks.GPU << endl << endl;
            }
            else {
                processor = -2; // processor = (processor - 1) % 3;
                countS = 1;
            }
        }
        return;
    case -2:
        s << "GPU ";
        // cout << "Hello GPU" << endl;
        QueryPerformanceCounter(&start);
        GPUImplementation();
        QueryPerformanceCounter(&stop);
        clocks.GPU += stop.QuadPart - start.QuadPart;;
        // cout << stop.QuadPart - start.QuadPart << " clocks" << endl;
        if (++countS > SAMPLE_COUNT) {
            if (--sampleMode == 0) {
                if (clocks.CPU > clocks.GPU) {
                    processor = 2;
                    reviseCount = REVISE_COUNT_MIN;
                    alignedCount = 0;
                }
                else {
                    processor = 1;
                    reviseCount += REVISE_COUNT_STEP * ++alignedCount;
                }
                lastRevisedClock.QuadPart = stop.QuadPart;
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

    QueryPerformanceCounter(&stop_cover);
    duration = stop_cover.QuadPart - start_cover.QuadPart;

    int* attr = getAttributes();
    for (int i = 0; i < 3; i++) {
        s << attr[i] << " ";
    }
    s << duration << endl;
    logExTime(s.str());

    if (countL > reviseCount) {
        sampleMode = 2;
        countS = 1;
        countL = 1;
        processor = -processor;
        clocks = { 0, 0, 0.0, 0.0 };
    }
}

void ComputationalModel::setProcessor(int p) {
    processor = p;
}


void ComputationalModel::prepareLogging() {

}

/* static method run by a thread to reset the flow if the input stream is burst and sparsed */
void ComputationalModel::resetOverPeriodIfBurst(ComputationalModel* cm)
{
    LARGE_INTEGER now, frequency, reviseBoundary;
    QueryPerformanceFrequency(&frequency);
    reviseBoundary.QuadPart = frequency.QuadPart * cm->revisePeriod;
    while (true) {
        this_thread::sleep_for(chrono::seconds(cm->revisePeriod));
        QueryPerformanceCounter(&now);
        if (now.QuadPart - cm->lastRevisedClock.QuadPart > reviseBoundary.QuadPart) {
            cm->resetFlow();    // reset the flow
        }
    }
}

void ComputationalModel::setOperationalMode(bool om) {
    operationalMode = om;
}

void ComputationalModel::logExTime(string str) {
    if (!Logger::isOpen()) {
        Logger::open(LOG_FILE_NAME);
    }
    Logger::write(str);
}

