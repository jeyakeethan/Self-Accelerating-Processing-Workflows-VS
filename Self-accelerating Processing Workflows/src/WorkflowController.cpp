#include "WorkflowController.h"

#include <ComputationalModel.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <functional>
#include <list> 

#include <iostream>
using namespace std;

static long tCPU=0, tGPU=0, countCPU=0, countGPU=0, countT = 0;

WorkflowController::WorkflowController() {}
WorkflowController::~WorkflowController() {}

void WorkflowController::launchBenchmarkUpdater()
{
  std::thread([=]()
  {
    while (true)
    {
      reviseAllBenchmarks();
      auto x = std::chrono::steady_clock::now() + std::chrono::milliseconds(REVISE_PERIOD);
      std::this_thread::sleep_until(x);
    }
  }).detach();
}

void WorkflowController::reviseAllBenchmarks()
{
  std::cout << "To do" << std::endl;
}

void WorkflowController::updateArrayAdditionBenchmark()
{
  std::cout << "To do" << std::endl;
}

void WorkflowController::registerModel(int objectId) {
    
}

void WorkflowController::updateCPUTime(ComputationalModel * cModel, clock_t start, clock_t stop) {
    clock_t delay = stop - start;
    int time = (int)(delay * 100000);
    cout << "CPU Time: " << time << " ns" << endl;
    countCPU++;
    countT++;
    tCPU += (int)time;

    cout << tCPU <<","<<tGPU << endl << endl;

    if (countT > 9 && tCPU > tGPU) {
        countT = 0;
        cModel->setProcessor(1);
    }
}

void WorkflowController::updateGPUTime(ComputationalModel * cModel, clock_t start, clock_t stop) {
    clock_t delay = stop - start;
    int time = (int)(delay *100000);
    cout << "GPU Time: " << time << " ns" << endl;
    countGPU++;
    countT++;
    tGPU += (int)time;

    cout << tCPU <<","<<tGPU << endl << endl;

    if (countT > 9 && tCPU<tGPU) {
        countT = 0;
        cModel->setProcessor(1);
    }
}
