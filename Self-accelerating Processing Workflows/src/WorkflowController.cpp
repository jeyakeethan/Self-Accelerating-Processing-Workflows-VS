#include "WorkflowController.h"

#include <ComputationalModel.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <functional>
#include <list> 

#include <iostream>
using namespace std;

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

void WorkflowController::updateElapsedTime(int objectId, clock_t start, clock_t stop, int processor) {
    // To a async function
    clock_t delay = stop - start;
    float time = (float)delay / CLOCKS_PER_SEC * 1000000;
    if (processor == 0) {
        cout << "CPU Time: " << time << " ns" << endl;
        clog << "CPU Time: " << time << " ns" << endl;
    }
    else {
        cout << "GPU Time: " << time << " ns" << endl;
        clog << "GPU Time: " << time << " ns" << endl;
    }

    // cout << tCPU <<","<<tGPU << endl << endl;
}
