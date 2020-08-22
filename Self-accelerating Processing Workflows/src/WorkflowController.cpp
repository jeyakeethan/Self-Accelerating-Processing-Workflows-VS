#include <iostream>
#include <Constants.h>
#include <WorkflowController.h>

#include <ComputationalModel.h>
#include <iostream>
#include <chrono>
#include <time.h>
#include <thread>
#include <functional>
#include <list>
#include <mutex>

using namespace std;

list<ComputationalModel*> registeredModels;

WorkflowController::WorkflowController() {}
WorkflowController::~WorkflowController() {}

std::mutex mtx;

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

void WorkflowController::registerModel(ComputationalModel * cModel) {
    registeredModels.push_back(cModel);
}

void WorkflowController::updateCPUTime(ComputationalModel* cModel, clock_t start, clock_t stop) {
    clock_t delay = stop - start;
    cout << delay << " clocks" << endl;

    mtx.lock();
    cModel->clocks.CPU += delay;
    cModel->counts++;

    if (cModel->counts > REVISE_COUNT) {
        cModel->clocks.CPUmean = (float)cModel->clocks.CPU / REVISE_COUNT;
        cout << cModel->clocks.CPUmean << "," << cModel->clocks.GPUmean << endl << endl;
        cModel->counts = 0;
        if (cModel->clocks.CPUmean > cModel->clocks.GPUmean) {
            cModel->setProcessor(1);
        }
    }
    mtx.unlock();
}

void WorkflowController::updateGPUTime(ComputationalModel * cModel, clock_t start, clock_t stop) {
    clock_t delay = stop - start;
    cout << delay << " clocks" << endl;

    mtx.lock();
    cModel->clocks.GPU += delay;
    cModel->counts++;

    if (cModel->counts > REVISE_COUNT) {
        cModel->clocks.GPUmean = (float)cModel->clocks.GPU / REVISE_COUNT;
        cout << cModel->clocks.CPUmean << "," << cModel->clocks.GPUmean << endl << endl;
        cModel->counts = 0;
        if (cModel->clocks.GPUmean > cModel->clocks.CPUmean) {
            cModel->setProcessor(0);
        }
    }
    mtx.unlock();
}
