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
    ;

    cout << cModel->clocks.CPU / REVISE_COUNT <<","<< cModel->clocks.GPU / REVISE_COUNT << endl << endl;

    mtx.lock();
    cModel->clocks.CPU += delay;
    cModel->counts.CPU++;
    mtx.unlock();

    if (cModel->counts.CPU - cModel->counts.GPU > REVISE_COUNT && cModel->clocks.CPU / REVISE_COUNT > cModel->clocks.GPU/ REVISE_COUNT) {
        mtx.lock();
        // cModel->counts = {0, 0};
        // cModel->clocks = {0, 0};
        cModel->setProcessor(1);
        mtx.unlock();
    }
}

void WorkflowController::updateGPUTime(ComputationalModel * cModel, clock_t start, clock_t stop) {
    clock_t delay = stop - start;
    cout << delay << " clocks" << endl;
    ;

    cout << cModel->clocks.CPU << "," << cModel->clocks.GPU << endl << endl;

    mtx.lock();
    cModel->clocks.GPU += delay;
    cModel->counts.GPU++;
    mtx.unlock();

    if (cModel->counts.GPU - cModel->counts.CPU > REVISE_COUNT && cModel->clocks.GPU/ REVISE_COUNT > cModel->clocks.CPU/ REVISE_COUNT) {
        mtx.lock();
        // cModel->counts = { 0, 0 };
        // cModel->clocks = { 0, 0 };
        cModel->setProcessor(0);
        mtx.unlock();
    }
}
