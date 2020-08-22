#include <iostream>
#include <Constants.h>
#include <WorkflowController.h>

#include <ComputationalModel.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <functional>
#include <map>
#include <time.h> 

using namespace std;

struct Count { int CPU, GPU; };
struct Clock { clock_t CPU, GPU; };

static map<int, Clock> clocks;
static map<int, Count> counts;

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
    clocks[objectId] = {0, 0};
    counts[objectId] = {0, 0};
}

void WorkflowController::updateCPUTime(ComputationalModel* cModel, clock_t start, clock_t stop) {
    clock_t delay = stop - start;
    cout << delay << " clocks" << endl;
    int _id = int(&cModel);
    counts[_id].CPU++;
    clocks[_id].CPU += delay;

    cout << clocks[_id].CPU <<","<< clocks[_id].GPU << endl << endl;

    if (counts[_id].CPU + counts[_id].GPU > 9 && clocks[_id].CPU > clocks[_id].GPU) {
        counts[_id].CPU = 0;
        counts[_id].GPU = 0;
        cModel->setProcessor(1);
    }
}

void WorkflowController::updateGPUTime(ComputationalModel * cModel, clock_t start, clock_t stop) {
    clock_t delay = stop - start;
    cout << delay << " clocks" << endl;
    int _id = int(&cModel);
    counts[_id].GPU++;
    clocks[_id].GPU += delay;

    cout << clocks[_id].CPU << "," << clocks[_id].GPU << endl << endl;

    if (counts[_id].CPU + counts[_id].GPU > 9 && clocks[_id].GPU > clocks[_id].CPU) {
        counts[_id].CPU = 0;
        counts[_id].GPU = 0;
        cModel->setProcessor(1);
    }
}
