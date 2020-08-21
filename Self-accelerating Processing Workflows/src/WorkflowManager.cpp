#include "WorkflowManager.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <functional>

int revisePeriod = REVISE_PERIOD;

void WorkflowManager::launchBenchmarkUpdater()
{
  std::thread([=]()
  {
    while (true)
    {
      reviseAllBenchmarks();
      auto x = std::chrono::steady_clock::now() + std::chrono::milliseconds(revisePeriod);
      std::this_thread::sleep_until(x);
    }
  }).detach();
}

void WorkflowManager::reviseAllBenchmarks()
{
  std::cout << "To do" << std::endl;
}

void WorkflowManager::updateArrayAdditionBenchmark()
{
  std::cout << "To do" << std::endl;
}

