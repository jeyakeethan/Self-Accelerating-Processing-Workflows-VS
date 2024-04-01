#include <iostream>
#include <PdhCPUCounter.h>

using namespace std;

int main()
{
    processor_time = std::make_unique<PdhCPUCounter>("\\Process(Monitor)\\% Processor Time");
    // Here we're reported about 0.35 +/- 0.05
    double cpu_utilization = processor_time->getCPUUtilization();
}
