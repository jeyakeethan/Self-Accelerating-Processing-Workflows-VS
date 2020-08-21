#ifndef WORKFLOWCONTROLLER_H
#define WORKFLOWCONTROLLER_H

#include <time.h>
#include <ComputationalModel.h>

#define REVISE_PERIOD 10000          //revise period of the benchmarks in seconds

class WorkflowController
{
    public:
        WorkflowController();
        virtual ~WorkflowController();
        void launchBenchmarkUpdater();
        void reviseAllBenchmarks();
        void updateArrayAdditionBenchmark();
        static void registerModel(int objectId);
        static void updateCPUTime(ComputationalModel * cModel, clock_t start, clock_t stop);
        static void updateGPUTime(ComputationalModel * cModel, clock_t start, clock_t stop);

    protected:

    private:
};

#endif // WORKFLOWCONTROLLER_H
