#ifndef WORKFLOWMANAGER_H
#define WORKFLOWMANAGER_H

#define REVISE_PERIOD 10000          //revise period of the benchmarks in seconds

class WorkflowManager
{
    public:
        WorkflowManager();
        virtual ~WorkflowManager();
        void launchBenchmarkUpdater();
        void reviseAllBenchmarks();
        void updateArrayAdditionBenchmark();

    protected:

    private:
};

#endif // WORKFLOWMANAGER_H
