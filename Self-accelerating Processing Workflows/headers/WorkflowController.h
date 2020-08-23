#ifndef WORKFLOWCONTROLLER_H
#define WORKFLOWCONTROLLER_H
#include <ComputationalModel.h>

class WorkflowController
{
    public:
        WorkflowController();
        virtual ~WorkflowController();
        void launchBenchmarkUpdater();
        void reviseAllBenchmarks();
        void updateArrayAdditionBenchmark();
        static void registerModel(ComputationalModel * cModel);
        static void changeProcessor(ComputationalModel * cModel);

    protected:

    private:
};

#endif // WORKFLOWCONTROLLER_H
