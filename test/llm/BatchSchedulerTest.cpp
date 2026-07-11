//
//  BatchSchedulerTest.cpp
//  MNN
//

#include <MNN/MNNDefine.h>
#include "../MNNTestSuite.h"
#include "../../transformers/llm/engine/include/llm/BatchScheduler.hpp"

using namespace MNN::Transformer;

class BatchSchedulerDualPipelineSplitTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        BatchScheduler scheduler;
        scheduler.setDualPipelineMode(true, 2);
        const int reqId = scheduler.addRequest({1, 2, 3, 4});

        auto first = scheduler.schedule(4, 1);
        MNNTEST_ASSERT(first != nullptr);
        MNNTEST_ASSERT(first->culLen == 2);
        MNNTEST_ASSERT(first->inputs.size() == 1);
        MNNTEST_ASSERT(first->inputs[0] == std::vector<int>({1, 2}));
        MNNTEST_ASSERT(first->calLen[0] == 2);
        MNNTEST_ASSERT(first->pos[0] == 0);
        MNNTEST_ASSERT(first->reqId[0] == reqId);
        MNNTEST_ASSERT(BatchScheduler::judgeState(first->state[0], BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqId), BatchScheduler::RequestState::PREFILL));

        auto second = scheduler.schedule(4, 1);
        MNNTEST_ASSERT(second != nullptr);
        MNNTEST_ASSERT(second->culLen == 2);
        MNNTEST_ASSERT(second->inputs.size() == 1);
        MNNTEST_ASSERT(second->inputs[0] == std::vector<int>({3, 4}));
        MNNTEST_ASSERT(second->calLen[0] == 2);
        MNNTEST_ASSERT(second->pos[0] == 2);
        MNNTEST_ASSERT(second->reqId[0] == reqId);
        MNNTEST_ASSERT(BatchScheduler::judgeState(second->state[0], BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqId), BatchScheduler::RequestState::DECODE));

        auto decode = scheduler.schedule(4, 1);
        MNNTEST_ASSERT(decode != nullptr);
        MNNTEST_ASSERT(decode->culLen == 1);
        MNNTEST_ASSERT(decode->inputs[0] == std::vector<int>({4}));
        MNNTEST_ASSERT(decode->calLen[0] == 1);
        MNNTEST_ASSERT(BatchScheduler::judgeState(decode->state[0], BatchScheduler::RequestState::DECODE));
        return true;
    }
};

class BatchSchedulerSingleTokenPrefillStateTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        BatchScheduler scheduler;
        const int reqId = scheduler.addRequest(std::vector<int>({7}));

        auto prefill = scheduler.schedule(1, 1);
        MNNTEST_ASSERT(prefill != nullptr);
        MNNTEST_ASSERT(prefill->calLen[0] == 1);
        MNNTEST_ASSERT(prefill->inputs[0] == std::vector<int>({7}));
        MNNTEST_ASSERT(prefill->reqId[0] == reqId);
        MNNTEST_ASSERT(BatchScheduler::judgeState(prefill->state[0], BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqId), BatchScheduler::RequestState::DECODE));
        return true;
    }
};

class BatchSchedulerDualPipelineMultiRequestSplitTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        BatchScheduler scheduler;
        scheduler.setDualPipelineMode(true, 2);
        std::vector<int> reqIds = scheduler.addRequest({{1, 2, 3, 4}, {11, 12, 13, 14}});

        auto first = scheduler.schedule(4, 2);
        MNNTEST_ASSERT(first != nullptr);
        MNNTEST_ASSERT(first->culLen == 4);
        MNNTEST_ASSERT(first->inputs.size() == 2);
        MNNTEST_ASSERT(first->inputs[0] == std::vector<int>({1, 2}));
        MNNTEST_ASSERT(first->inputs[1] == std::vector<int>({11, 12}));
        MNNTEST_ASSERT(first->calLen[0] == 2);
        MNNTEST_ASSERT(first->calLen[1] == 2);
        MNNTEST_ASSERT(first->pos[0] == 0);
        MNNTEST_ASSERT(first->pos[1] == 0);
        MNNTEST_ASSERT(first->reqId[0] == reqIds[0]);
        MNNTEST_ASSERT(first->reqId[1] == reqIds[1]);
        MNNTEST_ASSERT(BatchScheduler::judgeState(first->state[0], BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(first->state[1], BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqIds[0]), BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqIds[1]), BatchScheduler::RequestState::PREFILL));

        auto second = scheduler.schedule(4, 2);
        MNNTEST_ASSERT(second != nullptr);
        MNNTEST_ASSERT(second->culLen == 4);
        MNNTEST_ASSERT(second->inputs.size() == 2);
        MNNTEST_ASSERT(second->inputs[0] == std::vector<int>({3, 4}));
        MNNTEST_ASSERT(second->inputs[1] == std::vector<int>({13, 14}));
        MNNTEST_ASSERT(second->calLen[0] == 2);
        MNNTEST_ASSERT(second->calLen[1] == 2);
        MNNTEST_ASSERT(second->pos[0] == 2);
        MNNTEST_ASSERT(second->pos[1] == 2);
        MNNTEST_ASSERT(second->reqId[0] == reqIds[0]);
        MNNTEST_ASSERT(second->reqId[1] == reqIds[1]);
        MNNTEST_ASSERT(second->state.size() == 2);
        MNNTEST_ASSERT(BatchScheduler::judgeState(second->state[0], BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(second->state[1], BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqIds[0]), BatchScheduler::RequestState::DECODE));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqIds[1]), BatchScheduler::RequestState::DECODE));
        return true;
    }
};

class BatchSchedulerDualPipelineSingleTokenPrefillTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        BatchScheduler scheduler;
        scheduler.setDualPipelineMode(true, 2);
        const int reqId = scheduler.addRequest(std::vector<int>({7}));

        auto prefill = scheduler.schedule(1, 1);
        MNNTEST_ASSERT(prefill != nullptr);
        MNNTEST_ASSERT(prefill->culLen == 1);
        MNNTEST_ASSERT(prefill->inputs.size() == 1);
        MNNTEST_ASSERT(prefill->inputs[0] == std::vector<int>({7}));
        MNNTEST_ASSERT(prefill->calLen[0] == 1);
        MNNTEST_ASSERT(prefill->reqId[0] == reqId);
        MNNTEST_ASSERT(prefill->state.size() == 1);
        MNNTEST_ASSERT(BatchScheduler::judgeState(prefill->state[0], BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqId), BatchScheduler::RequestState::DECODE));
        return true;
    }
};

MNNTestSuiteRegister(BatchSchedulerDualPipelineSplitTest, "llm/batch_scheduler_dual_pipeline_split");
MNNTestSuiteRegister(BatchSchedulerSingleTokenPrefillStateTest, "llm/batch_scheduler_single_token_prefill_state");
MNNTestSuiteRegister(BatchSchedulerDualPipelineMultiRequestSplitTest, "llm/batch_scheduler_dual_pipeline_multi_request_split");
MNNTestSuiteRegister(BatchSchedulerDualPipelineSingleTokenPrefillTest, "llm/batch_scheduler_dual_pipeline_single_token_prefill");
