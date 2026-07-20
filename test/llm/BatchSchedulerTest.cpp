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
        MNNTEST_ASSERT(first->pipelineId == 0);
        MNNTEST_ASSERT(first->segmentIndex == 0);
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
        MNNTEST_ASSERT(second->pipelineId == 0);
        MNNTEST_ASSERT(second->segmentIndex == 1);
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
        MNNTEST_ASSERT(first->pipelineId == 0);
        MNNTEST_ASSERT(first->segmentIndex == 0);
        MNNTEST_ASSERT(first->culLen == 2);
        MNNTEST_ASSERT(first->inputs.size() == 1);
        MNNTEST_ASSERT(first->inputs[0] == std::vector<int>({1, 2}));
        MNNTEST_ASSERT(first->calLen[0] == 2);
        MNNTEST_ASSERT(first->pos[0] == 0);
        MNNTEST_ASSERT(first->reqId[0] == reqIds[0]);
        MNNTEST_ASSERT(BatchScheduler::judgeState(first->state[0], BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqIds[0]), BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqIds[1]), BatchScheduler::RequestState::PREFILL));

        auto second = scheduler.schedule(4, 2);
        MNNTEST_ASSERT(second != nullptr);
        MNNTEST_ASSERT(second->pipelineId == 1);
        MNNTEST_ASSERT(second->segmentIndex == 0);
        MNNTEST_ASSERT(second->culLen == 2);
        MNNTEST_ASSERT(second->inputs.size() == 1);
        MNNTEST_ASSERT(second->inputs[0] == std::vector<int>({11, 12}));
        MNNTEST_ASSERT(second->calLen[0] == 2);
        MNNTEST_ASSERT(second->pos[0] == 0);
        MNNTEST_ASSERT(second->reqId[0] == reqIds[1]);
        MNNTEST_ASSERT(second->state.size() == 1);
        MNNTEST_ASSERT(BatchScheduler::judgeState(second->state[0], BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqIds[0]), BatchScheduler::RequestState::PREFILL));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqIds[1]), BatchScheduler::RequestState::PREFILL));

        auto third = scheduler.schedule(4, 2);
        MNNTEST_ASSERT(third != nullptr);
        MNNTEST_ASSERT(third->pipelineId == 0);
        MNNTEST_ASSERT(third->segmentIndex == 1);
        MNNTEST_ASSERT(third->culLen == 2);
        MNNTEST_ASSERT(third->inputs.size() == 1);
        MNNTEST_ASSERT(third->inputs[0] == std::vector<int>({3, 4}));
        MNNTEST_ASSERT(third->pos[0] == 2);
        MNNTEST_ASSERT(third->reqId[0] == reqIds[0]);

        auto fourth = scheduler.schedule(4, 2);
        MNNTEST_ASSERT(fourth != nullptr);
        MNNTEST_ASSERT(fourth->pipelineId == 1);
        MNNTEST_ASSERT(fourth->segmentIndex == 1);
        MNNTEST_ASSERT(fourth->culLen == 2);
        MNNTEST_ASSERT(fourth->inputs.size() == 1);
        MNNTEST_ASSERT(fourth->inputs[0] == std::vector<int>({13, 14}));
        MNNTEST_ASSERT(fourth->pos[0] == 2);
        MNNTEST_ASSERT(fourth->reqId[0] == reqIds[1]);
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqIds[0]), BatchScheduler::RequestState::DECODE));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqIds[1]), BatchScheduler::RequestState::DECODE));
        return true;
    }
};

class BatchSchedulerDualPipelineFourRequestPartitionTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        BatchScheduler scheduler;
        scheduler.setDualPipelineMode(true, 2);
        std::vector<int> reqIds = scheduler.addRequest({{1, 2}, {11, 12}, {21, 22}, {31, 32}});

        auto first = scheduler.schedule(2, 4);
        MNNTEST_ASSERT(first != nullptr);
        MNNTEST_ASSERT(first->pipelineId == 0);
        MNNTEST_ASSERT(first->segmentIndex == 0);
        MNNTEST_ASSERT(first->culLen == 2);
        MNNTEST_ASSERT(first->inputs.size() == 2);
        MNNTEST_ASSERT(first->inputs[0] == std::vector<int>({1}));
        MNNTEST_ASSERT(first->inputs[1] == std::vector<int>({11}));
        MNNTEST_ASSERT(first->reqId[0] == reqIds[0]);
        MNNTEST_ASSERT(first->reqId[1] == reqIds[1]);

        auto second = scheduler.schedule(2, 4);
        MNNTEST_ASSERT(second != nullptr);
        MNNTEST_ASSERT(second->pipelineId == 1);
        MNNTEST_ASSERT(second->segmentIndex == 0);
        MNNTEST_ASSERT(second->culLen == 2);
        MNNTEST_ASSERT(second->inputs.size() == 2);
        MNNTEST_ASSERT(second->inputs[0] == std::vector<int>({21}));
        MNNTEST_ASSERT(second->inputs[1] == std::vector<int>({31}));
        MNNTEST_ASSERT(second->reqId[0] == reqIds[2]);
        MNNTEST_ASSERT(second->reqId[1] == reqIds[3]);

        auto third = scheduler.schedule(2, 4);
        MNNTEST_ASSERT(third != nullptr);
        MNNTEST_ASSERT(third->pipelineId == 0);
        MNNTEST_ASSERT(third->segmentIndex == 1);
        MNNTEST_ASSERT(third->inputs.size() == 2);
        MNNTEST_ASSERT(third->inputs[0] == std::vector<int>({2}));
        MNNTEST_ASSERT(third->inputs[1] == std::vector<int>({12}));

        auto fourth = scheduler.schedule(2, 4);
        MNNTEST_ASSERT(fourth != nullptr);
        MNNTEST_ASSERT(fourth->pipelineId == 1);
        MNNTEST_ASSERT(fourth->segmentIndex == 1);
        MNNTEST_ASSERT(fourth->inputs.size() == 2);
        MNNTEST_ASSERT(fourth->inputs[0] == std::vector<int>({22}));
        MNNTEST_ASSERT(fourth->inputs[1] == std::vector<int>({32}));
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

class BatchSchedulerDualPipelineScheduleWaveTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        BatchScheduler scheduler;
        scheduler.setDualPipelineMode(true, 2);
        std::vector<int> reqIds = scheduler.addRequest({{1, 2, 3, 4}, {11, 12, 13, 14}});

        std::vector<std::shared_ptr<BatchScheduler::Chunk>> firstWave = scheduler.scheduleWave(4, 2);
        MNNTEST_ASSERT(firstWave.size() == 2);
        MNNTEST_ASSERT(firstWave[0]->segmentIndex == 0);
        MNNTEST_ASSERT(firstWave[1]->segmentIndex == 0);
        MNNTEST_ASSERT(firstWave[0]->pipelineId == 0);
        MNNTEST_ASSERT(firstWave[1]->pipelineId == 1);
        MNNTEST_ASSERT(firstWave[0]->reqId[0] == reqIds[0]);
        MNNTEST_ASSERT(firstWave[1]->reqId[0] == reqIds[1]);
        MNNTEST_ASSERT(firstWave[0]->inputs[0] == std::vector<int>({1, 2}));
        MNNTEST_ASSERT(firstWave[1]->inputs[0] == std::vector<int>({11, 12}));

        std::vector<std::shared_ptr<BatchScheduler::Chunk>> secondWave = scheduler.scheduleWave(4, 2);
        MNNTEST_ASSERT(secondWave.size() == 2);
        MNNTEST_ASSERT(secondWave[0]->segmentIndex == 1);
        MNNTEST_ASSERT(secondWave[1]->segmentIndex == 1);
        MNNTEST_ASSERT(secondWave[0]->pipelineId == 0);
        MNNTEST_ASSERT(secondWave[1]->pipelineId == 1);
        MNNTEST_ASSERT(secondWave[0]->reqId[0] == reqIds[0]);
        MNNTEST_ASSERT(secondWave[1]->reqId[0] == reqIds[1]);
        MNNTEST_ASSERT(secondWave[0]->inputs[0] == std::vector<int>({3, 4}));
        MNNTEST_ASSERT(secondWave[1]->inputs[0] == std::vector<int>({13, 14}));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqIds[0]), BatchScheduler::RequestState::DECODE));
        MNNTEST_ASSERT(BatchScheduler::judgeState(scheduler.state(reqIds[1]), BatchScheduler::RequestState::DECODE));
        return true;
    }
};

class BatchSchedulerDualPipelineSingleRequestScheduleWaveTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        BatchScheduler scheduler;
        scheduler.setDualPipelineMode(true, 2);
        const int reqId = scheduler.addRequest({1, 2, 3, 4});

        std::vector<std::shared_ptr<BatchScheduler::Chunk>> firstWave = scheduler.scheduleWave(4, 2);
        MNNTEST_ASSERT(firstWave.size() == 1);
        MNNTEST_ASSERT(firstWave[0]->pipelineId == 0);
        MNNTEST_ASSERT(firstWave[0]->segmentIndex == 0);
        MNNTEST_ASSERT(firstWave[0]->reqId[0] == reqId);
        MNNTEST_ASSERT(firstWave[0]->inputs[0] == std::vector<int>({1, 2}));

        std::vector<std::shared_ptr<BatchScheduler::Chunk>> secondWave = scheduler.scheduleWave(4, 2);
        MNNTEST_ASSERT(secondWave.size() == 1);
        MNNTEST_ASSERT(secondWave[0]->pipelineId == 0);
        MNNTEST_ASSERT(secondWave[0]->segmentIndex == 1);
        MNNTEST_ASSERT(secondWave[0]->reqId[0] == reqId);
        MNNTEST_ASSERT(secondWave[0]->inputs[0] == std::vector<int>({3, 4}));
        return true;
    }
};

class BatchSchedulerDualPipelinePipelinePersistenceTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        BatchScheduler scheduler;
        scheduler.setDualPipelineMode(true, 2);
        std::vector<int> reqIds = scheduler.addRequest(std::vector<std::vector<int>>{{7}, {17}});

        std::vector<std::shared_ptr<BatchScheduler::Chunk>> firstWave = scheduler.scheduleWave(1, 2);
        MNNTEST_ASSERT(firstWave.size() == 2);
        MNNTEST_ASSERT(firstWave[0]->pipelineId == 0);
        MNNTEST_ASSERT(firstWave[1]->pipelineId == 1);
        MNNTEST_ASSERT(firstWave[0]->reqId[0] == reqIds[0]);
        MNNTEST_ASSERT(firstWave[1]->reqId[0] == reqIds[1]);

        std::vector<std::shared_ptr<BatchScheduler::Chunk>> secondWave = scheduler.scheduleWave(1, 2);
        MNNTEST_ASSERT(secondWave.size() == 2);
        MNNTEST_ASSERT(secondWave[0]->pipelineId == 0);
        MNNTEST_ASSERT(secondWave[1]->pipelineId == 1);
        MNNTEST_ASSERT(secondWave[0]->reqId[0] == reqIds[0]);
        MNNTEST_ASSERT(secondWave[1]->reqId[0] == reqIds[1]);
        MNNTEST_ASSERT(BatchScheduler::judgeState(secondWave[0]->state[0], BatchScheduler::RequestState::DECODE));
        MNNTEST_ASSERT(BatchScheduler::judgeState(secondWave[1]->state[0], BatchScheduler::RequestState::DECODE));
        return true;
    }
};

class BatchSchedulerDualPipelineSplitCountClampTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        BatchScheduler scheduler;
        scheduler.setDualPipelineMode(true, 4);
        std::vector<int> reqIds = scheduler.addRequest(std::vector<std::vector<int>>{{1, 2, 3, 4}, {11, 12, 13, 14}, {21, 22, 23, 24}, {31, 32, 33, 34}});

        std::vector<std::shared_ptr<BatchScheduler::Chunk>> wave = scheduler.scheduleWave(4, 4);
        MNNTEST_ASSERT(wave.size() == 2);
        MNNTEST_ASSERT(wave[0]->pipelineId == 0);
        MNNTEST_ASSERT(wave[1]->pipelineId == 1);
        MNNTEST_ASSERT(wave[0]->segmentIndex == 0);
        MNNTEST_ASSERT(wave[1]->segmentIndex == 0);
        MNNTEST_ASSERT(wave[0]->reqId[0] == reqIds[0]);
        MNNTEST_ASSERT(wave[1]->reqId[0] == reqIds[2]);
        return true;
    }
};

class BatchSchedulerDualPipelineThreeRequestWaveTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        BatchScheduler scheduler;
        scheduler.setDualPipelineMode(true, 2);
        std::vector<int> reqIds = scheduler.addRequest(
            std::vector<std::vector<int>>{{1}, {11}, {21}});

        std::vector<std::shared_ptr<BatchScheduler::Chunk>> wave = scheduler.scheduleWave(1, 3);
        MNNTEST_ASSERT(wave.size() == 2);
        MNNTEST_ASSERT(wave[0]->pipelineId == 0);
        MNNTEST_ASSERT(wave[1]->pipelineId == 1);
        MNNTEST_ASSERT(wave[0]->reqId.size() == 2);
        MNNTEST_ASSERT(wave[1]->reqId.size() == 1);
        MNNTEST_ASSERT(wave[0]->reqId[0] == reqIds[0]);
        MNNTEST_ASSERT(wave[0]->reqId[1] == reqIds[1]);
        MNNTEST_ASSERT(wave[1]->reqId[0] == reqIds[2]);
        return true;
    }
};

MNNTestSuiteRegister(BatchSchedulerDualPipelineSplitTest, "llm/batch_scheduler_dual_pipeline_split");
MNNTestSuiteRegister(BatchSchedulerSingleTokenPrefillStateTest, "llm/batch_scheduler_single_token_prefill_state");
MNNTestSuiteRegister(BatchSchedulerDualPipelineMultiRequestSplitTest, "llm/batch_scheduler_dual_pipeline_multi_request_split");
MNNTestSuiteRegister(BatchSchedulerDualPipelineFourRequestPartitionTest, "llm/batch_scheduler_dual_pipeline_four_request_partition");
MNNTestSuiteRegister(BatchSchedulerDualPipelineSingleTokenPrefillTest, "llm/batch_scheduler_dual_pipeline_single_token_prefill");
MNNTestSuiteRegister(BatchSchedulerDualPipelineScheduleWaveTest, "llm/batch_scheduler_dual_pipeline_schedule_wave");
MNNTestSuiteRegister(BatchSchedulerDualPipelineSingleRequestScheduleWaveTest, "llm/batch_scheduler_dual_pipeline_single_request_schedule_wave");
MNNTestSuiteRegister(BatchSchedulerDualPipelinePipelinePersistenceTest, "llm/batch_scheduler_dual_pipeline_pipeline_persistence");
MNNTestSuiteRegister(BatchSchedulerDualPipelineSplitCountClampTest, "llm/batch_scheduler_dual_pipeline_split_count_clamp");
MNNTestSuiteRegister(BatchSchedulerDualPipelineThreeRequestWaveTest, "llm/batch_scheduler_dual_pipeline_three_request_wave");
