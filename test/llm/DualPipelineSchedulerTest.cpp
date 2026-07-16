//
//  DualPipelineSchedulerTest.cpp
//  MNN
//

#include <MNN/MNNDefine.h>
#include "../MNNTestSuite.h"
#include "../../transformers/llm/engine/include/llm/DualPipelineScheduler.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <vector>

using namespace MNN::Transformer;

namespace {

struct GraphEvent {
    std::string type;
    std::string graphId;
};

struct GraphRecorder {
    std::string failedGraphId;

    DualPipelineScheduler::Callbacks callbacks() {
        DualPipelineScheduler::Callbacks cb;
        cb.onGraphLoad = [this](const DualPipelineScheduler::GraphRequest& request) {
            record("load", request.graphId);
            return request.graphId != failedGraphId;
        };
        cb.onGraphRelease = [this](const DualPipelineScheduler::GraphRequest& request) {
            record("release", request.graphId);
        };
        return cb;
    }

    bool waitForCount(size_t count, int timeoutMs = 1000) {
        std::unique_lock<std::mutex> lock(mutex);
        return condition.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this, count]() {
            return events.size() >= count;
        });
    }

    std::vector<GraphEvent> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        return events;
    }

private:
    void record(const std::string& type, const std::string& graphId) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            events.push_back({type, graphId});
        }
        condition.notify_all();
    }

    mutable std::mutex mutex;
    std::condition_variable condition;
    std::vector<GraphEvent> events;
};

DualPipelineScheduler::GraphRequest makeLoadRequest(const std::string& graphId) {
    DualPipelineScheduler::GraphRequest request;
    request.action = DualPipelineScheduler::GRAPH_LOAD;
    request.graphId = graphId;
    request.graphPath = "/models/qnn/" + graphId + ".bin";
    request.allGraphName.push_back("prefill");
    request.allGraphName.push_back("decode");
    return request;
}

DualPipelineScheduler::PipelineGraphWave makeWave(int pipelineId,
                                                  int ownerRequestId,
                                                  const std::vector<std::string>& graphIds) {
    DualPipelineScheduler::PipelineGraphWave wave;
    wave.pipelineId = pipelineId;
    wave.ownerRequestIds.push_back(ownerRequestId);
    for (size_t i = 0; i < graphIds.size(); ++i) {
        wave.graphs.push_back(makeLoadRequest(graphIds[i]));
    }
    return wave;
}

bool startScheduler(DualPipelineScheduler& scheduler,
                    GraphRecorder& recorder,
                    size_t maxResidentGraphs = 5,
                    int lookahead = 2) {
    DualPipelineScheduler::Config config;
    config.maxResidentGraphs = maxResidentGraphs;
    config.graphPrefetchLookahead = lookahead;
    config.callbacks = recorder.callbacks();
    MNNTEST_ASSERT(scheduler.configure(config));
    MNNTEST_ASSERT(scheduler.start());
    return true;
}

} // namespace

class DualPipelineSchedulerExecutionOrderGapTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder, 8, 1));

        const DualPipelineScheduler::PipelineGraphWave wave = makeWave(
            0, 17, {"order_a", "pruned_b", "order_c", "unused_tail_d"});
        MNNTEST_ASSERT(scheduler.beginGraphPrefetchWave({wave}));
        MNNTEST_ASSERT(recorder.waitForCount(2));
        MNNTEST_ASSERT(scheduler.beginStageWave({0}));

        MNNTEST_ASSERT(scheduler.enterGraphStage(0, 0));
        MNNTEST_ASSERT(scheduler.leaveGraphStage(0, 0));
        MNNTEST_ASSERT(scheduler.enterGraphStage(0, 2));
        const DualPipelineScheduler::GraphWindowSnapshot advanced = scheduler.graphWindowSnapshot();
        MNNTEST_ASSERT(advanced.pipelines.size() == 1);
        MNNTEST_ASSERT(advanced.pipelines[0].currentGraphIndex == 2);
        MNNTEST_ASSERT(advanced.pipelines[0].requestedUntil == 3);
        MNNTEST_ASSERT(scheduler.leaveGraphStage(0, 2));

        MNNTEST_ASSERT(scheduler.finishStageWave());
        MNNTEST_ASSERT(scheduler.finishGraphPrefetchWave());
        scheduler.stop();

        const std::vector<GraphEvent> events = recorder.snapshot();
        std::set<std::string> loaded;
        std::set<std::string> released;
        for (size_t i = 0; i < events.size(); ++i) {
            if (events[i].type == "load") {
                loaded.insert(events[i].graphId);
            } else if (events[i].type == "release") {
                released.insert(events[i].graphId);
            }
        }
        MNNTEST_ASSERT(loaded == std::set<std::string>({"order_a", "pruned_b", "order_c", "unused_tail_d"}));
        MNNTEST_ASSERT(released == loaded);
        return true;
    }
};

class DualPipelineSchedulerDynamicLoadPriorityTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder, 16, 2));

        std::vector<DualPipelineScheduler::PipelineGraphWave> waves;
        waves.push_back(makeWave(0, 1, {"priority_a", "priority_b", "priority_c", "priority_d"}));
        waves.push_back(makeWave(1, 2, {"priority_e", "priority_f", "priority_g", "priority_h"}));
        MNNTEST_ASSERT(scheduler.beginGraphPrefetchWave(waves));
        MNNTEST_ASSERT(recorder.waitForCount(6));
        scheduler.cancelGraphPrefetchWave();
        MNNTEST_ASSERT(!scheduler.finishGraphPrefetchWave());
        scheduler.stop();

        const std::vector<GraphEvent> events = recorder.snapshot();
        MNNTEST_ASSERT(events.size() == 12);
        MNNTEST_ASSERT(events[0].graphId == "priority_a");
        MNNTEST_ASSERT(events[1].graphId == "priority_e");
        MNNTEST_ASSERT(events[2].graphId == "priority_b");
        MNNTEST_ASSERT(events[3].graphId == "priority_f");
        MNNTEST_ASSERT(events[4].graphId == "priority_c");
        MNNTEST_ASSERT(events[5].graphId == "priority_g");
        return true;
    }
};

class DualPipelineSchedulerSharedGraphTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder, 8, 0));

        std::vector<DualPipelineScheduler::PipelineGraphWave> waves;
        waves.push_back(makeWave(0, 1, {"shared_graph"}));
        waves.push_back(makeWave(1, 2, {"shared_graph"}));
        MNNTEST_ASSERT(scheduler.beginGraphPrefetchWave(waves));
        MNNTEST_ASSERT(recorder.waitForCount(1));
        MNNTEST_ASSERT(scheduler.beginStageWave({0, 1}));
        MNNTEST_ASSERT(scheduler.enterGraphStage(0, 0));
        MNNTEST_ASSERT(scheduler.leaveGraphStage(0, 0));
        MNNTEST_ASSERT(scheduler.enterGraphStage(1, 0));
        MNNTEST_ASSERT(scheduler.leaveGraphStage(1, 0));
        MNNTEST_ASSERT(scheduler.finishStageWave());
        MNNTEST_ASSERT(scheduler.finishGraphPrefetchWave());
        scheduler.stop();

        const std::vector<GraphEvent> events = recorder.snapshot();
        MNNTEST_ASSERT(events.size() == 2);
        MNNTEST_ASSERT(events[0].type == "load");
        MNNTEST_ASSERT(events[1].type == "release");
        return true;
    }
};

class DualPipelineSchedulerGraphLoadFailureTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        recorder.failedGraphId = "failed_graph";
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder, 8, 0));

        const DualPipelineScheduler::PipelineGraphWave wave = makeWave(0, 1, {"failed_graph"});
        MNNTEST_ASSERT(scheduler.beginGraphPrefetchWave({wave}));
        MNNTEST_ASSERT(scheduler.beginStageWave({0}));
        MNNTEST_ASSERT(!scheduler.enterGraphStage(0, 0));
        scheduler.cancelStageWave();
        scheduler.cancelGraphPrefetchWave();
        MNNTEST_ASSERT(!scheduler.finishStageWave());
        MNNTEST_ASSERT(!scheduler.finishGraphPrefetchWave());
        scheduler.stop();
        MNNTEST_ASSERT(recorder.snapshot().size() == 1);
        return true;
    }
};

class DualPipelineSchedulerWaveReuseTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder, 8, 0));
        const DualPipelineScheduler::PipelineGraphWave wave = makeWave(0, 1, {"reuse_graph"});

        for (int i = 0; i < 2; ++i) {
            MNNTEST_ASSERT(scheduler.beginGraphPrefetchWave({wave}));
            MNNTEST_ASSERT(scheduler.beginStageWave({0}));
            MNNTEST_ASSERT(scheduler.enterGraphStage(0, 0));
            MNNTEST_ASSERT(scheduler.leaveGraphStage(0, 0));
            MNNTEST_ASSERT(scheduler.finishStageWave());
            MNNTEST_ASSERT(scheduler.finishGraphPrefetchWave());
        }
        scheduler.stop();
        MNNTEST_ASSERT(recorder.waitForCount(4));
        return true;
    }
};

MNNTestSuiteRegister(DualPipelineSchedulerExecutionOrderGapTest, "llm/dual_pipeline_scheduler_execution_order_gap");
MNNTestSuiteRegister(DualPipelineSchedulerDynamicLoadPriorityTest, "llm/dual_pipeline_scheduler_dynamic_load_priority");
MNNTestSuiteRegister(DualPipelineSchedulerSharedGraphTest, "llm/dual_pipeline_scheduler_shared_graph");
MNNTestSuiteRegister(DualPipelineSchedulerGraphLoadFailureTest, "llm/dual_pipeline_scheduler_graph_load_failure");
MNNTestSuiteRegister(DualPipelineSchedulerWaveReuseTest, "llm/dual_pipeline_scheduler_wave_reuse");
