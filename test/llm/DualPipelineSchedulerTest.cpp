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
#include <thread>
#include <vector>

using namespace MNN::Transformer;

namespace {

struct GraphEvent {
    std::string type;
    std::string graphId;
};

struct GraphRecorder {
    std::string failedGraphId;
    std::string blockedGraphId;

    DualPipelineScheduler::Callbacks callbacks() {
        DualPipelineScheduler::Callbacks cb;
        cb.onGraphLoad = [this](const DualPipelineScheduler::GraphRequest& request) {
            record("load", request.graphId);
            waitIfBlocked(request.graphId);
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

    bool waitForBlockedLoad(int timeoutMs = 1000) {
        std::unique_lock<std::mutex> lock(mutex);
        return condition.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this]() {
            return blockedLoadStarted;
        });
    }

    void unblockLoad() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            loadUnblocked = true;
        }
        condition.notify_all();
    }

private:
    void waitIfBlocked(const std::string& graphId) {
        if (graphId != blockedGraphId) {
            return;
        }
        std::unique_lock<std::mutex> lock(mutex);
        blockedLoadStarted = true;
        condition.notify_all();
        condition.wait(lock, [this]() {
            return loadUnblocked;
        });
    }

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
    bool blockedLoadStarted = false;
    bool loadUnblocked = false;
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
        MNNTEST_ASSERT(loaded == std::set<std::string>({"order_a", "pruned_b", "order_c"}));
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
        const std::vector<GraphEvent> events = recorder.snapshot();
        MNNTEST_ASSERT(events.size() == 2);
        MNNTEST_ASSERT(events[0].type == "load");
        MNNTEST_ASSERT(events[0].graphId == "reuse_graph");
        MNNTEST_ASSERT(events[1].type == "release");
        MNNTEST_ASSERT(events[1].graphId == "reuse_graph");
        return true;
    }
};

class DualPipelineSchedulerResidentLruTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder, 2, 0));

        const std::vector<std::string> graphIds = {"lru_a", "lru_b", "lru_c"};
        for (size_t i = 0; i < graphIds.size(); ++i) {
            const DualPipelineScheduler::PipelineGraphWave wave = makeWave(0, 10 + static_cast<int>(i), {graphIds[i]});
            MNNTEST_ASSERT(scheduler.beginGraphPrefetchWave({wave}));
            MNNTEST_ASSERT(scheduler.beginStageWave({0}));
            MNNTEST_ASSERT(scheduler.enterGraphStage(0, 0));
            MNNTEST_ASSERT(scheduler.leaveGraphStage(0, 0));
            MNNTEST_ASSERT(scheduler.finishStageWave());
            MNNTEST_ASSERT(scheduler.finishGraphPrefetchWave());
        }
        scheduler.stop();

        const std::vector<GraphEvent> events = recorder.snapshot();
        MNNTEST_ASSERT(events.size() == 6);
        MNNTEST_ASSERT(events[0].type == "load" && events[0].graphId == "lru_a");
        MNNTEST_ASSERT(events[1].type == "load" && events[1].graphId == "lru_b");
        MNNTEST_ASSERT(events[2].type == "release" && events[2].graphId == "lru_a");
        MNNTEST_ASSERT(events[3].type == "load" && events[3].graphId == "lru_c");
        std::set<std::string> shutdownReleases;
        shutdownReleases.insert(events[4].graphId);
        shutdownReleases.insert(events[5].graphId);
        MNNTEST_ASSERT(events[4].type == "release" && events[5].type == "release");
        MNNTEST_ASSERT(shutdownReleases == std::set<std::string>({"lru_b", "lru_c"}));
        return true;
    }
};

class DualPipelineSchedulerRequestOwnerCleanupTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder, 2, 0));
        const DualPipelineScheduler::PipelineGraphWave wave = makeWave(0, 23, {"owner_graph"});

        MNNTEST_ASSERT(scheduler.beginGraphPrefetchWave({wave}));
        MNNTEST_ASSERT(scheduler.beginStageWave({0}));
        MNNTEST_ASSERT(scheduler.enterGraphStage(0, 0));
        MNNTEST_ASSERT(scheduler.leaveGraphStage(0, 0));
        MNNTEST_ASSERT(scheduler.finishStageWave());
        MNNTEST_ASSERT(scheduler.finishGraphPrefetchWave());
        MNNTEST_ASSERT(scheduler.releaseRequestGraphs(23) == 1);
        MNNTEST_ASSERT(scheduler.releaseRequestGraphs(23) == 0);
        scheduler.stop();
        return true;
    }
};

class DualPipelineSchedulerActiveGraphProtectionTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder, 1, 0));
        const std::vector<DualPipelineScheduler::PipelineGraphWave> waves = {
            makeWave(0, 31, {"active_a"}),
            makeWave(1, 32, {"active_b"})
        };

        MNNTEST_ASSERT(scheduler.beginGraphPrefetchWave(waves));
        MNNTEST_ASSERT(recorder.waitForCount(2));
        MNNTEST_ASSERT(scheduler.beginStageWave({0, 1}));
        MNNTEST_ASSERT(scheduler.enterGraphStage(0, 0));
        MNNTEST_ASSERT(scheduler.leaveGraphStage(0, 0));
        MNNTEST_ASSERT(scheduler.enterGraphStage(1, 0));
        MNNTEST_ASSERT(scheduler.leaveGraphStage(1, 0));
        MNNTEST_ASSERT(scheduler.finishStageWave());
        MNNTEST_ASSERT(scheduler.finishGraphPrefetchWave());
        scheduler.stop();

        const std::vector<GraphEvent> events = recorder.snapshot();
        MNNTEST_ASSERT(events.size() == 4);
        MNNTEST_ASSERT(events[0].type == "load" && events[0].graphId == "active_a");
        MNNTEST_ASSERT(events[1].type == "load" && events[1].graphId == "active_b");
        return true;
    }
};

class DualPipelineSchedulerExecutionBeforeLookaheadTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        recorder.blockedGraphId = "gate_a";
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder, 2, 1));
        const DualPipelineScheduler::PipelineGraphWave wave = makeWave(0, 41, {"gate_a", "gate_b"});

        MNNTEST_ASSERT(scheduler.beginGraphPrefetchWave({wave}));
        MNNTEST_ASSERT(recorder.waitForBlockedLoad());
        MNNTEST_ASSERT(scheduler.beginStageWave({0}));
        bool entered = false;
        bool graphStageWaiting = false;
        std::thread enterThread([&scheduler, &entered]() {
            entered = scheduler.enterGraphStage(0, 0);
        });
        for (int i = 0; i < 100; ++i) {
            const DualPipelineScheduler::GraphWindowSnapshot snapshot = scheduler.graphWindowSnapshot();
            if (!snapshot.pipelines.empty() && snapshot.pipelines[0].currentGraphIndex == 0) {
                graphStageWaiting = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        recorder.unblockLoad();
        enterThread.join();
        MNNTEST_ASSERT(graphStageWaiting);
        MNNTEST_ASSERT(entered);
        MNNTEST_ASSERT(recorder.snapshot().size() == 1);

        MNNTEST_ASSERT(scheduler.leaveGraphStage(0, 0));
        MNNTEST_ASSERT(recorder.waitForCount(2));
        MNNTEST_ASSERT(scheduler.finishStageWave());
        MNNTEST_ASSERT(scheduler.finishGraphPrefetchWave());
        scheduler.stop();
        return true;
    }
};

MNNTestSuiteRegister(DualPipelineSchedulerExecutionOrderGapTest, "llm/dual_pipeline_scheduler_execution_order_gap");
MNNTestSuiteRegister(DualPipelineSchedulerDynamicLoadPriorityTest, "llm/dual_pipeline_scheduler_dynamic_load_priority");
MNNTestSuiteRegister(DualPipelineSchedulerSharedGraphTest, "llm/dual_pipeline_scheduler_shared_graph");
MNNTestSuiteRegister(DualPipelineSchedulerGraphLoadFailureTest, "llm/dual_pipeline_scheduler_graph_load_failure");
MNNTestSuiteRegister(DualPipelineSchedulerWaveReuseTest, "llm/dual_pipeline_scheduler_wave_reuse");
MNNTestSuiteRegister(DualPipelineSchedulerResidentLruTest, "llm/dual_pipeline_scheduler_resident_lru");
MNNTestSuiteRegister(DualPipelineSchedulerRequestOwnerCleanupTest, "llm/dual_pipeline_scheduler_request_owner_cleanup");
MNNTestSuiteRegister(DualPipelineSchedulerActiveGraphProtectionTest, "llm/dual_pipeline_scheduler_active_graph_protection");
MNNTestSuiteRegister(DualPipelineSchedulerExecutionBeforeLookaheadTest, "llm/dual_pipeline_scheduler_execution_before_lookahead");
