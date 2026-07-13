//
//  DualPipelineSchedulerTest.cpp
//  MNN
//

#include <MNN/MNNDefine.h>
#include "../MNNTestSuite.h"
#include "../../transformers/llm/engine/include/llm/DualPipelineScheduler.hpp"

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace MNN::Transformer;

namespace {

struct GraphEvent {
    std::string type;
    std::string graphId;
    std::string reason;
    std::string baseDir;
    std::string npuDir;
    std::string relativePath;
    std::string targetGraphName;
    uint64_t offset;
    uint64_t size;
    int shapeIndex;
};

struct GraphRecorder {
    DualPipelineScheduler::Callbacks callbacks() {
        DualPipelineScheduler::Callbacks cb;
        cb.onGraphLoad = [this](const DualPipelineScheduler::GraphRequest& request) {
            recordGraph("load", request);
            return true;
        };
        cb.onGraphRelease = [this](const DualPipelineScheduler::GraphRequest& request) {
            recordGraph("release", request);
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
    void recordGraph(const std::string& type, const DualPipelineScheduler::GraphRequest& request) {
        GraphEvent event;
        event.type = type;
        event.graphId = request.graphId;
        event.reason = request.reason;
        event.baseDir = request.baseDir;
        event.npuDir = request.npuDir;
        event.relativePath = request.relativePath;
        event.targetGraphName = request.targetGraphName;
        event.offset = request.offset;
        event.size = request.size;
        event.shapeIndex = request.shapeIndex;
        record(event);
    }

    void record(const GraphEvent& event) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            events.push_back(event);
        }
        condition.notify_all();
    }

    mutable std::mutex mutex;
    std::condition_variable condition;
    std::vector<GraphEvent> events;
};

DualPipelineScheduler::GraphRequest makeLoadRequest(const std::string& graphId, int requestId, bool pinned = false) {
    DualPipelineScheduler::GraphRequest request;
    request.action = DualPipelineScheduler::GRAPH_LOAD;
    request.requestId = requestId;
    request.graphId = graphId;
    request.graphPath = "/models/qnn/" + graphId + ".bin";
    request.baseDir = "/models";
    request.npuDir = "qnn";
    request.relativePath = "graphs/" + graphId + ".serialized";
    request.offset = static_cast<uint64_t>(requestId) * 100;
    request.size = static_cast<uint64_t>(requestId) * 1000;
    request.allGraphName.push_back("prefill");
    request.allGraphName.push_back("decode");
    request.targetGraphName = graphId + "_target";
    request.shapeIndex = requestId;
    request.reason = "load";
    request.draftGraph = pinned;
    request.pinResident = pinned;
    return request;
}

DualPipelineScheduler::GraphRequest makeReleaseRequest(const std::string& graphId, const std::string& reason) {
    DualPipelineScheduler::GraphRequest request;
    request.action = DualPipelineScheduler::GRAPH_RELEASE;
    request.requestId = 77;
    request.graphId = graphId;
    request.baseDir = "/models";
    request.npuDir = "qnn";
    request.relativePath = "graphs/" + graphId + ".serialized";
    request.offset = 700;
    request.size = 7000;
    request.allGraphName.push_back("prefill");
    request.allGraphName.push_back("decode");
    request.targetGraphName = graphId + "_target";
    request.shapeIndex = 7;
    request.reason = reason;
    return request;
}

bool startScheduler(DualPipelineScheduler& scheduler, GraphRecorder& recorder, size_t maxResidentGraphs = 5) {
    DualPipelineScheduler::Config config;
    config.maxResidentGraphs = maxResidentGraphs;
    config.callbacks = recorder.callbacks();
    MNNTEST_ASSERT(scheduler.configure(config));
    MNNTEST_ASSERT(scheduler.start());
    return true;
}

} // namespace

class DualPipelineSchedulerGraphLoadCompleteReleaseTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder));

        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("graph_a", 1)));
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("graph_b", 2)));
        MNNTEST_ASSERT(recorder.waitForCount(2));

        scheduler.markExecutionComplete("graph_a");
        scheduler.markExecutionComplete("graph_b");
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeReleaseRequest("graph_a", "request_complete")));
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeReleaseRequest("graph_b", "request_complete")));
        MNNTEST_ASSERT(recorder.waitForCount(4));

        scheduler.stop();
        const std::vector<GraphEvent> events = recorder.snapshot();
        MNNTEST_ASSERT(events.size() == 4);
        MNNTEST_ASSERT(events[0].type == "load" && events[0].graphId == "graph_a");
        MNNTEST_ASSERT(events[1].type == "load" && events[1].graphId == "graph_b");
        MNNTEST_ASSERT(events[2].type == "release" && events[2].graphId == "graph_a");
        MNNTEST_ASSERT(events[3].type == "release" && events[3].graphId == "graph_b");
        MNNTEST_ASSERT(events[0].baseDir == "/models");
        MNNTEST_ASSERT(events[0].npuDir == "qnn");
        MNNTEST_ASSERT(events[0].relativePath == "graphs/graph_a.serialized");
        MNNTEST_ASSERT(events[0].targetGraphName == "graph_a_target");
        MNNTEST_ASSERT(events[0].offset == 100);
        MNNTEST_ASSERT(events[0].size == 1000);
        MNNTEST_ASSERT(events[0].shapeIndex == 1);
        MNNTEST_ASSERT(events[2].reason == "request_complete");
        return true;
    }
};

class DualPipelineSchedulerQueuedCompleteAfterLoadTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder));

        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("queued_graph", 1)));
        scheduler.markExecutionComplete("queued_graph");
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeReleaseRequest("queued_graph", "queued_release")));
        MNNTEST_ASSERT(recorder.waitForCount(2));

        scheduler.stop();
        const std::vector<GraphEvent> events = recorder.snapshot();
        MNNTEST_ASSERT(events.size() == 2);
        MNNTEST_ASSERT(events[0].type == "load" && events[0].graphId == "queued_graph");
        MNNTEST_ASSERT(events[1].type == "release" && events[1].graphId == "queued_graph");
        MNNTEST_ASSERT(events[1].reason == "queued_release");
        return true;
    }
};

class DualPipelineSchedulerGraphLoadFailureTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        DualPipelineScheduler scheduler;
        bool releaseCalled = false;
        DualPipelineScheduler::Config config;
        config.callbacks.onGraphLoad = [](const DualPipelineScheduler::GraphRequest&) {
            return false;
        };
        config.callbacks.onGraphRelease = [&releaseCalled](const DualPipelineScheduler::GraphRequest&) {
            releaseCalled = true;
        };
        MNNTEST_ASSERT(scheduler.configure(config));
        MNNTEST_ASSERT(scheduler.start());

        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("missing_graph", 9)));
        MNNTEST_ASSERT(!scheduler.waitForGraphsReady(std::vector<std::string>({"missing_graph"})));
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeReleaseRequest("missing_graph", "after_failed_load")));
        scheduler.stop();

        const DualPipelineScheduler::Snapshot snapshot = scheduler.snapshot();
        MNNTEST_ASSERT(snapshot.residentGraphs == 0);
        MNNTEST_ASSERT(snapshot.graphs.size() == 1);
        MNNTEST_ASSERT(snapshot.graphs[0].graphId == "missing_graph");
        MNNTEST_ASSERT(!snapshot.graphs[0].resident);
        MNNTEST_ASSERT(snapshot.graphs[0].activeUseCount == 0);
        MNNTEST_ASSERT(!releaseCalled);
        return true;
    }
};

class DualPipelineSchedulerWaitForGraphReadyTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        DualPipelineScheduler scheduler;
        std::mutex gateMutex;
        std::condition_variable gateCondition;
        bool loadEntered = false;
        bool allowLoad = false;
        DualPipelineScheduler::Config config;
        config.callbacks.onGraphLoad = [&](const DualPipelineScheduler::GraphRequest&) {
            std::unique_lock<std::mutex> lock(gateMutex);
            loadEntered = true;
            gateCondition.notify_all();
            gateCondition.wait(lock, [&]() { return allowLoad; });
            return true;
        };
        MNNTEST_ASSERT(scheduler.configure(config));
        MNNTEST_ASSERT(scheduler.start());
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("ready_graph", 10)));
        {
            std::unique_lock<std::mutex> lock(gateMutex);
            MNNTEST_ASSERT(gateCondition.wait_for(lock, std::chrono::seconds(1), [&]() { return loadEntered; }));
        }

        std::atomic<bool> waitFinished(false);
        bool ready = false;
        std::thread waiter([&]() {
            ready = scheduler.waitForGraphsReady(std::vector<std::string>({"ready_graph"}));
            waitFinished.store(true);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const bool finishedEarly = waitFinished.load();
        {
            std::lock_guard<std::mutex> lock(gateMutex);
            allowLoad = true;
        }
        gateCondition.notify_all();
        waiter.join();
        scheduler.stop();
        MNNTEST_ASSERT(!finishedEarly);
        MNNTEST_ASSERT(ready);
        return true;
    }
};

class DualPipelineSchedulerSharedGraphUseCountTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder));

        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("shared_graph", 1)));
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("shared_graph", 2)));
        MNNTEST_ASSERT(recorder.waitForCount(1));

        scheduler.markExecutionComplete("shared_graph");
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeReleaseRequest("shared_graph", "first_request_done")));
        MNNTEST_ASSERT(!recorder.waitForCount(2, 100));

        scheduler.markExecutionComplete("shared_graph");
        MNNTEST_ASSERT(recorder.waitForCount(2));

        scheduler.stop();
        const std::vector<GraphEvent> events = recorder.snapshot();
        MNNTEST_ASSERT(events.size() == 2);
        MNNTEST_ASSERT(events[0].type == "load" && events[0].graphId == "shared_graph");
        MNNTEST_ASSERT(events[1].type == "release" && events[1].graphId == "shared_graph");
        MNNTEST_ASSERT(events[1].reason == "first_request_done");
        return true;
    }
};

class DualPipelineSchedulerPinnedGraphDefaultReleaseTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder));

        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("draft_graph", 3, true)));
        MNNTEST_ASSERT(recorder.waitForCount(1));
        scheduler.markExecutionComplete("draft_graph");
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeReleaseRequest("draft_graph", "normal_release")));
        MNNTEST_ASSERT(!recorder.waitForCount(2, 100));

        const DualPipelineScheduler::Snapshot snapshot = scheduler.snapshot();
        scheduler.stop();
        MNNTEST_ASSERT(snapshot.residentGraphs == 1);
        MNNTEST_ASSERT(snapshot.graphs.size() == 1);
        MNNTEST_ASSERT(snapshot.graphs[0].graphId == "draft_graph");
        MNNTEST_ASSERT(snapshot.graphs[0].resident);
        MNNTEST_ASSERT(snapshot.graphs[0].pinned);
        return true;
    }
};

class DualPipelineSchedulerPinnedGraphForcedReleaseTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder));

        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("forced_graph", 4, true)));
        MNNTEST_ASSERT(recorder.waitForCount(1));
        scheduler.markExecutionComplete("forced_graph");

        DualPipelineScheduler::GraphRequest forceRelease = makeReleaseRequest("forced_graph", "force");
        forceRelease.forceRelease = true;
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(forceRelease));
        MNNTEST_ASSERT(recorder.waitForCount(2));

        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("unpin_graph", 5, true)));
        MNNTEST_ASSERT(recorder.waitForCount(3));
        scheduler.markExecutionComplete("unpin_graph");

        DualPipelineScheduler::GraphRequest unpinRelease = makeReleaseRequest("unpin_graph", "unpin");
        unpinRelease.unpinAfterRelease = true;
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(unpinRelease));
        MNNTEST_ASSERT(recorder.waitForCount(4));

        scheduler.stop();
        const std::vector<GraphEvent> events = recorder.snapshot();
        MNNTEST_ASSERT(events.size() == 4);
        MNNTEST_ASSERT(events[1].type == "release" && events[1].graphId == "forced_graph");
        MNNTEST_ASSERT(events[1].reason == "force");
        MNNTEST_ASSERT(events[3].type == "release" && events[3].graphId == "unpin_graph");
        MNNTEST_ASSERT(events[3].reason == "unpin");
        return true;
    }
};

class DualPipelineSchedulerMaxResidentGraphsLruEvictTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphRecorder recorder;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(startScheduler(scheduler, recorder, 2));

        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("lru_a", 1)));
        MNNTEST_ASSERT(recorder.waitForCount(1));
        scheduler.markExecutionComplete("lru_a");
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("lru_b", 2)));
        MNNTEST_ASSERT(recorder.waitForCount(2));
        scheduler.markExecutionComplete("lru_b");

        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("lru_c", 3)));
        MNNTEST_ASSERT(recorder.waitForCount(4));
        scheduler.markExecutionComplete("lru_c");
        MNNTEST_ASSERT(scheduler.enqueueRequestGraph(makeLoadRequest("lru_d", 4)));
        MNNTEST_ASSERT(recorder.waitForCount(6));

        scheduler.stop();
        const std::vector<GraphEvent> events = recorder.snapshot();
        MNNTEST_ASSERT(events.size() == 6);
        MNNTEST_ASSERT(events[0].type == "load" && events[0].graphId == "lru_a");
        MNNTEST_ASSERT(events[1].type == "load" && events[1].graphId == "lru_b");
        MNNTEST_ASSERT(events[2].type == "release" && events[2].graphId == "lru_a");
        MNNTEST_ASSERT(events[3].type == "load" && events[3].graphId == "lru_c");
        MNNTEST_ASSERT(events[4].type == "release" && events[4].graphId == "lru_b");
        MNNTEST_ASSERT(events[5].type == "load" && events[5].graphId == "lru_d");
        MNNTEST_ASSERT(events[2].reason == "maxResidentGraphs");
        MNNTEST_ASSERT(events[4].relativePath == "graphs/lru_b.serialized");
        return true;
    }
};

MNNTestSuiteRegister(DualPipelineSchedulerGraphLoadCompleteReleaseTest, "llm/dual_pipeline_scheduler_load_complete_release");
MNNTestSuiteRegister(DualPipelineSchedulerQueuedCompleteAfterLoadTest, "llm/dual_pipeline_scheduler_queued_complete_after_load");
MNNTestSuiteRegister(DualPipelineSchedulerGraphLoadFailureTest, "llm/dual_pipeline_scheduler_graph_load_failure");
MNNTestSuiteRegister(DualPipelineSchedulerWaitForGraphReadyTest, "llm/dual_pipeline_scheduler_wait_for_graph_ready");
MNNTestSuiteRegister(DualPipelineSchedulerSharedGraphUseCountTest, "llm/dual_pipeline_scheduler_shared_graph_use_count");
MNNTestSuiteRegister(DualPipelineSchedulerPinnedGraphDefaultReleaseTest, "llm/dual_pipeline_scheduler_pinned_default_release");
MNNTestSuiteRegister(DualPipelineSchedulerPinnedGraphForcedReleaseTest, "llm/dual_pipeline_scheduler_pinned_forced_release");
MNNTestSuiteRegister(DualPipelineSchedulerMaxResidentGraphsLruEvictTest, "llm/dual_pipeline_scheduler_lru_evict");
