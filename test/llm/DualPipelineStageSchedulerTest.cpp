//
//  DualPipelineStageSchedulerTest.cpp
//  MNN
//

#include <MNN/MNNDefine.h>
#include "../MNNTestSuite.h"
#include "../../transformers/llm/engine/include/llm/DualPipelineScheduler.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

using namespace MNN::Transformer;

class DualPipelineStageHostQnnTimelineTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        (void)precision;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(scheduler.beginStageWave(std::vector<int>({0, 1})));

        std::mutex mutex;
        std::condition_variable condition;
        bool qnnStarted = false;
        bool hostFinished = false;
        std::atomic<bool> threadSucceeded(true);
        std::chrono::steady_clock::time_point qnnBegin;
        std::chrono::steady_clock::time_point qnnEnd;
        std::chrono::steady_clock::time_point hostBegin;
        std::chrono::steady_clock::time_point hostEnd;

        std::thread qnn([&]() {
            if (!scheduler.enterStage(0, DualPipelineScheduler::STAGE_QNN)) {
                threadSucceeded.store(false);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    qnnStarted = true;
                    hostFinished = true;
                }
                condition.notify_all();
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                qnnBegin = std::chrono::steady_clock::now();
                qnnStarted = true;
            }
            condition.notify_all();
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [&]() { return hostFinished; });
                qnnEnd = std::chrono::steady_clock::now();
            }
            if (!scheduler.leaveStage(0)) {
                threadSucceeded.store(false);
            }
        });

        std::thread host([&]() {
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [&]() { return qnnStarted; });
            }
            if (!scheduler.enterStage(1, DualPipelineScheduler::STAGE_HOST)) {
                threadSucceeded.store(false);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    hostFinished = true;
                }
                condition.notify_all();
                return;
            }
            hostBegin = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            hostEnd = std::chrono::steady_clock::now();
            if (!scheduler.leaveStage(1)) {
                threadSucceeded.store(false);
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                hostFinished = true;
            }
            condition.notify_all();
        });

        qnn.join();
        host.join();

        const DualPipelineScheduler::StageSnapshot snapshot = scheduler.stageSnapshot();
        MNNTEST_ASSERT(threadSucceeded.load());
        MNNTEST_ASSERT(snapshot.completedStages == 2);
        MNNTEST_ASSERT(snapshot.completedHostStages == 1);
        MNNTEST_ASSERT(snapshot.completedQnnStages == 1);
        MNNTEST_ASSERT(snapshot.maxConcurrentHostStages == 1);
        MNNTEST_ASSERT(snapshot.maxConcurrentQnnStages == 1);
        MNNTEST_ASSERT(snapshot.hostQnnOverlapGrants > 0);
        MNNTEST_ASSERT(qnnBegin <= hostBegin);
        MNNTEST_ASSERT(hostBegin < hostEnd);
        MNNTEST_ASSERT(hostEnd <= qnnEnd);
        MNNTEST_ASSERT(scheduler.finishStageWave());
        return true;
    }
};

class DualPipelineStageQnnReadyQueueTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        (void)precision;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(scheduler.beginStageWave(std::vector<int>({0, 1})));

        std::mutex mutex;
        std::condition_variable condition;
        bool firstQnnStarted = false;
        bool releaseFirstQnn = false;
        std::atomic<int> activeQnn(0);
        std::atomic<int> maxActiveQnn(0);
        std::atomic<bool> threadSucceeded(true);

        auto updateMax = [&]() {
            const int active = ++activeQnn;
            int previous = maxActiveQnn.load();
            while (active > previous && !maxActiveQnn.compare_exchange_weak(previous, active)) {
            }
        };

        std::thread first([&]() {
            if (!scheduler.enterStage(0, DualPipelineScheduler::STAGE_QNN)) {
                threadSucceeded.store(false);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    firstQnnStarted = true;
                }
                condition.notify_all();
                return;
            }
            updateMax();
            {
                std::lock_guard<std::mutex> lock(mutex);
                firstQnnStarted = true;
            }
            condition.notify_all();
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [&]() { return releaseFirstQnn; });
            }
            --activeQnn;
            if (!scheduler.leaveStage(0)) {
                threadSucceeded.store(false);
            }
        });

        std::thread second([&]() {
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [&]() { return firstQnnStarted; });
            }
            if (!scheduler.enterStage(1, DualPipelineScheduler::STAGE_QNN)) {
                threadSucceeded.store(false);
                return;
            }
            updateMax();
            --activeQnn;
            if (!scheduler.leaveStage(1)) {
                threadSucceeded.store(false);
            }
        });

        bool observedQueuedQnn = false;
        for (int i = 0; i < 100; ++i) {
            if (scheduler.stageSnapshot().qnnReadyStages == 1) {
                observedQueuedQnn = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            releaseFirstQnn = true;
        }
        condition.notify_all();
        first.join();
        second.join();

        const DualPipelineScheduler::StageSnapshot snapshot = scheduler.stageSnapshot();
        MNNTEST_ASSERT(threadSucceeded.load());
        MNNTEST_ASSERT(observedQueuedQnn);
        MNNTEST_ASSERT(maxActiveQnn.load() == 1);
        MNNTEST_ASSERT(snapshot.maxConcurrentQnnStages == 1);
        MNNTEST_ASSERT(snapshot.completedStages == 2);
        MNNTEST_ASSERT(snapshot.completedHostStages == 0);
        MNNTEST_ASSERT(snapshot.completedQnnStages == 2);
        MNNTEST_ASSERT(scheduler.finishStageWave());
        return true;
    }
};

class DualPipelineStageHostReadyQueueTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        (void)precision;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(scheduler.beginStageWave(std::vector<int>({0, 1})));

        std::mutex mutex;
        std::condition_variable condition;
        bool firstHostStarted = false;
        bool releaseFirstHost = false;
        std::atomic<int> activeHost(0);
        std::atomic<int> maxActiveHost(0);
        std::atomic<bool> threadSucceeded(true);

        auto updateMax = [&]() {
            const int active = ++activeHost;
            int previous = maxActiveHost.load();
            while (active > previous && !maxActiveHost.compare_exchange_weak(previous, active)) {
            }
        };

        std::thread first([&]() {
            if (!scheduler.enterStage(0, DualPipelineScheduler::STAGE_HOST)) {
                threadSucceeded.store(false);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    firstHostStarted = true;
                }
                condition.notify_all();
                return;
            }
            updateMax();
            {
                std::lock_guard<std::mutex> lock(mutex);
                firstHostStarted = true;
            }
            condition.notify_all();
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [&]() { return releaseFirstHost; });
            }
            --activeHost;
            if (!scheduler.leaveStage(0)) {
                threadSucceeded.store(false);
            }
        });

        std::thread second([&]() {
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [&]() { return firstHostStarted; });
            }
            if (!scheduler.enterStage(1, DualPipelineScheduler::STAGE_HOST)) {
                threadSucceeded.store(false);
                return;
            }
            updateMax();
            --activeHost;
            if (!scheduler.leaveStage(1)) {
                threadSucceeded.store(false);
            }
        });

        bool observedQueuedHost = false;
        for (int i = 0; i < 100; ++i) {
            if (scheduler.stageSnapshot().hostReadyStages == 1) {
                observedQueuedHost = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            releaseFirstHost = true;
        }
        condition.notify_all();
        first.join();
        second.join();

        const DualPipelineScheduler::StageSnapshot snapshot = scheduler.stageSnapshot();
        MNNTEST_ASSERT(threadSucceeded.load());
        MNNTEST_ASSERT(observedQueuedHost);
        MNNTEST_ASSERT(maxActiveHost.load() == 1);
        MNNTEST_ASSERT(snapshot.completedHostStages == 2);
        MNNTEST_ASSERT(snapshot.completedQnnStages == 0);
        MNNTEST_ASSERT(snapshot.maxConcurrentHostStages == 1);
        MNNTEST_ASSERT(scheduler.finishStageWave());
        return true;
    }
};

class DualPipelineStageCancellationTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        (void)precision;
        DualPipelineScheduler scheduler;
        MNNTEST_ASSERT(scheduler.beginStageWave(std::vector<int>({0, 1})));
        MNNTEST_ASSERT(scheduler.enterStage(0, DualPipelineScheduler::STAGE_QNN));

        std::atomic<bool> waiterReturned(false);
        bool waiterGranted = true;
        std::thread waiter([&]() {
            waiterGranted = scheduler.enterStage(1, DualPipelineScheduler::STAGE_QNN);
            waiterReturned.store(true);
        });
        for (int i = 0; i < 100 && scheduler.stageSnapshot().qnnReadyStages == 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        scheduler.cancelStageWave();
        waiter.join();

        MNNTEST_ASSERT(waiterReturned.load());
        MNNTEST_ASSERT(!waiterGranted);
        MNNTEST_ASSERT(!scheduler.leaveStage(0));
        MNNTEST_ASSERT(!scheduler.finishStageWave());
        return true;
    }
};

MNNTestSuiteRegister(DualPipelineStageHostQnnTimelineTest, "llm/dual_pipeline_stage_host_qnn_timeline");
MNNTestSuiteRegister(DualPipelineStageQnnReadyQueueTest, "llm/dual_pipeline_stage_qnn_ready_queue");
MNNTestSuiteRegister(DualPipelineStageHostReadyQueueTest, "llm/dual_pipeline_stage_host_ready_queue");
MNNTestSuiteRegister(DualPipelineStageCancellationTest, "llm/dual_pipeline_stage_cancellation");
