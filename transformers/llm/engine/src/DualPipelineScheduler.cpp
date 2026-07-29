//
//  DualPipelineScheduler.cpp
//  MNN
//

#include "llm/DualPipelineScheduler.hpp"
#include "llm/AcceptanceTrace.hpp"

#include <algorithm>
#include <limits>

namespace MNN {
namespace Transformer {

DualPipelineScheduler::StageSnapshot::StageSnapshot()
    : waveActive(false),
      cancelled(false),
      hostReadyStages(0),
      qnnReadyStages(0),
      activeHostStages(0),
      activeQnnStages(0),
      maxConcurrentHostStages(0),
      maxConcurrentQnnStages(0),
      completedStages(0),
      completedHostStages(0),
      completedQnnStages(0),
      hostQnnOverlapGrants(0) {
}

DualPipelineScheduler::GraphRequest::GraphRequest()
    : action(GRAPH_LOAD),
      pipelineId(-1),
      graphIndex(-1),
      shapeIndex(-1),
      bucketSize(-1),
      offset(0),
      size(0),
      draftGraph(false),
      pinResident(false),
      forceRelease(false),
      unpinAfterRelease(false) {
}

DualPipelineScheduler::GraphRecord::GraphRecord()
    : resident(false),
      pinned(false),
      activeUseCount(0),
      lastUseSequence(0) {
}

DualPipelineScheduler::PipelineGraphWave::PipelineGraphWave()
    : pipelineId(-1) {
}

DualPipelineScheduler::PipelineGraphProgress::PipelineGraphProgress()
    : pipelineId(-1),
      currentGraphIndex(-1),
      requestedUntil(-1),
      completedGraphIndex(-1),
      executingGraphIndex(-1) {
}

DualPipelineScheduler::GraphWindowSnapshot::GraphWindowSnapshot()
    : waveActive(false),
      cancelled(false),
      pendingLoadTasks(0) {
}

DualPipelineScheduler::Config::Config()
    : maxResidentGraphs(5),
      graphPrefetchLookahead(2) {
}

DualPipelineScheduler::Task::Task()
    : type(TASK_GRAPH_LOAD),
      enqueueSequence(0) {
}

DualPipelineScheduler::GraphState::GraphState()
    : loadFinished(false) {
}

DualPipelineScheduler::StageWaiter::StageWaiter()
    : pipelineId(-1),
      backend(STAGE_HOST),
      granted(false) {
}

DualPipelineScheduler::PipelineGraphState::PipelineGraphState()
    : currentGraphIndex(-1),
      requestedUntil(-1),
      completedGraphIndex(-1),
      executingGraphIndex(-1) {
}

DualPipelineScheduler::DualPipelineScheduler()
    : mNextSequence(1),
      mNextTaskSequence(1),
      mRunning(false),
      mStopRequested(false),
      mGraphPrefetchWaveActive(false),
      mGraphPrefetchCancelled(false),
      mQnnExecutionActive(false),
      mActiveHostStages(0),
      mActiveQnnStages(0),
      mMaxConcurrentHostStages(0),
      mMaxConcurrentQnnStages(0),
      mCompletedStages(0),
      mCompletedHostStages(0),
      mCompletedQnnStages(0),
      mHostQnnOverlapGrants(0),
      mStageWaveActive(false),
      mStageCancelled(false) {
}

DualPipelineScheduler::~DualPipelineScheduler() {
    stop();
}

bool DualPipelineScheduler::configure(const Config& config) {
    std::lock_guard<std::mutex> lock(mMutex);
    mConfig = config;
    if (mConfig.maxResidentGraphs <= 0) {
        mConfig.maxResidentGraphs = 5;
    }
    return true;
}

bool DualPipelineScheduler::start() {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mRunning) {
        return true;
    }
    mStopRequested = false;
    mWorker = std::thread(&DualPipelineScheduler::_workerLoop, this);
    mRunning = true;
    return true;
}

void DualPipelineScheduler::stop() {
    Callbacks callbacks;
    std::vector<GraphRequest> releaseRequests;
    cancelStageWave();
    cancelGraphPrefetchWave();
    finishGraphPrefetchWave();
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mRunning && !mWorker.joinable()) {
            return;
        }
        mStopRequested = true;
    }
    mCondition.notify_all();
    if (mWorker.joinable()) {
        mWorker.join();
    }
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mRunning = false;
        mStopRequested = false;
        callbacks = mConfig.callbacks;
        for (std::map<std::string, GraphState>::iterator iter = mGraphs.begin(); iter != mGraphs.end(); ++iter) {
            if (!iter->second.record.resident) {
                continue;
            }
            GraphRequest releaseRequest = iter->second.lastRequest;
            releaseRequest.action = GRAPH_RELEASE;
            releaseRequest.reason = "scheduler_stopped";
            releaseRequest.forceRelease = true;
            releaseRequest.unpinAfterRelease = true;
            releaseRequests.push_back(releaseRequest);
        }
        mGraphs.clear();
        mTasks.clear();
    }
    if (callbacks.onGraphRelease) {
        for (size_t i = 0; i < releaseRequests.size(); ++i) {
            callbacks.onGraphRelease(releaseRequests[i]);
        }
    }
}

bool DualPipelineScheduler::beginGraphPrefetchWave(const std::vector<PipelineGraphWave>& waves) {
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mGraphPrefetchWaveActive) {
            return false;
        }
        std::set<int> pipelineIds;
        for (size_t i = 0; i < waves.size(); ++i) {
            if (waves[i].pipelineId < 0 || !pipelineIds.insert(waves[i].pipelineId).second) {
                return false;
            }
            for (size_t graphIndex = 0; graphIndex < waves[i].graphs.size(); ++graphIndex) {
                if (waves[i].graphs[graphIndex].graphId.empty()) {
                    return false;
                }
            }
        }

        mPipelineGraphs.clear();
        mGraphPrefetchCancelled = false;
        for (size_t i = 0; i < waves.size(); ++i) {
            if (waves[i].graphs.empty()) {
                continue;
            }
            PipelineGraphState& state = mPipelineGraphs[waves[i].pipelineId];
            state.graphs = waves[i].graphs;
            state.ownerRequestIds = waves[i].ownerRequestIds;
        }
        mGraphPrefetchWaveActive = !mPipelineGraphs.empty();
        if (mGraphPrefetchWaveActive) {
            for (std::map<int, PipelineGraphState>::iterator iter = mPipelineGraphs.begin(); iter != mPipelineGraphs.end(); ++iter) {
                _extendGraphWindowLocked(iter->first, mConfig.graphPrefetchLookahead);
            }
        }
    }
    mCondition.notify_all();
    return true;
}

bool DualPipelineScheduler::enterGraphStage(int pipelineId, int graphIndex) {
    std::string graphId;
    {
        std::unique_lock<std::mutex> lock(mMutex);
        std::map<int, PipelineGraphState>::iterator iter = mPipelineGraphs.find(pipelineId);
        if (!mGraphPrefetchWaveActive || mGraphPrefetchCancelled || iter == mPipelineGraphs.end() ||
            graphIndex <= iter->second.currentGraphIndex || graphIndex < 0 ||
            graphIndex >= static_cast<int>(iter->second.graphs.size())) {
            return false;
        }
        const int firstSkippedGraph = iter->second.currentGraphIndex + 1;
        if (graphIndex > firstSkippedGraph) {
            _removePendingGraphLoadsLocked(pipelineId, firstSkippedGraph, graphIndex);
            for (std::set<int>::iterator skipped = iter->second.registeredGraphIndices.lower_bound(firstSkippedGraph);
                 skipped != iter->second.registeredGraphIndices.end() && *skipped < graphIndex;) {
                _enqueueGraphCompletionLocked(iter->second, *skipped);
                skipped = iter->second.registeredGraphIndices.erase(skipped);
            }
            if (iter->second.requestedUntil < graphIndex) {
                iter->second.requestedUntil = graphIndex - 1;
            }
        }
        iter->second.currentGraphIndex = graphIndex;
        _extendGraphWindowLocked(pipelineId, graphIndex + mConfig.graphPrefetchLookahead);
        graphId = iter->second.graphs[graphIndex].graphId;
        mPendingQnnGraphs[pipelineId] = graphId;
        mCondition.notify_all();
        mCondition.wait(lock, [this, &graphId]() {
            if (mStopRequested || mGraphPrefetchCancelled) {
                return true;
            }
            std::map<std::string, GraphState>::const_iterator graph = mGraphs.find(graphId);
            return graph != mGraphs.end() && graph->second.loadFinished;
        });
        if (mStopRequested || mGraphPrefetchCancelled) {
            mPendingQnnGraphs.erase(pipelineId);
            mCondition.notify_all();
            return false;
        }
        std::map<std::string, GraphState>::const_iterator graph = mGraphs.find(graphId);
        if (graph == mGraphs.end() || !graph->second.record.resident) {
            mPendingQnnGraphs.erase(pipelineId);
            mCondition.notify_all();
            return false;
        }
    }

    if (!enterStage(pipelineId, STAGE_QNN)) {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mPendingQnnGraphs.erase(pipelineId);
        }
        mCondition.notify_all();
        return false;
    }
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mPendingQnnGraphs.erase(pipelineId);
        std::map<int, PipelineGraphState>::iterator iter = mPipelineGraphs.find(pipelineId);
        if (iter == mPipelineGraphs.end() || iter->second.currentGraphIndex != graphIndex) {
            mCondition.notify_all();
            leaveStage(pipelineId);
            return false;
        }
        mQnnExecutionActive = true;
        iter->second.executingGraphIndex = graphIndex;
    }
    return true;
}

bool DualPipelineScheduler::leaveGraphStage(int pipelineId, int graphIndex) {
    bool tracked = false;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        std::map<int, PipelineGraphState>::iterator iter = mPipelineGraphs.find(pipelineId);
        if (iter != mPipelineGraphs.end() && iter->second.executingGraphIndex == graphIndex) {
            iter->second.executingGraphIndex = -1;
            tracked = true;
        }
    }
    if (!tracked) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mMutex);
        mQnnExecutionActive = false;
        mCondition.notify_all();
    }
    const bool stageSucceeded = leaveStage(pipelineId);
    {
        std::lock_guard<std::mutex> lock(mMutex);
        std::map<int, PipelineGraphState>::iterator iter = mPipelineGraphs.find(pipelineId);
        if (iter != mPipelineGraphs.end()) {
            _enqueueGraphCompletionLocked(iter->second, graphIndex);
            iter->second.completedGraphIndex = graphIndex;
        }
    }
    mCondition.notify_all();
    return stageSucceeded;
}

void DualPipelineScheduler::cancelGraphPrefetchWave() {
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mGraphPrefetchWaveActive) {
            return;
        }
        mGraphPrefetchCancelled = true;
    }
    mCondition.notify_all();
}

bool DualPipelineScheduler::finishGraphPrefetchWave() {
    bool succeeded = true;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mGraphPrefetchWaveActive) {
            return true;
        }
        succeeded = !mGraphPrefetchCancelled;
        for (std::map<int, PipelineGraphState>::iterator iter = mPipelineGraphs.begin(); iter != mPipelineGraphs.end(); ++iter) {
            PipelineGraphState& state = iter->second;
            succeeded = succeeded && state.completedGraphIndex == state.currentGraphIndex && state.executingGraphIndex < 0;
            _removePendingGraphLoadsLocked(iter->first, state.completedGraphIndex + 1, -1);
            for (std::set<int>::const_iterator graph = state.registeredGraphIndices.begin();
                 graph != state.registeredGraphIndices.end(); ++graph) {
                if (*graph > state.completedGraphIndex) {
                    _enqueueGraphCompletionLocked(state, *graph);
                }
            }
        }
        mPipelineGraphs.clear();
        mPendingQnnGraphs.clear();
        mGraphPrefetchWaveActive = false;
        mGraphPrefetchCancelled = false;
    }
    mCondition.notify_all();
    return succeeded;
}

DualPipelineScheduler::GraphWindowSnapshot DualPipelineScheduler::graphWindowSnapshot() const {
    std::lock_guard<std::mutex> lock(mMutex);
    GraphWindowSnapshot result;
    result.waveActive = mGraphPrefetchWaveActive;
    result.cancelled = mGraphPrefetchCancelled;
    for (std::deque<Task>::const_iterator task = mTasks.begin(); task != mTasks.end(); ++task) {
        if (task->type == TASK_GRAPH_LOAD) {
            ++result.pendingLoadTasks;
        }
    }
    for (std::map<int, PipelineGraphState>::const_iterator iter = mPipelineGraphs.begin(); iter != mPipelineGraphs.end(); ++iter) {
        PipelineGraphProgress progress;
        progress.pipelineId = iter->first;
        progress.currentGraphIndex = iter->second.currentGraphIndex;
        progress.requestedUntil = iter->second.requestedUntil;
        progress.completedGraphIndex = iter->second.completedGraphIndex;
        progress.executingGraphIndex = iter->second.executingGraphIndex;
        result.pipelines.push_back(progress);
    }
    return result;
}

bool DualPipelineScheduler::beginStageWave(const std::vector<int>& pipelineIds) {
    std::lock_guard<std::mutex> lock(mStageMutex);
    if (mStageWaveActive || pipelineIds.empty()) {
        return false;
    }
    mHostReadyStages.clear();
    mQnnReadyStages.clear();
    mStagePipelines.clear();
    mActiveStages.clear();
    for (size_t i = 0; i < pipelineIds.size(); ++i) {
        if (pipelineIds[i] < 0 || mStagePipelines.find(pipelineIds[i]) != mStagePipelines.end()) {
            mStagePipelines.clear();
            return false;
        }
        mStagePipelines.insert(pipelineIds[i]);
    }
    mActiveHostStages = 0;
    mActiveQnnStages = 0;
    mMaxConcurrentHostStages = 0;
    mMaxConcurrentQnnStages = 0;
    mCompletedStages = 0;
    mCompletedHostStages = 0;
    mCompletedQnnStages = 0;
    mHostQnnOverlapGrants = 0;
    mStageCancelled = false;
    mStageWaveActive = true;
    return true;
}

bool DualPipelineScheduler::enterStage(int pipelineId, StageBackend backend) {
    std::shared_ptr<StageWaiter> waiter(new StageWaiter);
    std::unique_lock<std::mutex> lock(mStageMutex);
    if (!mStageWaveActive || mStageCancelled ||
        mStagePipelines.find(pipelineId) == mStagePipelines.end() ||
        mActiveStages.find(pipelineId) != mActiveStages.end()) {
        return false;
    }
    waiter->pipelineId = pipelineId;
    waiter->backend = backend;
    if (backend == STAGE_QNN) {
        mQnnReadyStages.push_back(waiter);
    } else {
        mHostReadyStages.push_back(waiter);
    }
    _grantReadyStagesLocked();
    mStageCondition.wait(lock, [this, &waiter]() {
        return waiter->granted || mStageCancelled;
    });
    return waiter->granted && !mStageCancelled;
}

bool DualPipelineScheduler::leaveStage(int pipelineId) {
    std::lock_guard<std::mutex> lock(mStageMutex);
    std::map<int, StageBackend>::iterator iter = mActiveStages.find(pipelineId);
    if (iter == mActiveStages.end()) {
        return false;
    }
    if (iter->second == STAGE_QNN) {
        if (mActiveQnnStages > 0) {
            --mActiveQnnStages;
        }
        ++mCompletedQnnStages;
    } else if (mActiveHostStages > 0) {
        --mActiveHostStages;
        ++mCompletedHostStages;
    }
    mActiveStages.erase(iter);
    ++mCompletedStages;
    if (!mStageCancelled) {
        _grantReadyStagesLocked();
    }
    mStageCondition.notify_all();
    return !mStageCancelled;
}

void DualPipelineScheduler::cancelStageWave() {
    {
        std::lock_guard<std::mutex> lock(mStageMutex);
        if (!mStageWaveActive) {
            return;
        }
        mStageCancelled = true;
        mHostReadyStages.clear();
        mQnnReadyStages.clear();
    }
    mStageCondition.notify_all();
}

bool DualPipelineScheduler::finishStageWave() {
    std::lock_guard<std::mutex> lock(mStageMutex);
    if (!mStageWaveActive) {
        return false;
    }
    const bool succeeded = !mStageCancelled && mHostReadyStages.empty() && mQnnReadyStages.empty() &&
                           mActiveStages.empty();
    mHostReadyStages.clear();
    mQnnReadyStages.clear();
    mStagePipelines.clear();
    mActiveStages.clear();
    mActiveHostStages = 0;
    mActiveQnnStages = 0;
    mStageWaveActive = false;
    AcceptanceTrace::log("event=stage_summary host_completed=%llu qnn_completed=%llu overlap_grants=%llu max_host=%zu max_qnn=%zu cancelled=%d",
                         static_cast<unsigned long long>(mCompletedHostStages),
                         static_cast<unsigned long long>(mCompletedQnnStages),
                         static_cast<unsigned long long>(mHostQnnOverlapGrants),
                         mMaxConcurrentHostStages, mMaxConcurrentQnnStages,
                         mStageCancelled ? 1 : 0);
    return succeeded;
}

DualPipelineScheduler::StageSnapshot DualPipelineScheduler::stageSnapshot() const {
    std::lock_guard<std::mutex> lock(mStageMutex);
    StageSnapshot result;
    result.waveActive = mStageWaveActive;
    result.cancelled = mStageCancelled;
    result.hostReadyStages = mHostReadyStages.size();
    result.qnnReadyStages = mQnnReadyStages.size();
    result.activeHostStages = mActiveHostStages;
    result.activeQnnStages = mActiveQnnStages;
    result.maxConcurrentHostStages = mMaxConcurrentHostStages;
    result.maxConcurrentQnnStages = mMaxConcurrentQnnStages;
    result.completedStages = mCompletedStages;
    result.completedHostStages = mCompletedHostStages;
    result.completedQnnStages = mCompletedQnnStages;
    result.hostQnnOverlapGrants = mHostQnnOverlapGrants;
    return result;
}

void DualPipelineScheduler::_grantReadyStagesLocked() {
    if (mActiveHostStages == 0 && !mHostReadyStages.empty()) {
        std::shared_ptr<StageWaiter> waiter = mHostReadyStages.front();
        mHostReadyStages.pop_front();
        mActiveStages[waiter->pipelineId] = waiter->backend;
        ++mActiveHostStages;
        if (mActiveHostStages > mMaxConcurrentHostStages) {
            mMaxConcurrentHostStages = mActiveHostStages;
        }
        waiter->granted = true;
        if (mActiveQnnStages > 0) {
            ++mHostQnnOverlapGrants;
        }
    }
    if (mActiveQnnStages == 0 && !mQnnReadyStages.empty()) {
        std::shared_ptr<StageWaiter> waiter = mQnnReadyStages.front();
        mQnnReadyStages.pop_front();
        mActiveStages[waiter->pipelineId] = waiter->backend;
        ++mActiveQnnStages;
        if (mActiveQnnStages > mMaxConcurrentQnnStages) {
            mMaxConcurrentQnnStages = mActiveQnnStages;
        }
        waiter->granted = true;
        if (mActiveHostStages > 0) {
            ++mHostQnnOverlapGrants;
        }
    }
    mStageCondition.notify_all();
}

void DualPipelineScheduler::_enqueueTaskLocked(const Task& source) {
    Task task = source;
    task.enqueueSequence = mNextTaskSequence++;
    mTasks.push_back(task);
}

int DualPipelineScheduler::_graphLoadDistanceLocked(const Task& task) const {
    if (task.type != TASK_GRAPH_LOAD || task.graphRequest.pipelineId < 0 || task.graphRequest.graphIndex < 0) {
        return std::numeric_limits<int>::max();
    }
    std::map<int, PipelineGraphState>::const_iterator iter = mPipelineGraphs.find(task.graphRequest.pipelineId);
    if (iter == mPipelineGraphs.end()) {
        return std::numeric_limits<int>::max();
    }
    return std::max(0, task.graphRequest.graphIndex - iter->second.currentGraphIndex);
}

DualPipelineScheduler::Task DualPipelineScheduler::_popNextTaskLocked() {
    size_t candidate = 0;
    if (!mTasks.empty() && mTasks.front().type == TASK_GRAPH_LOAD) {
        const int firstDistance = _graphLoadDistanceLocked(mTasks[candidate]);
        int bestDistance = firstDistance;
        int bestPipeline = mTasks[candidate].graphRequest.pipelineId;
        uint64_t bestSequence = mTasks[candidate].enqueueSequence;
        for (size_t i = candidate + 1; i < mTasks.size() && mTasks[i].type == TASK_GRAPH_LOAD; ++i) {
            const int distance = _graphLoadDistanceLocked(mTasks[i]);
            const int pipelineId = mTasks[i].graphRequest.pipelineId;
            const uint64_t sequence = mTasks[i].enqueueSequence;
            if (distance < bestDistance ||
                (distance == bestDistance && pipelineId < bestPipeline) ||
                (distance == bestDistance && pipelineId == bestPipeline && sequence < bestSequence)) {
                candidate = i;
                bestDistance = distance;
                bestPipeline = pipelineId;
                bestSequence = sequence;
            }
        }
    }
    Task task = mTasks[candidate];
    mTasks.erase(mTasks.begin() + candidate);
    return task;
}

void DualPipelineScheduler::_enqueueGraphIndexLocked(int pipelineId, int graphIndex) {
    std::map<int, PipelineGraphState>::iterator iter = mPipelineGraphs.find(pipelineId);
    if (iter == mPipelineGraphs.end() || graphIndex < 0 || graphIndex >= static_cast<int>(iter->second.graphs.size())) {
        return;
    }
    GraphRequest request = iter->second.graphs[graphIndex];
    request.action = GRAPH_LOAD;
    request.pipelineId = pipelineId;
    request.graphIndex = graphIndex;
    request.ownerRequestIds = iter->second.ownerRequestIds;
    if (request.ownerRequestIds.empty()) {
        AcceptanceTrace::log("event=qnn_bucket lane=%d request_id=-1 request_scope=engine graph_index=%d shape_index=%d bucket=%d",
                             pipelineId, graphIndex, request.shapeIndex, request.bucketSize);
    } else {
        for (size_t ownerIndex = 0; ownerIndex < request.ownerRequestIds.size(); ++ownerIndex) {
            AcceptanceTrace::log("event=qnn_bucket lane=%d request_id=%d request_scope=engine graph_index=%d shape_index=%d bucket=%d",
                                 pipelineId, request.ownerRequestIds[ownerIndex], graphIndex,
                                 request.shapeIndex, request.bucketSize);
        }
    }
    Task task;
    task.type = TASK_GRAPH_LOAD;
    task.graphRequest = request;
    _enqueueTaskLocked(task);
    iter->second.requestedUntil = graphIndex;
}

void DualPipelineScheduler::_extendGraphWindowLocked(int pipelineId, int targetGraphIndex) {
    std::map<int, PipelineGraphState>::iterator iter = mPipelineGraphs.find(pipelineId);
    if (iter == mPipelineGraphs.end() || iter->second.graphs.empty()) {
        return;
    }
    const int lastGraphIndex = static_cast<int>(iter->second.graphs.size()) - 1;
    const int target = std::min(lastGraphIndex, std::max(0, targetGraphIndex));
    for (int graphIndex = iter->second.requestedUntil + 1; graphIndex <= target; ++graphIndex) {
        _enqueueGraphIndexLocked(pipelineId, graphIndex);
    }
}

void DualPipelineScheduler::_enqueueGraphCompletionLocked(PipelineGraphState& state,
                                                          int graphIndex) {
    if (graphIndex < 0 || graphIndex >= static_cast<int>(state.graphs.size())) {
        return;
    }
    const GraphRequest& graph = state.graphs[graphIndex];
    Task complete;
    complete.type = TASK_GRAPH_COMPLETE;
    complete.graphId = graph.graphId;
    _enqueueTaskLocked(complete);
}

void DualPipelineScheduler::_removePendingGraphLoadsLocked(int pipelineId, int firstGraphIndex, int endGraphIndex) {
    for (std::deque<Task>::iterator iter = mTasks.begin(); iter != mTasks.end();) {
        if (iter->type == TASK_GRAPH_LOAD && iter->graphRequest.pipelineId == pipelineId &&
            iter->graphRequest.graphIndex >= firstGraphIndex &&
            (endGraphIndex < 0 || iter->graphRequest.graphIndex < endGraphIndex)) {
            iter = mTasks.erase(iter);
        } else {
            ++iter;
        }
    }
}

void DualPipelineScheduler::_workerLoop() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            while (!mStopRequested && !_hasRunnableTaskLocked()) {
                mCondition.wait(lock);
            }
            // Drain queued graph work before releasing the resident cache.
            if (mStopRequested && mTasks.empty()) {
                break;
            }
            task = _popNextTaskLocked();
        }
        _processTask(task);
    }
}

bool DualPipelineScheduler::_hasRunnableTaskLocked() const {
    if (mTasks.empty() || mQnnExecutionActive) {
        return false;
    }
    for (std::map<int, std::string>::const_iterator iter = mPendingQnnGraphs.begin();
         iter != mPendingQnnGraphs.end(); ++iter) {
        std::map<std::string, GraphState>::const_iterator graph = mGraphs.find(iter->second);
        if (graph == mGraphs.end() || !graph->second.loadFinished) {
            return true;
        }
    }
    return mPendingQnnGraphs.empty();
}

void DualPipelineScheduler::_processTask(const Task& task) {
    if (task.type == TASK_GRAPH_COMPLETE) {
        _processGraphComplete(task.graphId);
        return;
    }
    _processGraphRequest(task.graphRequest);
}

void DualPipelineScheduler::_processGraphRequest(const GraphRequest& request) {
    Callbacks callbacks;
    std::vector<GraphRequest> evictReleaseRequests;
    bool shouldLoad = false;
    bool hasCapacity = true;

    {
        std::lock_guard<std::mutex> lock(mMutex);
        callbacks = mConfig.callbacks;
        GraphState& state = mGraphs[request.graphId];
        bool wasResident = state.record.resident;
        if (!wasResident) {
            hasCapacity = _planEvictionsLocked(request.graphId, &evictReleaseRequests);
        }
        state.lastRequest = request;
        state.record.graphId = request.graphId;
        state.record.pinned = state.record.pinned || request.draftGraph || request.pinResident;
        if (request.pipelineId >= 0 && request.graphIndex >= 0) {
            std::map<int, PipelineGraphState>::iterator pipeline = mPipelineGraphs.find(request.pipelineId);
            if (pipeline != mPipelineGraphs.end()) {
                pipeline->second.registeredGraphIndices.insert(request.graphIndex);
            }
        }
        state.record.lastUseSequence = mNextSequence++;
        shouldLoad = !wasResident && hasCapacity;
        if (shouldLoad) {
            ++state.record.activeUseCount;
            state.loadFinished = false;
        } else if (!hasCapacity) {
            state.record.pinned = false;
            state.record.activeUseCount = 0;
            state.record.resident = false;
            state.loadFinished = true;
        } else {
            ++state.record.activeUseCount;
            state.loadFinished = true;
        }
    }

    if (callbacks.onGraphRelease) {
        for (size_t i = 0; i < evictReleaseRequests.size(); ++i) {
            callbacks.onGraphRelease(evictReleaseRequests[i]);
        }
    }
    if (shouldLoad) {
        const bool loadSucceeded = !callbacks.onGraphLoad || callbacks.onGraphLoad(request);
        {
            std::lock_guard<std::mutex> lock(mMutex);
            std::map<std::string, GraphState>::iterator iter = mGraphs.find(request.graphId);
            if (iter != mGraphs.end()) {
                iter->second.loadFinished = true;
                iter->second.record.resident = loadSucceeded;
                if (!loadSucceeded) {
                    iter->second.record.pinned = false;
                    iter->second.record.activeUseCount = 0;
                    iter->second.record.lastUseSequence = mNextSequence++;
                }
            }
        }
        mCondition.notify_all();
    } else if (!hasCapacity) {
        mCondition.notify_all();
    }
}

void DualPipelineScheduler::_processGraphComplete(const std::string& graphId) {
    std::lock_guard<std::mutex> lock(mMutex);
    std::map<std::string, GraphState>::iterator iter = mGraphs.find(graphId);
    if (iter == mGraphs.end()) {
        return;
    }
    if (iter->second.record.activeUseCount > 0) {
        --iter->second.record.activeUseCount;
    }
    iter->second.record.lastUseSequence = mNextSequence++;
}

bool DualPipelineScheduler::_planEvictionsLocked(const std::string& incomingGraphId,
                                                 std::vector<GraphRequest>* releaseRequests) {
    size_t residentCount = _residentGraphCountLocked();
    while (residentCount >= mConfig.maxResidentGraphs) {
        std::map<std::string, GraphState>::iterator candidate = mGraphs.end();
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (std::map<std::string, GraphState>::iterator iter = mGraphs.begin(); iter != mGraphs.end(); ++iter) {
            const GraphRecord& record = iter->second.record;
            if (!record.resident || record.pinned || record.activeUseCount > 0 || iter->first == incomingGraphId) {
                continue;
            }
            if (record.lastUseSequence < oldest) {
                oldest = record.lastUseSequence;
                candidate = iter;
            }
        }
        if (candidate == mGraphs.end()) {
            break;
        }

        GraphRequest releaseRequest = candidate->second.lastRequest;
        releaseRequest.action = GRAPH_RELEASE;
        releaseRequest.reason = "maxResidentGraphs";
        releaseRequest.forceRelease = true;
        releaseRequest.unpinAfterRelease = true;

        candidate->second.record.resident = false;
        candidate->second.loadFinished = false;
        candidate->second.record.activeUseCount = 0;
        releaseRequests->push_back(releaseRequest);
        --residentCount;
    }
    return residentCount < mConfig.maxResidentGraphs;
}

size_t DualPipelineScheduler::_residentGraphCountLocked() const {
    size_t count = 0;
    for (std::map<std::string, GraphState>::const_iterator iter = mGraphs.begin(); iter != mGraphs.end(); ++iter) {
        if (iter->second.record.resident) {
            ++count;
        }
    }
    return count;
}

} // namespace Transformer
} // namespace MNN
