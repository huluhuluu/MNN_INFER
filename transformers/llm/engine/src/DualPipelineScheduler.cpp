//
//  DualPipelineScheduler.cpp
//  MNN
//

#include "llm/DualPipelineScheduler.hpp"

#include <limits>

namespace MNN {
namespace Transformer {

DualPipelineScheduler::PrefetchResizeRequest::PrefetchResizeRequest()
    : requestId(-1),
      layerIndex(-1),
      opIndex(-1) {
}

DualPipelineScheduler::PrefetchWindow::PrefetchWindow()
    : requestId(-1),
      startLayerIndex(-1),
      maxOpCount(0) {
}

DualPipelineScheduler::GraphRequest::GraphRequest()
    : action(GRAPH_LOAD),
      requestId(-1),
      offset(0),
      size(0),
      shapeIndex(-1),
      draftGraph(false),
      pinResident(false),
      forceRelease(false),
      unpinAfterRelease(false) {
}

DualPipelineScheduler::GraphRecord::GraphRecord()
    : resident(false),
      pinned(false),
      pendingRelease(false),
      activeUseCount(0),
      lastUseSequence(0) {
}

DualPipelineScheduler::Snapshot::Snapshot()
    : residentGraphs(0) {
}

DualPipelineScheduler::Config::Config()
    : maxResidentGraphs(5) {
}

DualPipelineScheduler::Task::Task()
    : type(TASK_PREFETCH_RESIZE) {
}

DualPipelineScheduler::GraphState::GraphState()
    : loadFinished(false) {
}

DualPipelineScheduler::DualPipelineScheduler()
    : mNextSequence(1),
      mRunning(false),
      mStopRequested(false) {
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
    }
}

bool DualPipelineScheduler::enqueueRequestGraph(const GraphRequest& request) {
    if (request.graphId.empty()) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mMutex);
        Task task;
        task.type = request.action == GRAPH_RELEASE ? TASK_GRAPH_RELEASE : TASK_GRAPH_LOAD;
        task.graphRequest = request;
        mTasks.push_back(task);
    }
    mCondition.notify_one();
    return true;
}

bool DualPipelineScheduler::enqueuePrefetchWindow(const PrefetchWindow& window) {
    {
        std::lock_guard<std::mutex> lock(mMutex);
        Task task;
        task.type = TASK_PREFETCH_RESIZE;
        task.prefetchWindow = window;
        mTasks.push_back(task);
    }
    mCondition.notify_one();
    return true;
}

bool DualPipelineScheduler::waitForGraphsReady(const std::vector<std::string>& graphIds) {
    if (graphIds.empty()) {
        return true;
    }
    std::unique_lock<std::mutex> lock(mMutex);
    mCondition.wait(lock, [this, &graphIds]() {
        if (mStopRequested) {
            return true;
        }
        for (size_t i = 0; i < graphIds.size(); ++i) {
            std::map<std::string, GraphState>::const_iterator iter = mGraphs.find(graphIds[i]);
            if (iter == mGraphs.end() || !iter->second.loadFinished) {
                return false;
            }
        }
        return true;
    });
    if (mStopRequested) {
        return false;
    }
    for (size_t i = 0; i < graphIds.size(); ++i) {
        std::map<std::string, GraphState>::const_iterator iter = mGraphs.find(graphIds[i]);
        if (iter == mGraphs.end() || !iter->second.record.resident) {
            return false;
        }
    }
    return true;
}

void DualPipelineScheduler::markExecutionComplete(const std::string& graphId) {
    if (graphId.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mMutex);
        // Queue completion behind graph load to preserve worker ordering.
        Task task;
        task.type = TASK_GRAPH_COMPLETE;
        task.graphId = graphId;
        mTasks.push_back(task);
    }
    mCondition.notify_one();
}

DualPipelineScheduler::Snapshot DualPipelineScheduler::snapshot() const {
    std::lock_guard<std::mutex> lock(mMutex);
    return _snapshotLocked();
}

void DualPipelineScheduler::_workerLoop() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            while (!mStopRequested && mTasks.empty()) {
                mCondition.wait(lock);
            }
            // Drain already queued releases/loads before exiting the resident worker.
            if (mStopRequested && mTasks.empty()) {
                break;
            }
            task = mTasks.front();
            mTasks.pop_front();
        }
        _processTask(task);
    }
}

void DualPipelineScheduler::_processTask(const Task& task) {
    if (task.type == TASK_PREFETCH_RESIZE) {
        _processPrefetchWindow(task.prefetchWindow);
        return;
    }
    if (task.type == TASK_GRAPH_COMPLETE) {
        _processGraphComplete(task.graphId);
        return;
    }
    _processGraphRequest(task.graphRequest);
}

void DualPipelineScheduler::_processPrefetchWindow(const PrefetchWindow& window) {
    Callbacks callbacks;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        callbacks = mConfig.callbacks;
    }

    if (!callbacks.onPrefetchResize) {
        return;
    }
    int count = 0;
    for (size_t i = 0; i < window.requests.size(); ++i) {
        if (window.maxOpCount > 0 && count >= window.maxOpCount) {
            break;
        }
        callbacks.onPrefetchResize(window.requests[i]);
        ++count;
    }
}

void DualPipelineScheduler::_processGraphRequest(const GraphRequest& request) {
    Callbacks callbacks;
    std::vector<GraphRequest> evictReleaseRequests;
    GraphRequest callbackRequest = request;
    bool shouldLoad = false;
    bool shouldRelease = false;

    {
        std::lock_guard<std::mutex> lock(mMutex);
        callbacks = mConfig.callbacks;
        if (request.action == GRAPH_RELEASE) {
            std::map<std::string, GraphState>::iterator iter = mGraphs.find(request.graphId);
            if (iter != mGraphs.end()) {
                const bool canReleasePinned = request.forceRelease || request.unpinAfterRelease;
                callbackRequest = _mergeReleaseRequestLocked(iter->second, request);
                if (iter->second.record.pinned && !canReleasePinned) {
                    iter->second.record.pendingRelease = false;
                    iter->second.record.lastUseSequence = mNextSequence++;
                } else if (request.forceRelease) {
                    iter->second.record.resident = false;
                    iter->second.record.pinned = false;
                    iter->second.record.activeUseCount = 0;
                    iter->second.record.pendingRelease = false;
                    iter->second.record.lastUseSequence = mNextSequence++;
                    shouldRelease = true;
                } else if (!iter->second.record.resident) {
                    iter->second.record.activeUseCount = 0;
                    iter->second.record.pendingRelease = false;
                } else if (iter->second.record.activeUseCount > 0) {
                    if (request.unpinAfterRelease) {
                        iter->second.record.pinned = false;
                    }
                    iter->second.record.pendingRelease = true;
                    iter->second.pendingReleaseRequest = callbackRequest;
                } else {
                    iter->second.record.resident = false;
                    if (request.unpinAfterRelease) {
                        iter->second.record.pinned = false;
                    }
                    iter->second.record.pendingRelease = false;
                    shouldRelease = true;
                }
            }
        } else {
            GraphState& state = mGraphs[request.graphId];
            bool wasResident = state.record.resident;
            if (!wasResident) {
                _planEvictionsLocked(request.graphId, &evictReleaseRequests);
            }
            state.lastRequest = request;
            state.record.graphId = request.graphId;
            state.record.pinned = state.record.pinned || request.draftGraph || request.pinResident;
            // The same resident QNN graph can be shared by two requests.
            ++state.record.activeUseCount;
            state.record.pendingRelease = false;
            state.record.lastUseSequence = mNextSequence++;
            shouldLoad = !wasResident;
            if (shouldLoad) {
                state.loadFinished = false;
            } else {
                state.loadFinished = true;
            }
        }
    }

    if (callbacks.onGraphRelease) {
        for (size_t i = 0; i < evictReleaseRequests.size(); ++i) {
            callbacks.onGraphRelease(evictReleaseRequests[i]);
        }
    }
    if (shouldRelease && callbacks.onGraphRelease) {
        callbacks.onGraphRelease(callbackRequest);
    }
    if (shouldLoad) {
        const bool loadSucceeded = !callbacks.onGraphLoad || callbacks.onGraphLoad(callbackRequest);
        {
            std::lock_guard<std::mutex> lock(mMutex);
            std::map<std::string, GraphState>::iterator iter = mGraphs.find(callbackRequest.graphId);
            if (iter != mGraphs.end()) {
                iter->second.loadFinished = true;
                iter->second.record.resident = loadSucceeded;
                if (!loadSucceeded) {
                    iter->second.record.pinned = false;
                    iter->second.record.activeUseCount = 0;
                    iter->second.record.pendingRelease = false;
                    iter->second.record.lastUseSequence = mNextSequence++;
                }
            }
        }
        mCondition.notify_all();
    }
}

void DualPipelineScheduler::_processGraphComplete(const std::string& graphId) {
    Callbacks callbacks;
    GraphRequest releaseRequest;
    bool shouldRelease = false;

    {
        std::lock_guard<std::mutex> lock(mMutex);
        callbacks = mConfig.callbacks;
        std::map<std::string, GraphState>::iterator iter = mGraphs.find(graphId);
        if (iter == mGraphs.end()) {
            return;
        }
        // Release only after all requests using this graph have reported completion.
        if (iter->second.record.activeUseCount > 0) {
            --iter->second.record.activeUseCount;
        }
        iter->second.record.lastUseSequence = mNextSequence++;
        if (iter->second.record.activeUseCount == 0 && iter->second.record.pendingRelease && !iter->second.record.pinned) {
            iter->second.record.resident = false;
            iter->second.record.pendingRelease = false;
            releaseRequest = _mergeReleaseRequestLocked(iter->second, iter->second.pendingReleaseRequest);
            shouldRelease = true;
        }
    }

    if (shouldRelease && callbacks.onGraphRelease) {
        callbacks.onGraphRelease(releaseRequest);
    }
}

void DualPipelineScheduler::_planEvictionsLocked(const std::string& incomingGraphId,
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
        candidate->second.record.pendingRelease = false;
        candidate->second.record.activeUseCount = 0;
        releaseRequests->push_back(releaseRequest);
        --residentCount;
    }
}

DualPipelineScheduler::GraphRequest DualPipelineScheduler::_mergeReleaseRequestLocked(const GraphState& state,
                                                                                      const GraphRequest& request) const {
    GraphRequest merged = state.lastRequest.graphId.empty() ? request : state.lastRequest;
    merged.action = GRAPH_RELEASE;
    merged.requestId = request.requestId;
    if (!request.graphId.empty()) {
        merged.graphId = request.graphId;
    }
    if (!request.graphPath.empty()) {
        merged.graphPath = request.graphPath;
    }
    if (!request.baseDir.empty()) {
        merged.baseDir = request.baseDir;
    }
    if (!request.npuDir.empty()) {
        merged.npuDir = request.npuDir;
    }
    if (!request.relativePath.empty()) {
        merged.relativePath = request.relativePath;
    }
    if (request.offset != 0) {
        merged.offset = request.offset;
    }
    if (request.size != 0) {
        merged.size = request.size;
    }
    if (!request.allGraphName.empty()) {
        merged.allGraphName = request.allGraphName;
    }
    if (!request.targetGraphName.empty()) {
        merged.targetGraphName = request.targetGraphName;
    }
    if (request.shapeIndex >= 0) {
        merged.shapeIndex = request.shapeIndex;
    }
    merged.reason = request.reason;
    merged.forceRelease = request.forceRelease;
    merged.unpinAfterRelease = request.unpinAfterRelease;
    return merged;
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

DualPipelineScheduler::Snapshot DualPipelineScheduler::_snapshotLocked() const {
    Snapshot snapshot;
    snapshot.residentGraphs = _residentGraphCountLocked();
    for (std::map<std::string, GraphState>::const_iterator iter = mGraphs.begin(); iter != mGraphs.end(); ++iter) {
        snapshot.graphs.push_back(iter->second.record);
    }
    return snapshot;
}

} // namespace Transformer
} // namespace MNN
