//
//  DualPipelineScheduler.hpp
//  MNN
//

#ifndef DUALPIPELINESCHEDULER_hpp
#define DUALPIPELINESCHEDULER_hpp

#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <string>
#include <thread>
#include <vector>

namespace MNN {
namespace Transformer {

class DualPipelineScheduler {
public:
    enum StageBackend {
        STAGE_HOST = 0,
        STAGE_QNN = 1
    };

    struct StageSnapshot {
        bool waveActive;
        bool cancelled;
        size_t hostReadyStages;
        size_t qnnReadyStages;
        size_t activeHostStages;
        size_t activeQnnStages;
        size_t maxConcurrentHostStages;
        size_t maxConcurrentQnnStages;
        uint64_t completedStages;
        uint64_t completedHostStages;
        uint64_t completedQnnStages;
        uint64_t hostQnnOverlapGrants;

        StageSnapshot();
    };

    enum GraphAction {
        GRAPH_LOAD = 0,
        GRAPH_RELEASE = 1
    };

    enum TaskType {
        TASK_GRAPH_LOAD = 0,
        TASK_GRAPH_COMPLETE = 1
    };

    struct GraphRequest {
        GraphAction action;
        int pipelineId;
        int graphIndex;
        int shapeIndex;
        int bucketSize;
        std::vector<int> ownerRequestIds;
        std::string graphId;
        std::string graphPath;
        uint64_t offset;
        uint64_t size;
        // Resident device bytes. If zero, the scheduler treats the graph size
        // as unknown and preserves legacy count-only admission behavior.
        uint64_t residentBytes;
        std::vector<std::string> allGraphName;
        std::string reason;
        bool draftGraph;
        bool pinResident;
        bool forceRelease;
        bool unpinAfterRelease;

        GraphRequest();
    };

    struct GraphRecord {
        std::string graphId;
        bool resident;
        bool pinned;
        int activeUseCount;
        uint64_t lastUseSequence;
        uint64_t residentBytes;

        GraphRecord();
    };

    struct PipelineGraphWave {
        int pipelineId;
        std::vector<int> ownerRequestIds;
        std::vector<GraphRequest> graphs;

        PipelineGraphWave();
    };

    struct PipelineGraphProgress {
        int pipelineId;
        int currentGraphIndex;
        int requestedUntil;
        int completedGraphIndex;
        int executingGraphIndex;

        PipelineGraphProgress();
    };

    struct GraphWindowSnapshot {
        bool waveActive;
        bool cancelled;
        size_t pendingLoadTasks;
        std::vector<PipelineGraphProgress> pipelines;

        GraphWindowSnapshot();
    };

    struct MemorySnapshot {
        size_t budgetBytes;
        size_t pinnedBytes;
        size_t kvCacheBytes;
        size_t workspaceBytes;
        size_t residentGraphBytes;
        size_t residentGraphCount;
        size_t accountedBytes;

        MemorySnapshot();
    };

    struct Callbacks {
        std::function<bool(const GraphRequest&)> onGraphLoad;
        std::function<void(const GraphRequest&)> onGraphRelease;
    };

    struct Config {
        size_t maxResidentGraphs;
        int graphPrefetchLookahead;
        // Zero keeps the pre-budget behavior for deployments without a
        // reliable device-memory budget.
        size_t memoryBudgetBytes;
        size_t pinnedMemoryBytes;
        size_t kvCacheBytes;
        size_t workspaceBytes;
        Callbacks callbacks;

        Config();
    };

    DualPipelineScheduler();
    ~DualPipelineScheduler();

    bool configure(const Config& config);
    bool start();
    void stop();

    bool beginGraphPrefetchWave(const std::vector<PipelineGraphWave>& waves);
    bool enterGraphStage(int pipelineId, int graphIndex);
    bool leaveGraphStage(int pipelineId, int graphIndex);
    void cancelGraphPrefetchWave();
    bool finishGraphPrefetchWave();
    GraphWindowSnapshot graphWindowSnapshot() const;
    MemorySnapshot memorySnapshot() const;
    // Update non-graph reservations before admitting the next graph wave.
    // Values are owned by the caller's runtime manager and are not evicted by
    // this scheduler.
    bool setMemoryUsage(size_t kvCacheBytes, size_t workspaceBytes);
    bool releaseResidentGraph(const std::string& graphId);

    bool beginStageWave(const std::vector<int>& pipelineIds);
    bool enterStage(int pipelineId, StageBackend backend);
    bool leaveStage(int pipelineId);
    void cancelStageWave();
    bool finishStageWave();
    StageSnapshot stageSnapshot() const;

private:
    struct Task {
        TaskType type;
        GraphRequest graphRequest;
        std::string graphId;
        uint64_t enqueueSequence;

        Task();
    };

    struct GraphState {
        GraphRecord record;
        GraphRequest lastRequest;
        bool loadFinished;

        GraphState();
    };

    struct StageWaiter {
        int pipelineId;
        StageBackend backend;
        bool granted;

        StageWaiter();
    };

    struct PipelineGraphState {
        std::vector<GraphRequest> graphs;
        std::vector<int> ownerRequestIds;
        std::set<int> registeredGraphIndices;
        int currentGraphIndex;
        int requestedUntil;
        int completedGraphIndex;
        int executingGraphIndex;

        PipelineGraphState();
    };

    void _workerLoop();
    bool _hasRunnableTaskLocked() const;
    Task _popNextTaskLocked();
    int _graphLoadDistanceLocked(const Task& task) const;
    void _enqueueTaskLocked(const Task& task);
    void _extendGraphWindowLocked(int pipelineId, int targetGraphIndex);
    void _enqueueGraphIndexLocked(int pipelineId, int graphIndex);
    void _enqueueGraphCompletionLocked(PipelineGraphState& state, int graphIndex);
    void _removePendingGraphLoadsLocked(int pipelineId, int firstGraphIndex, int endGraphIndex);
    void _processTask(const Task& task);
    void _processGraphRequest(const GraphRequest& request);
    void _processGraphComplete(const std::string& graphId);
    bool _planEvictionsLocked(const std::string& incomingGraphId,
                              uint64_t incomingGraphBytes,
                              std::vector<GraphRequest>* releaseRequests);
    size_t _residentGraphCountLocked() const;
    size_t _residentGraphBytesLocked() const;
    size_t _accountedBytesLocked() const;
    void _grantReadyStagesLocked();

    mutable std::mutex mMutex;
    std::condition_variable mCondition;
    std::deque<Task> mTasks;
    std::thread mWorker;
    Config mConfig;
    std::map<std::string, GraphState> mGraphs;
    std::map<int, PipelineGraphState> mPipelineGraphs;
    std::map<int, std::string> mPendingQnnGraphs;
    uint64_t mNextSequence;
    uint64_t mNextTaskSequence;
    bool mRunning;
    bool mStopRequested;
    bool mGraphPrefetchWaveActive;
    bool mGraphPrefetchCancelled;
    bool mQnnExecutionActive;
    size_t mKvCacheBytes;
    size_t mWorkspaceBytes;

    mutable std::mutex mStageMutex;
    std::condition_variable mStageCondition;
    std::deque<std::shared_ptr<StageWaiter>> mHostReadyStages;
    std::deque<std::shared_ptr<StageWaiter>> mQnnReadyStages;
    std::set<int> mStagePipelines;
    std::map<int, StageBackend> mActiveStages;
    size_t mActiveHostStages;
    size_t mActiveQnnStages;
    size_t mMaxConcurrentHostStages;
    size_t mMaxConcurrentQnnStages;
    uint64_t mCompletedStages;
    uint64_t mCompletedHostStages;
    uint64_t mCompletedQnnStages;
    uint64_t mHostQnnOverlapGrants;
    bool mStageWaveActive;
    bool mStageCancelled;
};

} // namespace Transformer
} // namespace MNN

#endif /* DUALPIPELINESCHEDULER_hpp */
