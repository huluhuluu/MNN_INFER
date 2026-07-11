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
#include <mutex>
#include <stdint.h>
#include <string>
#include <thread>
#include <vector>

namespace MNN {
namespace Transformer {

class DualPipelineScheduler {
public:
    enum GraphAction {
        GRAPH_LOAD = 0,
        GRAPH_RELEASE = 1
    };

    enum TaskType {
        TASK_PREFETCH_RESIZE = 0,
        TASK_GRAPH_LOAD = 1,
        TASK_GRAPH_RELEASE = 2,
        TASK_GRAPH_COMPLETE = 3
    };

    struct TensorShape {
        std::vector<int> dims;
    };

    struct PrefetchResizeRequest {
        int requestId;
        int layerIndex;
        int opIndex;
        std::string opName;
        std::string backend;
        std::vector<TensorShape> inputShapes;
        std::vector<TensorShape> outputShapes;

        PrefetchResizeRequest();
    };

    struct PrefetchWindow {
        int requestId;
        int startLayerIndex;
        int maxOpCount;
        std::vector<PrefetchResizeRequest> requests;

        PrefetchWindow();
    };

    struct GraphRequest {
        GraphAction action;
        int requestId;
        std::string graphId;
        std::string graphPath;
        std::string baseDir;
        std::string npuDir;
        std::string relativePath;
        uint64_t offset;
        uint64_t size;
        std::vector<std::string> allGraphName;
        std::string targetGraphName;
        int shapeIndex;
        std::string reason;
        bool draftGraph;
        bool pinResident;
        bool forceRelease;
        bool unpinAfterRelease;

        GraphRequest();
    };

    struct EvictPlan {
        std::string graphId;
        std::string reason;
        uint64_t sequence;

        EvictPlan();
    };

    struct GraphRecord {
        std::string graphId;
        bool resident;
        bool pinned;
        bool inUse;
        bool pendingRelease;
        int activeUseCount;
        uint64_t lastUseSequence;

        GraphRecord();
    };

    struct Snapshot {
        bool running;
        bool stopRequested;
        size_t queuedTasks;
        size_t residentGraphs;
        size_t maxResidentGraphs;
        uint64_t scheduledTasks;
        uint64_t completedExecutions;
        uint64_t evictionBlockedCount;
        std::vector<GraphRecord> graphs;
        std::vector<EvictPlan> evictPlans;

        Snapshot();
    };

    struct Status {
        bool running;
        size_t queuedTasks;
        size_t residentGraphs;
        size_t maxResidentGraphs;
        uint64_t scheduledTasks;
        uint64_t completedExecutions;
        uint64_t evictionBlockedCount;

        Status();
    };

    struct Callbacks {
        std::function<void(const PrefetchResizeRequest&)> onPrefetchResize;
        std::function<void(const GraphRequest&)> onGraphLoad;
        std::function<void(const GraphRequest&)> onGraphRelease;
        std::function<void(const EvictPlan&)> onEvictPlan;
    };

    struct Config {
        size_t maxResidentGraphs;
        size_t maxEvictPlanHistory;
        Callbacks callbacks;

        Config();
    };

    DualPipelineScheduler();
    ~DualPipelineScheduler();

    bool configure(const Config& config);
    bool start();
    void stop();

    bool enqueueRequestGraph(const GraphRequest& request);
    bool enqueuePrefetchWindow(const PrefetchWindow& window);
    void markExecutionComplete(const std::string& graphId);

    Snapshot snapshot() const;
    Status status() const;

private:
    struct Task {
        TaskType type;
        PrefetchWindow prefetchWindow;
        GraphRequest graphRequest;
        std::string graphId;
        uint64_t sequence;

        Task();
    };

    struct GraphState {
        GraphRecord record;
        GraphRequest lastRequest;
        GraphRequest pendingReleaseRequest;

        GraphState();
    };

    void _workerLoop();
    void _processTask(const Task& task);
    void _processPrefetchWindow(const PrefetchWindow& window);
    void _processGraphRequest(const GraphRequest& request);
    void _processGraphComplete(const std::string& graphId);
    void _planEvictionsLocked(const std::string& incomingGraphId,
                              std::vector<EvictPlan>* plans,
                              std::vector<GraphRequest>* releaseRequests);
    GraphRequest _mergeReleaseRequestLocked(const GraphState& state, const GraphRequest& request) const;
    size_t _residentGraphCountLocked() const;
    void _rememberEvictPlanLocked(const EvictPlan& plan);
    Snapshot _snapshotLocked() const;

    mutable std::mutex mMutex;
    std::condition_variable mCondition;
    std::deque<Task> mTasks;
    std::thread mWorker;
    Config mConfig;
    std::map<std::string, GraphState> mGraphs;
    std::vector<EvictPlan> mEvictPlans;
    uint64_t mNextSequence;
    uint64_t mScheduledTasks;
    uint64_t mCompletedExecutions;
    uint64_t mEvictionBlockedCount;
    bool mConfigured;
    bool mRunning;
    bool mStopRequested;
};

} // namespace Transformer
} // namespace MNN

#endif /* DUALPIPELINESCHEDULER_hpp */
