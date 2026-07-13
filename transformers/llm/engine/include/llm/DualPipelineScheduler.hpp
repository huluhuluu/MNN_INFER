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

    struct GraphRecord {
        std::string graphId;
        bool resident;
        bool pinned;
        bool pendingRelease;
        int activeUseCount;
        uint64_t lastUseSequence;

        GraphRecord();
    };

    struct Snapshot {
        size_t residentGraphs;
        std::vector<GraphRecord> graphs;

        Snapshot();
    };

    struct Callbacks {
        std::function<void(const PrefetchResizeRequest&)> onPrefetchResize;
        std::function<bool(const GraphRequest&)> onGraphLoad;
        std::function<void(const GraphRequest&)> onGraphRelease;
    };

    struct Config {
        size_t maxResidentGraphs;
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
    bool waitForGraphsReady(const std::vector<std::string>& graphIds);
    void markExecutionComplete(const std::string& graphId);

    Snapshot snapshot() const;

private:
    struct Task {
        TaskType type;
        PrefetchWindow prefetchWindow;
        GraphRequest graphRequest;
        std::string graphId;

        Task();
    };

    struct GraphState {
        GraphRecord record;
        GraphRequest lastRequest;
        GraphRequest pendingReleaseRequest;
        bool loadFinished;

        GraphState();
    };

    void _workerLoop();
    void _processTask(const Task& task);
    void _processPrefetchWindow(const PrefetchWindow& window);
    void _processGraphRequest(const GraphRequest& request);
    void _processGraphComplete(const std::string& graphId);
    void _planEvictionsLocked(const std::string& incomingGraphId,
                              std::vector<GraphRequest>* releaseRequests);
    GraphRequest _mergeReleaseRequestLocked(const GraphState& state, const GraphRequest& request) const;
    size_t _residentGraphCountLocked() const;
    Snapshot _snapshotLocked() const;

    mutable std::mutex mMutex;
    std::condition_variable mCondition;
    std::deque<Task> mTasks;
    std::thread mWorker;
    Config mConfig;
    std::map<std::string, GraphState> mGraphs;
    uint64_t mNextSequence;
    bool mRunning;
    bool mStopRequested;
};

} // namespace Transformer
} // namespace MNN

#endif /* DUALPIPELINESCHEDULER_hpp */
