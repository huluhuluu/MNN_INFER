//
//  DualPipelineGraph.hpp
//  MNN
//

#ifndef DUALPIPELINEGRAPH_hpp
#define DUALPIPELINEGRAPH_hpp

#include "llm/DualPipelineScheduler.hpp"

#include <stdint.h>
#include <string>
#include <vector>

namespace MNN {
class Session;
namespace Transformer {

struct TensorShape {
    std::vector<int> dims;
};

struct QnnGraphInfo {
    std::string path;
    std::string baseDir;
    std::string npuDir;
    std::string relativePath;
    uint64_t offset;
    uint64_t size;
    std::vector<std::string> allGraphName;
    std::vector<int> bucketSizes;
    bool pin;
    bool draft;

    QnnGraphInfo();
};

struct OpInfo {
    std::string opName;
    std::string plannedBackend;
    bool actualBackendKnown;
    std::string actualBackend;
    std::vector<TensorShape> inputShapes;
    bool isPlugin;
    std::string pluginType;
    QnnGraphInfo qnn;

    OpInfo();
};

struct GraphSnapshot {
    std::vector<OpInfo> ops;
};

bool isQnnPluginOp(const OpInfo& op);
int selectQnnBucketSize(const QnnGraphInfo& qnn, int requestGroupSize);
int selectQnnCompatibleBucketSize(const GraphSnapshot& snapshot, int requiredSize);

GraphSnapshot buildGraphSnapshot(const MNN::Session* session,
                                 int pipelineIndex,
                                 const std::string& baseDir,
                                 const std::string& npuDir);

GraphSnapshot buildQnnGraphSnapshotFromModel(const std::string& modelPath,
                                             const std::string& baseDir,
                                             const std::string& npuDir);

GraphSnapshot mergeQnnGraphSnapshotsInExecutionOrder(const GraphSnapshot& executionSnapshot,
                                                     const GraphSnapshot& modelSnapshot);

std::vector<DualPipelineScheduler::GraphRequest> buildQnnGraphRequests(const GraphSnapshot& snapshot,
                                                                       int start,
                                                                       int maxK,
                                                                       int reqId);

} // namespace Transformer
} // namespace MNN

#endif /* DUALPIPELINEGRAPH_hpp */
