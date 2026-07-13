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
    std::string targetGraphName;
    int shapeIndex;
    bool pin;
    bool draft;

    QnnGraphInfo();
};

struct OpInfo {
    int layerIndex;
    int opIndex;
    std::string opName;
    int opTypeId;
    std::string opTypeName;
    std::string plannedBackend;
    bool actualBackendKnown;
    std::string actualBackend;
    std::vector<TensorShape> inputShapes;
    std::vector<TensorShape> outputShapes;
    bool isPlugin;
    std::string pluginType;
    QnnGraphInfo qnn;

    OpInfo();
};

struct GraphSnapshot {
    std::vector<OpInfo> ops;
};

bool isCpuOp(const OpInfo& op);
bool isOpenCLOp(const OpInfo& op);
bool isQnnPluginOp(const OpInfo& op);
int selectQnnBucketSize(const QnnGraphInfo& qnn, int requestGroupSize);

GraphSnapshot buildGraphSnapshot(const MNN::Session* session,
                                 int pipelineIndex,
                                 const std::string& baseDir,
                                 const std::string& npuDir);

DualPipelineScheduler::PrefetchWindow buildPrefetchWindow(const GraphSnapshot& snapshot,
                                                          int start,
                                                          int maxK,
                                                          int reqId);
std::vector<DualPipelineScheduler::GraphRequest> buildQnnGraphRequests(const GraphSnapshot& snapshot,
                                                                       int start,
                                                                       int maxK,
                                                                       int reqId,
                                                                       int requestGroupSize = 1);

} // namespace Transformer
} // namespace MNN

#endif /* DUALPIPELINEGRAPH_hpp */
