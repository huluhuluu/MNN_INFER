//
//  DualPipelineGraph.cpp
//  MNN
//

#include "llm/DualPipelineGraph.hpp"

#include "core/Backend.hpp"
#include "core/Command.hpp"
#include "core/Execution.hpp"
#include "core/Schedule.hpp"
#include "core/Session.hpp"
#include "MNN_generated.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>

namespace MNN {
namespace Transformer {

namespace {

std::string lowerString(const std::string& value) {
    std::string result = value;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        if (c >= 'A' && c <= 'Z') {
            return static_cast<char>(c - 'A' + 'a');
        }
        return static_cast<char>(c);
    });
    return result;
}

bool containsToken(const std::string& value, const std::string& token) {
    return lowerString(value).find(token) != std::string::npos;
}

std::string effectiveBackend(const OpInfo& op) {
    if (op.actualBackendKnown && !op.actualBackend.empty()) {
        return op.actualBackend;
    }
    return op.plannedBackend;
}

std::string forwardTypeName(MNNForwardType type) {
    switch (type) {
        case MNN_FORWARD_CPU:
            return "CPU";
        case MNN_FORWARD_OPENCL:
            return "OpenCL";
        case MNN_FORWARD_CUDA:
            return "CUDA";
        case MNN_FORWARD_METAL:
            return "Metal";
        case MNN_FORWARD_OPENGL:
            return "OpenGL";
        case MNN_FORWARD_VULKAN:
            return "Vulkan";
        case MNN_FORWARD_NN:
            return "NN";
        case MNN_CONVERT_QNN:
            return "QNN";
        case MNN_FORWARD_AUTO:
            return "Auto";
        default:
            return "ForwardType_" + std::to_string(static_cast<int>(type));
    }
}

TensorShape copyTensorShape(const Tensor* tensor) {
    TensorShape shape;
    if (tensor != nullptr) {
        shape.dims = tensor->shape();
    }
    return shape;
}

std::vector<TensorShape> copyTensorShapes(const std::vector<Tensor*>& tensors) {
    std::vector<TensorShape> shapes;
    shapes.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        shapes.push_back(copyTensorShape(tensors[i]));
    }
    return shapes;
}

const Attribute* findAttr(const Plugin* plugin, const char* key) {
    if (plugin == nullptr || plugin->attr() == nullptr || key == nullptr) {
        return nullptr;
    }
    const auto attrs = plugin->attr();
    for (flatbuffers::uoffset_t i = 0; i < attrs->size(); ++i) {
        const Attribute* attr = attrs->Get(i);
        if (attr != nullptr && attr->key() != nullptr && attr->key()->str() == key) {
            return attr;
        }
    }
    return nullptr;
}

std::string attrString(const Plugin* plugin, const char* key) {
    const Attribute* attr = findAttr(plugin, key);
    if (attr == nullptr || attr->s() == nullptr) {
        return "";
    }
    return attr->s()->str();
}

bool attrBool(const Plugin* plugin, const char* key, bool defaultValue) {
    const Attribute* attr = findAttr(plugin, key);
    if (attr == nullptr) {
        return defaultValue;
    }
    return attr->b();
}

std::vector<std::string> attrStringList(const Plugin* plugin, const char* key) {
    std::vector<std::string> result;
    const Attribute* attr = findAttr(plugin, key);
    if (attr == nullptr || attr->list() == nullptr || attr->list()->s() == nullptr) {
        return result;
    }
    const auto values = attr->list()->s();
    result.reserve(values->size());
    for (flatbuffers::uoffset_t i = 0; i < values->size(); ++i) {
        result.push_back(values->GetAsString(i)->str());
    }
    return result;
}

std::vector<int> attrIntList(const Plugin* plugin, const char* key) {
    std::vector<int> result;
    const Attribute* attr = findAttr(plugin, key);
    if (attr == nullptr || attr->list() == nullptr || attr->list()->i() == nullptr) {
        return result;
    }
    const auto values = attr->list()->i();
    result.reserve(values->size());
    for (flatbuffers::uoffset_t i = 0; i < values->size(); ++i) {
        result.push_back(values->Get(i));
    }
    return result;
}

std::vector<int> inferQnnBucketSizes(const Plugin* plugin,
                                     const std::vector<TensorShape>& inputShapes,
                                     const std::vector<std::string>& graphNames) {
    std::vector<int> result;
    if (graphNames.empty()) {
        return result;
    }
    const std::vector<int> allInputShape = attrIntList(plugin, "allInputShape");
    size_t inputShapeLength = 0;
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        if (inputShapes[i].dims.empty()) {
            inputShapeLength = 0;
            break;
        }
        inputShapeLength += inputShapes[i].dims.size();
    }
    if (inputShapeLength == 0 || allInputShape.size() != inputShapeLength * graphNames.size()) {
        if (allInputShape.size() % graphNames.size() != 0) {
            return result;
        }
        inputShapeLength = allInputShape.size() / graphNames.size();
    }
    if (inputShapeLength == 0) {
        return result;
    }

    int bestRange = 0;
    for (size_t dimIndex = 0; dimIndex < inputShapeLength; ++dimIndex) {
        std::vector<int> candidate;
        candidate.reserve(graphNames.size());
        int minValue = std::numeric_limits<int>::max();
        int maxValue = 0;
        bool valid = true;
        for (size_t variant = 0; variant < graphNames.size(); ++variant) {
            const int value = allInputShape[variant * inputShapeLength + dimIndex];
            if (value <= 0) {
                valid = false;
                break;
            }
            candidate.push_back(value);
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
        }
        if (!valid || maxValue == minValue) {
            continue;
        }
        const int range = maxValue - minValue;
        if (range > bestRange) {
            bestRange = range;
            result = candidate;
        }
    }
    return result;
}

int selectQnnShapeIndex(const QnnGraphInfo& qnn, int requestGroupSize) {
    if (qnn.bucketSizes.empty() || qnn.bucketSizes.size() != qnn.allGraphName.size()) {
        return -1;
    }
    const int groupSize = requestGroupSize > 0 ? requestGroupSize : 1;
    int selectedIndex = -1;
    int selectedBucket = std::numeric_limits<int>::max();
    for (size_t i = 0; i < qnn.bucketSizes.size(); ++i) {
        const int bucket = qnn.bucketSizes[i];
        if (bucket <= 0) {
            continue;
        }
        if (bucket >= groupSize && bucket < selectedBucket) {
            selectedBucket = bucket;
            selectedIndex = static_cast<int>(i);
        }
    }
    if (selectedIndex >= 0) {
        return selectedIndex;
    }
    return -1;
}

std::string qnnGraphResourceId(const OpInfo& op) {
    std::string graphId = !op.qnn.path.empty() ? op.qnn.path : op.opName;
    graphId += "#" + std::to_string(op.qnn.offset) + "#" + std::to_string(op.qnn.size);
    for (size_t i = 0; i < op.qnn.allGraphName.size(); ++i) {
        graphId += "#" + op.qnn.allGraphName[i];
    }
    return graphId;
}

std::string joinPath(const std::string& dir, const std::string& path);

std::string qnnGraphPathForShape(const OpInfo& op, int shapeIndex) {
    if (shapeIndex < 0 || shapeIndex >= static_cast<int>(op.qnn.graphPaths.size())) {
        return "";
    }
    const std::string& path = op.qnn.graphPaths[shapeIndex];
    if (path.empty() || path[0] == '/' || path[0] == '\\') {
        return path;
    }
    const std::string graphDir = !op.qnn.npuDir.empty() ? op.qnn.npuDir : op.qnn.baseDir;
    return joinPath(graphDir, path);
}

std::string qnnGraphResourceIdForShape(const OpInfo& op, int shapeIndex) {
    const std::string path = qnnGraphPathForShape(op, shapeIndex);
    const std::string graphName = shapeIndex >= 0 && shapeIndex < static_cast<int>(op.qnn.allGraphName.size())
        ? op.qnn.allGraphName[shapeIndex] : "";
    std::string graphId = !path.empty() ? path : op.opName;
    graphId += "#0#0#" + graphName;
    return graphId;
}

uint64_t attrUint64FromIntPair(const Plugin* plugin, const char* key) {
    const Attribute* attr = findAttr(plugin, key);
    if (attr == nullptr || attr->list() == nullptr || attr->list()->i() == nullptr || attr->list()->i()->size() != 2) {
        return 0;
    }
    // QNN converter stores 64-bit offset/size as two int32 values: low32, high32.
    const int* data = attr->list()->i()->data();
    uint32_t low = 0;
    uint32_t high = 0;
    std::memcpy(&low, &data[0], sizeof(uint32_t));
    std::memcpy(&high, &data[1], sizeof(uint32_t));
    return (static_cast<uint64_t>(high) << 32) | static_cast<uint64_t>(low);
}

std::string joinPath(const std::string& dir, const std::string& path) {
    if (path.empty()) {
        return "";
    }
    if (path.size() > 0 && (path[0] == '/' || path[0] == '\\')) {
        return path;
    }
    if (dir.empty()) {
        return path;
    }
    const char tail = dir[dir.size() - 1];
    if (tail == '/' || tail == '\\') {
        return dir + path;
    }
    return dir + "/" + path;
}

void fillQnnInfo(const Op* op, const std::string& baseDir, const std::string& npuDir, OpInfo* info) {
    if (op == nullptr || info == nullptr || op->type() != OpType_Plugin) {
        return;
    }
    const Plugin* plugin = op->main_as_Plugin();
    if (plugin == nullptr) {
        return;
    }
    info->isPlugin = true;
    if (plugin->type() != nullptr) {
        info->pluginType = plugin->type()->str();
    }
    const bool hasQnnGraphAttrs = findAttr(plugin, "path") != nullptr && findAttr(plugin, "allGraphName") != nullptr;
    const bool hasQnnTypeHint = containsToken(info->pluginType, "qnn") || containsToken(effectiveBackend(*info), "qnn");
    if (!hasQnnTypeHint && !hasQnnGraphAttrs) {
        return;
    }

    const std::string relativePath = attrString(plugin, "path");
    const std::string graphDir = !npuDir.empty() ? npuDir : baseDir;
    // Keep both original attrs and resolved path for future backend callbacks.
    info->qnn.baseDir = baseDir;
    info->qnn.npuDir = npuDir;
    info->qnn.relativePath = relativePath;
    info->qnn.path = joinPath(graphDir, relativePath);
    info->qnn.offset = attrUint64FromIntPair(plugin, "offset");
    info->qnn.size = attrUint64FromIntPair(plugin, "size");
    info->qnn.allGraphName = attrStringList(plugin, "allGraphName");
    info->qnn.graphPaths = attrStringList(plugin, "allGraphPath");
    if (info->qnn.graphPaths.size() != info->qnn.allGraphName.size()) {
        info->qnn.graphPaths.clear();
    }
    info->qnn.bucketSizes = inferQnnBucketSizes(plugin, info->inputShapes, info->qnn.allGraphName);
    info->qnn.draft = attrBool(plugin, "draftGraph", containsToken(info->opName, "draft"));
    info->qnn.pin = attrBool(plugin, "pinResident", info->qnn.draft);
}

void fillActualBackend(const Schedule::OpCacheInfo& cacheInfo, OpInfo* info) {
    if (info == nullptr) {
        return;
    }
    const std::vector<std::shared_ptr<Command>>& commands = cacheInfo.executeBuffer.command;
    for (size_t i = 0; i < commands.size(); ++i) {
        const std::shared_ptr<Command>& command = commands[i];
        if (!command || !command->execution || command->execution->backend() == nullptr) {
            continue;
        }
        info->actualBackendKnown = true;
        info->actualBackend = forwardTypeName(command->execution->backend()->type());
        return;
    }
}

void fillActualBackend(const Command& command, OpInfo* info) {
    if (info == nullptr || !command.execution || command.execution->backend() == nullptr) {
        return;
    }
    info->actualBackendKnown = true;
    info->actualBackend = forwardTypeName(command.execution->backend()->type());
}

std::string commandDebugName(const Command& command, const Op* originOp, int commandIndex, int totalIndex) {
    if (command.op != nullptr && command.op->name() != nullptr) {
        return command.op->name()->str();
    }
    if (originOp != nullptr && originOp->name() != nullptr) {
        return originOp->name()->str() + "_raster_" + std::to_string(commandIndex);
    }
    return "_raster_" + std::to_string(totalIndex);
}

OpInfo buildOpInfo(const Op* op,
                   const Op* originOp,
                   const std::vector<Tensor*>& inputs,
                   const std::string& opName,
                   const std::string& plannedBackend,
                   const std::string& baseDir,
                   const std::string& npuDir) {
    OpInfo info;
    info.opName = opName;
    info.plannedBackend = plannedBackend;
    info.inputShapes = copyTensorShapes(inputs);
    if (op != nullptr) {
        fillQnnInfo(op, baseDir, npuDir, &info);
    } else if (originOp != nullptr) {
        fillQnnInfo(originOp, baseDir, npuDir, &info);
    }
    return info;
}

} // namespace

QnnGraphInfo::QnnGraphInfo()
    : offset(0),
      size(0),
      pin(false),
      draft(false) {
}

OpInfo::OpInfo()
    : actualBackendKnown(false),
      isPlugin(false) {
}

bool isQnnPluginOp(const OpInfo& op) {
    if (!op.isPlugin) {
        return false;
    }
    if (containsToken(op.pluginType, "qnn") || containsToken(effectiveBackend(op), "qnn")) {
        return true;
    }
    return !op.qnn.allGraphName.empty() && (!op.qnn.relativePath.empty() || !op.qnn.path.empty());
}

int selectQnnBucketSize(const QnnGraphInfo& qnn, int requestGroupSize) {
    if (qnn.bucketSizes.empty() || qnn.bucketSizes.size() != qnn.allGraphName.size()) {
        return requestGroupSize > 0 ? requestGroupSize : 1;
    }
    const int shapeIndex = selectQnnShapeIndex(qnn, requestGroupSize);
    if (shapeIndex >= 0 && shapeIndex < static_cast<int>(qnn.bucketSizes.size())) {
        return qnn.bucketSizes[shapeIndex];
    }
    return -1;
}

int selectQnnCompatibleBucketSize(const GraphSnapshot& snapshot, int requiredSize) {
    int paddedSize = requiredSize > 0 ? requiredSize : 1;
    bool foundQnn = false;
    while (true) {
        int nextSize = paddedSize;
        for (size_t i = 0; i < snapshot.ops.size(); ++i) {
            const OpInfo& op = snapshot.ops[i];
            if (!isQnnPluginOp(op)) {
                continue;
            }
            foundQnn = true;
            const int bucketSize = selectQnnBucketSize(op.qnn, paddedSize);
            if (bucketSize < paddedSize) {
                return -1;
            }
            nextSize = std::max(nextSize, bucketSize);
        }
        if (nextSize == paddedSize) {
            return foundQnn ? paddedSize : requiredSize;
        }
        paddedSize = nextSize;
    }
}

GraphSnapshot buildGraphSnapshot(const MNN::Session* session,
                                 int pipelineIndex,
                                 const std::string& baseDir,
                                 const std::string& npuDir) {
    GraphSnapshot snapshot;
    if (session == nullptr || pipelineIndex < 0) {
        return snapshot;
    }

    const Schedule::PipelineInfo& pipelineInfo = session->getPipelineInfo(pipelineIndex);
    const std::vector<Schedule::OpCacheInfo>& opCaches = pipelineInfo.second;
    const std::string plannedBackend = forwardTypeName(pipelineInfo.first.info.type);
    int totalCommandIndex = 0;
    for (size_t i = 0; i < opCaches.size(); ++i) {
        const Schedule::OpCacheInfo& cacheInfo = opCaches[i];
        const std::vector<std::shared_ptr<Command>>& commands = cacheInfo.executeBuffer.command;
        if (!commands.empty()) {
            for (size_t commandIndex = 0; commandIndex < commands.size(); ++commandIndex) {
                const std::shared_ptr<Command>& command = commands[commandIndex];
                if (!command || command->op == nullptr) {
                    continue;
                }
                OpInfo info = buildOpInfo(command->op,
                                          cacheInfo.op,
                                          command->inputs,
                                          commandDebugName(*command, cacheInfo.op,
                                                           static_cast<int>(commandIndex), totalCommandIndex),
                                          plannedBackend,
                                          baseDir,
                                          npuDir);
                fillActualBackend(*command, &info);
                snapshot.ops.push_back(info);
                ++totalCommandIndex;
            }
            continue;
        }

        const Op* op = cacheInfo.op;
        const std::string opName = op != nullptr && op->name() != nullptr ? op->name()->str() : "";
        OpInfo info = buildOpInfo(op,
                                  op,
                                  cacheInfo.inputs,
                                  opName,
                                  plannedBackend,
                                  baseDir,
                                  npuDir);
        ++totalCommandIndex;
        fillActualBackend(cacheInfo, &info);
        snapshot.ops.push_back(info);
    }
    return snapshot;
}

GraphSnapshot buildQnnGraphSnapshotFromModel(const std::string& modelPath,
                                             const std::string& baseDir,
                                             const std::string& npuDir) {
    GraphSnapshot snapshot;
    std::ifstream input(modelPath.c_str(), std::ios::binary);
    if (!input.good()) {
        return snapshot;
    }
    std::vector<char> buffer((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (buffer.empty()) {
        return snapshot;
    }
    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size());
    if (!VerifyNetBuffer(verifier)) {
        return snapshot;
    }
    const Net* net = GetNet(buffer.data());
    if (net == nullptr || net->oplists() == nullptr) {
        return snapshot;
    }
    snapshot.ops.reserve(net->oplists()->size());
    for (flatbuffers::uoffset_t i = 0; i < net->oplists()->size(); ++i) {
        const Op* op = net->oplists()->Get(i);
        if (op == nullptr || op->type() != OpType_Plugin) {
            continue;
        }
        const std::string opName = op->name() == nullptr ? "" : op->name()->str();
        OpInfo info = buildOpInfo(op,
                                  op,
                                  std::vector<Tensor*>(),
                                  opName,
                                  "CPU",
                                  baseDir,
                                  npuDir);
        if (isQnnPluginOp(info)) {
            snapshot.ops.push_back(info);
        }
    }
    return snapshot;
}

GraphSnapshot mergeQnnGraphSnapshotsInExecutionOrder(const GraphSnapshot& executionSnapshot,
                                                     const GraphSnapshot& modelSnapshot) {
    GraphSnapshot result;
    std::map<std::string, const OpInfo*> modelOps;
    std::map<std::string, size_t> modelOpCounts;
    for (size_t i = 0; i < modelSnapshot.ops.size(); ++i) {
        const OpInfo& op = modelSnapshot.ops[i];
        if (isQnnPluginOp(op)) {
            ++modelOpCounts[op.opName];
            if (modelOps.find(op.opName) == modelOps.end()) {
                modelOps[op.opName] = &op;
            }
        }
    }

    std::set<std::string> executionNames;
    for (size_t i = 0; i < executionSnapshot.ops.size(); ++i) {
        const OpInfo& executionOp = executionSnapshot.ops[i];
        if (!isQnnPluginOp(executionOp)) {
            continue;
        }
        std::map<std::string, const OpInfo*>::const_iterator model = modelOps.find(executionOp.opName);
        result.ops.push_back(model == modelOps.end() ? executionOp : *model->second);
        executionNames.insert(executionOp.opName);
    }
    for (size_t i = 0; i < modelSnapshot.ops.size(); ++i) {
        const OpInfo& modelOp = modelSnapshot.ops[i];
        if (isQnnPluginOp(modelOp) &&
            (executionNames.find(modelOp.opName) == executionNames.end() || modelOpCounts[modelOp.opName] > 1)) {
            result.ops.push_back(modelOp);
        }
    }
    return result;
}

std::vector<DualPipelineScheduler::GraphRequest> buildQnnGraphRequests(const GraphSnapshot& snapshot,
                                                                       int start,
                                                                       int maxK) {
    return buildQnnGraphRequestsForSize(snapshot, start, maxK, 0);
}

std::vector<DualPipelineScheduler::GraphRequest> buildQnnGraphRequestsForSize(
    const GraphSnapshot& snapshot,
    int start,
    int maxK,
    int requestGroupSize) {
    std::vector<DualPipelineScheduler::GraphRequest> requests;
    if (start < 0 || maxK <= 0 || start >= static_cast<int>(snapshot.ops.size())) {
        return requests;
    }
    for (int i = start; i < static_cast<int>(snapshot.ops.size()) && static_cast<int>(requests.size()) < maxK; ++i) {
        const OpInfo& op = snapshot.ops[i];
        if (!isQnnPluginOp(op)) {
            continue;
        }
        DualPipelineScheduler::GraphRequest request;
        request.action = DualPipelineScheduler::GRAPH_LOAD;
        request.graphId = qnnGraphResourceId(op);
        request.graphPath = op.qnn.path;
        request.offset = op.qnn.offset;
        request.size = op.qnn.size;
        if (requestGroupSize > 0 && !op.qnn.graphPaths.empty()) {
            const int shapeIndex = selectQnnShapeIndex(op.qnn, requestGroupSize);
            if (shapeIndex < 0) {
                return {};
            }
            request.graphId = qnnGraphResourceIdForShape(op, shapeIndex);
            request.graphPath = qnnGraphPathForShape(op, shapeIndex);
            request.offset = 0;
            request.size = 0;
            request.shapeIndex = shapeIndex;
            request.bucketSize = op.qnn.bucketSizes[shapeIndex];
            request.allGraphName = {op.qnn.allGraphName[shapeIndex]};
        } else if (!op.qnn.graphPaths.empty()) {
            // The unqualified request is retained for compatibility with
            // callers that only need graph metadata. It must not be loaded.
            request.graphPath.clear();
        }
        if (request.allGraphName.empty()) {
            request.allGraphName = op.qnn.allGraphName;
        }
        request.residentBytes = request.size;
        request.draftGraph = op.qnn.draft;
        request.pinResident = op.qnn.pin;
        requests.push_back(request);
    }
    return requests;
}

} // namespace Transformer
} // namespace MNN
