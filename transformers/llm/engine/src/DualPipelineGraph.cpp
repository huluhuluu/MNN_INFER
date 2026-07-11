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

std::vector<DualPipelineScheduler::TensorShape> toSchedulerShapes(const std::vector<TensorShape>& shapes) {
    std::vector<DualPipelineScheduler::TensorShape> result;
    result.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        DualPipelineScheduler::TensorShape shape;
        shape.dims = shapes[i].dims;
        result.push_back(shape);
    }
    return result;
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
    if (!isQnnPluginOp(*info)) {
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
    info->qnn.shapeIndex = -1;
    if (!info->qnn.allGraphName.empty()) {
        info->qnn.shapeIndex = 0;
        info->qnn.targetGraphName = info->qnn.allGraphName[0];
    } else {
        info->qnn.targetGraphName = info->opName;
    }
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

} // namespace

QnnGraphInfo::QnnGraphInfo()
    : offset(0),
      size(0),
      shapeIndex(-1),
      pin(false),
      draft(false) {
}

OpInfo::OpInfo()
    : layerIndex(-1),
      opIndex(-1),
      opTypeId(0),
      actualBackendKnown(false),
      isPlugin(false) {
}

WindowRequest::WindowRequest()
    : requestId(-1),
      start(0),
      maxK(0) {
}

bool isCpuOp(const OpInfo& op) {
    const std::string backend = effectiveBackend(op);
    return containsToken(backend, "cpu");
}

bool isOpenCLOp(const OpInfo& op) {
    const std::string backend = effectiveBackend(op);
    return containsToken(backend, "opencl");
}

bool isQnnPluginOp(const OpInfo& op) {
    if (!op.isPlugin) {
        return false;
    }
    return containsToken(op.pluginType, "qnn") || containsToken(effectiveBackend(op), "qnn");
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
    snapshot.ops.reserve(opCaches.size());
    for (size_t i = 0; i < opCaches.size(); ++i) {
        const Schedule::OpCacheInfo& cacheInfo = opCaches[i];
        const Op* op = cacheInfo.op;
        OpInfo info;
        info.layerIndex = static_cast<int>(i);
        info.opIndex = static_cast<int>(i);
        info.plannedBackend = forwardTypeName(pipelineInfo.first.info.type);
        // Only copy already-known tensor shapes; do not trigger shape/geometry/resize work here.
        info.inputShapes = copyTensorShapes(cacheInfo.inputs);
        info.outputShapes = copyTensorShapes(cacheInfo.outputs);
        if (op != nullptr) {
            info.opTypeId = static_cast<int>(op->type());
            info.opTypeName = EnumNameOpType(op->type());
            if (op->name() != nullptr) {
                info.opName = op->name()->str();
            }
            fillQnnInfo(op, baseDir, npuDir, &info);
        }
        fillActualBackend(cacheInfo, &info);
        snapshot.ops.push_back(info);
    }
    return snapshot;
}

DualPipelineScheduler::PrefetchWindow buildPrefetchWindow(const GraphSnapshot& snapshot,
                                                          int start,
                                                          int maxK,
                                                          int reqId) {
    DualPipelineScheduler::PrefetchWindow window;
    window.requestId = reqId;
    window.startLayerIndex = (start >= 0 && start < static_cast<int>(snapshot.ops.size())) ? snapshot.ops[start].layerIndex : -1;
    window.maxOpCount = maxK > 0 ? maxK : 0;

    if (start < 0 || maxK <= 0 || start >= static_cast<int>(snapshot.ops.size())) {
        return window;
    }
    for (int i = start; i < static_cast<int>(snapshot.ops.size()) && static_cast<int>(window.requests.size()) < maxK; ++i) {
        const OpInfo& op = snapshot.ops[i];
        if (!isCpuOp(op) && !isOpenCLOp(op)) {
            continue;
        }
        DualPipelineScheduler::PrefetchResizeRequest request;
        request.requestId = reqId;
        request.layerIndex = op.layerIndex;
        request.opIndex = op.opIndex;
        request.opName = op.opName;
        request.backend = effectiveBackend(op);
        request.inputShapes = toSchedulerShapes(op.inputShapes);
        request.outputShapes = toSchedulerShapes(op.outputShapes);
        window.requests.push_back(request);
    }
    return window;
}

std::vector<DualPipelineScheduler::GraphRequest> buildQnnGraphRequests(const GraphSnapshot& snapshot,
                                                                       int start,
                                                                       int maxK,
                                                                       int reqId) {
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
        request.requestId = reqId;
        request.graphId = !op.qnn.targetGraphName.empty() ? op.qnn.targetGraphName : op.opName;
        request.graphPath = op.qnn.path;
        request.baseDir = op.qnn.baseDir;
        request.npuDir = op.qnn.npuDir;
        request.relativePath = op.qnn.relativePath;
        request.offset = op.qnn.offset;
        request.size = op.qnn.size;
        request.allGraphName = op.qnn.allGraphName;
        request.targetGraphName = op.qnn.targetGraphName;
        request.shapeIndex = op.qnn.shapeIndex;
        request.draftGraph = op.qnn.draft;
        request.pinResident = op.qnn.pin;
        requests.push_back(request);
    }
    return requests;
}

} // namespace Transformer
} // namespace MNN
