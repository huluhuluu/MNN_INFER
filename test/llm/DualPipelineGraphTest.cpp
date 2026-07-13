//
//  DualPipelineGraphTest.cpp
//  MNN
//

#include <MNN/MNNDefine.h>
#include "../MNNTestSuite.h"
#include "../../transformers/llm/engine/include/llm/DualPipelineGraph.hpp"

using namespace MNN::Transformer;

static TensorShape shape(const std::vector<int>& dims) {
    TensorShape result;
    result.dims = dims;
    return result;
}

static OpInfo makeOp(int opIndex, const std::string& name, const std::string& plannedBackend) {
    OpInfo op;
    op.layerIndex = opIndex / 2;
    op.opIndex = opIndex;
    op.opName = name;
    op.opTypeId = opIndex + 100;
    op.opTypeName = "TestOp";
    op.plannedBackend = plannedBackend;
    op.inputShapes.push_back(shape({1, 2, 3}));
    op.outputShapes.push_back(shape({1, 2, 4}));
    return op;
}

class DualPipelineGraphPrefetchWindowTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        snapshot.ops.push_back(makeOp(0, "cpu_0", "CPU"));
        snapshot.ops.push_back(makeOp(1, "qnn_plugin", "QNN"));
        snapshot.ops.back().isPlugin = true;
        snapshot.ops.back().pluginType = "QNN";
        snapshot.ops.back().qnn.path = "/tmp/qnn.bin";
        snapshot.ops.back().qnn.targetGraphName = "qnn_graph";
        snapshot.ops.push_back(makeOp(2, "opencl_2", "OpenCL"));
        snapshot.ops.push_back(makeOp(3, "cpu_3", "MNN_FORWARD_CPU"));

        auto window = buildPrefetchWindow(snapshot, 0, 2, 7);

        MNNTEST_ASSERT(window.requestId == 7);
        MNNTEST_ASSERT(window.startLayerIndex == 0);
        MNNTEST_ASSERT(window.maxOpCount == 2);
        MNNTEST_ASSERT(window.requests.size() == 2);
        MNNTEST_ASSERT(window.requests[0].requestId == 7);
        MNNTEST_ASSERT(window.requests[0].opIndex == 0);
        MNNTEST_ASSERT(window.requests[0].opName == "cpu_0");
        MNNTEST_ASSERT(window.requests[0].backend == "CPU");
        MNNTEST_ASSERT(window.requests[0].inputShapes[0].dims == std::vector<int>({1, 2, 3}));
        MNNTEST_ASSERT(window.requests[1].opIndex == 2);
        MNNTEST_ASSERT(window.requests[1].opName == "opencl_2");
        MNNTEST_ASSERT(window.requests[1].backend == "OpenCL");
        return true;
    }
};

class DualPipelineGraphQnnRequestTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo qnn = makeOp(4, "qnn_plugin", "QNN");
        qnn.isPlugin = true;
        qnn.pluginType = "QNN";
        qnn.qnn.path = "/models/qnn_context.bin";
        qnn.qnn.offset = 4096;
        qnn.qnn.size = 8192;
        qnn.qnn.allGraphName.push_back("all_graph");
        qnn.qnn.targetGraphName = "target_graph";
        snapshot.ops.push_back(qnn);

        auto requests = buildQnnGraphRequests(snapshot, 0, 4, 11);

        MNNTEST_ASSERT(requests.size() == 1);
        MNNTEST_ASSERT(requests[0].action == DualPipelineScheduler::GRAPH_LOAD);
        MNNTEST_ASSERT(requests[0].requestId == 11);
        MNNTEST_ASSERT(requests[0].graphId == "/models/qnn_context.bin#4096#8192#all_graph");
        MNNTEST_ASSERT(requests[0].graphPath == "/models/qnn_context.bin");
        MNNTEST_ASSERT(requests[0].offset == 4096);
        MNNTEST_ASSERT(requests[0].size == 8192);
        MNNTEST_ASSERT(requests[0].allGraphName == std::vector<std::string>({"all_graph"}));
        MNNTEST_ASSERT(requests[0].targetGraphName == "target_graph");
        MNNTEST_ASSERT(!requests[0].draftGraph);
        MNNTEST_ASSERT(!requests[0].pinResident);
        return true;
    }
};

class DualPipelineGraphQnnDraftPinTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo qnn = makeOp(5, "draft_plugin", "plugin");
        qnn.isPlugin = true;
        qnn.pluginType = "qnn_plugin";
        qnn.qnn.path = "/models/draft.bin";
        qnn.qnn.targetGraphName = "draft_graph";
        qnn.qnn.draft = true;
        qnn.qnn.pin = true;
        snapshot.ops.push_back(qnn);

        auto requests = buildQnnGraphRequests(snapshot, 0, 1, 13);

        MNNTEST_ASSERT(requests.size() == 1);
        MNNTEST_ASSERT(requests[0].draftGraph);
        MNNTEST_ASSERT(requests[0].pinResident);
        MNNTEST_ASSERT(requests[0].graphId == "/models/draft.bin#0#0");
        return true;
    }
};

class DualPipelineGraphQnnBucketSelectionTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo qnn = makeOp(7, "qnn_bucket_plugin", "QNN");
        qnn.isPlugin = true;
        qnn.pluginType = "QNN";
        qnn.qnn.allGraphName.push_back("graph_s128");
        qnn.qnn.allGraphName.push_back("graph_s8");
        qnn.qnn.allGraphName.push_back("graph_s1");
        qnn.qnn.bucketSizes.push_back(128);
        qnn.qnn.bucketSizes.push_back(8);
        qnn.qnn.bucketSizes.push_back(1);
        qnn.qnn.shapeIndex = 2;
        qnn.qnn.targetGraphName = "graph_s1";
        snapshot.ops.push_back(qnn);

        auto groupOne = buildQnnGraphRequests(snapshot, 0, 1, 21, 1);
        auto groupTwo = buildQnnGraphRequests(snapshot, 0, 1, 22, 2);
        auto groupLarge = buildQnnGraphRequests(snapshot, 0, 1, 23, 129);

        MNNTEST_ASSERT(groupOne[0].shapeIndex == 2);
        MNNTEST_ASSERT(groupOne[0].targetGraphName == "graph_s1");
        MNNTEST_ASSERT(groupOne[0].graphId == "qnn_bucket_plugin#0#0#graph_s128#graph_s8#graph_s1");
        MNNTEST_ASSERT(groupTwo[0].shapeIndex == 1);
        MNNTEST_ASSERT(groupTwo[0].targetGraphName == "graph_s8");
        MNNTEST_ASSERT(groupTwo[0].graphId == groupOne[0].graphId);
        MNNTEST_ASSERT(groupLarge[0].shapeIndex == 0);
        MNNTEST_ASSERT(groupLarge[0].targetGraphName == "graph_s128");
        MNNTEST_ASSERT(groupLarge[0].graphId == groupOne[0].graphId);
        MNNTEST_ASSERT(selectQnnBucketSize(qnn.qnn, 1) == 1);
        MNNTEST_ASSERT(selectQnnBucketSize(qnn.qnn, 2) == 8);
        MNNTEST_ASSERT(selectQnnBucketSize(qnn.qnn, 129) == 128);
        return true;
    }
};

class DualPipelineGraphQnnResourceIdTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo first = makeOp(8, "qnn_layer_0", "QNN");
        first.isPlugin = true;
        first.pluginType = "QNN";
        first.qnn.path = "/models/qnn/graph0.bin";
        first.qnn.allGraphName.push_back("graph_s8");
        first.qnn.targetGraphName = "graph_s8";
        snapshot.ops.push_back(first);

        OpInfo second = first;
        second.opName = "qnn_layer_1";
        second.qnn.path = "/models/qnn/graph1.bin";
        snapshot.ops.push_back(second);

        const auto requests = buildQnnGraphRequests(snapshot, 0, 2, 24, 2);
        MNNTEST_ASSERT(requests.size() == 2);
        MNNTEST_ASSERT(requests[0].targetGraphName == requests[1].targetGraphName);
        MNNTEST_ASSERT(requests[0].graphId != requests[1].graphId);
        return true;
    }
};

class DualPipelineGraphQnnMetadataWithoutTypeTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo qnn = makeOp(6, "exported_qnn_plugin", "CPU");
        qnn.isPlugin = true;
        qnn.pluginType = "Plugin";
        qnn.qnn.path = "/models/qnn/graph0.bin";
        qnn.qnn.relativePath = "qnn/graph0.bin";
        qnn.qnn.allGraphName.push_back("graph0");
        qnn.qnn.targetGraphName = "graph0";
        snapshot.ops.push_back(qnn);

        auto requests = buildQnnGraphRequests(snapshot, 0, 1, 19);

        MNNTEST_ASSERT(requests.size() == 1);
        MNNTEST_ASSERT(requests[0].graphId == "/models/qnn/graph0.bin#0#0#graph0");
        MNNTEST_ASSERT(requests[0].graphPath == "/models/qnn/graph0.bin");
        MNNTEST_ASSERT(requests[0].relativePath == "qnn/graph0.bin");
        MNNTEST_ASSERT(requests[0].allGraphName == std::vector<std::string>({"graph0"}));
        return true;
    }
};

class DualPipelineGraphSkipsNonQnnPluginTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo plugin = makeOp(6, "custom_plugin", "plugin");
        plugin.isPlugin = true;
        plugin.pluginType = "custom_cpu_plugin";
        plugin.qnn.path = "/models/not_qnn.bin";
        plugin.qnn.targetGraphName = "not_qnn";
        snapshot.ops.push_back(plugin);

        auto requests = buildQnnGraphRequests(snapshot, 0, 8, 17);

        MNNTEST_ASSERT(requests.empty());
        return true;
    }
};

MNNTestSuiteRegister(DualPipelineGraphPrefetchWindowTest, "llm/dual_pipeline_graph_prefetch_window");
MNNTestSuiteRegister(DualPipelineGraphQnnRequestTest, "llm/dual_pipeline_graph_qnn_request");
MNNTestSuiteRegister(DualPipelineGraphQnnDraftPinTest, "llm/dual_pipeline_graph_qnn_draft_pin");
MNNTestSuiteRegister(DualPipelineGraphQnnBucketSelectionTest, "llm/dual_pipeline_graph_qnn_bucket_selection");
MNNTestSuiteRegister(DualPipelineGraphQnnResourceIdTest, "llm/dual_pipeline_graph_qnn_resource_id");
MNNTestSuiteRegister(DualPipelineGraphQnnMetadataWithoutTypeTest, "llm/dual_pipeline_graph_qnn_metadata_without_type");
MNNTestSuiteRegister(DualPipelineGraphSkipsNonQnnPluginTest, "llm/dual_pipeline_graph_skip_non_qnn_plugin");
