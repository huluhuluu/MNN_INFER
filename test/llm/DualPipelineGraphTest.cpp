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

static OpInfo makeOp(const std::string& name, const std::string& plannedBackend) {
    OpInfo op;
    op.opName = name;
    op.plannedBackend = plannedBackend;
    op.inputShapes.push_back(shape({1, 2, 3}));
    return op;
}

class DualPipelineGraphQnnRequestTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo qnn = makeOp("qnn_plugin", "QNN");
        qnn.isPlugin = true;
        qnn.pluginType = "QNN";
        qnn.qnn.path = "/models/qnn_context.bin";
        qnn.qnn.offset = 4096;
        qnn.qnn.size = 8192;
        qnn.qnn.allGraphName.push_back("all_graph");
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
        MNNTEST_ASSERT(!requests[0].draftGraph);
        MNNTEST_ASSERT(!requests[0].pinResident);
        return true;
    }
};

class DualPipelineGraphQnnDraftPinTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo qnn = makeOp("draft_plugin", "plugin");
        qnn.isPlugin = true;
        qnn.pluginType = "qnn_plugin";
        qnn.qnn.path = "/models/draft.bin";
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

class DualPipelineGraphQnnPerShapePathTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo qnn = makeOp("qnn_per_shape_plugin", "QNN");
        qnn.isPlugin = true;
        qnn.pluginType = "QNN";
        qnn.qnn.path = "/models/qnn/graph0.bin";
        qnn.qnn.allGraphName = {"graph0", "graph1", "graph2"};
        qnn.qnn.baseDir = "/models";
        qnn.qnn.graphPaths = {"qnn/graph0_0.bin", "qnn/graph0_1.bin", "qnn/graph0_2.bin"};
        qnn.qnn.bucketSizes = {256, 32, 1};
        snapshot.ops.push_back(qnn);

        const auto requests = buildQnnGraphRequests(snapshot, 0, 1, 15);
        MNNTEST_ASSERT(requests.size() == 1);
        MNNTEST_ASSERT(requests[0].graphPath.empty());
        MNNTEST_ASSERT(requests[0].allGraphName == qnn.qnn.allGraphName);

        const auto selected = buildQnnGraphRequestsForSize(snapshot, 0, 1, 15, 20);
        MNNTEST_ASSERT(selected.size() == 1);
        MNNTEST_ASSERT(selected[0].graphPath == "/models/qnn/graph0_1.bin");
        MNNTEST_ASSERT(selected[0].allGraphName == std::vector<std::string>({"graph1"}));
        MNNTEST_ASSERT(selected[0].graphId == "/models/qnn/graph0_1.bin#0#0#graph1");
        MNNTEST_ASSERT(selected[0].offset == 0);
        MNNTEST_ASSERT(selected[0].size == 0);
        MNNTEST_ASSERT(selected[0].shapeIndex == 1);
        MNNTEST_ASSERT(selected[0].bucketSize == 32);
        MNNTEST_ASSERT(buildQnnGraphRequestsForSize(snapshot, 0, 1, 15, 257).empty());
        return true;
    }
};

class DualPipelineGraphQnnBucketSelectionTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo qnn = makeOp("qnn_bucket_plugin", "QNN");
        qnn.isPlugin = true;
        qnn.pluginType = "QNN";
        qnn.qnn.allGraphName.push_back("graph_s128");
        qnn.qnn.allGraphName.push_back("graph_s8");
        qnn.qnn.allGraphName.push_back("graph_s1");
        qnn.qnn.bucketSizes.push_back(128);
        qnn.qnn.bucketSizes.push_back(8);
        qnn.qnn.bucketSizes.push_back(1);
        snapshot.ops.push_back(qnn);

        const auto requests = buildQnnGraphRequests(snapshot, 0, 1, 21);
        MNNTEST_ASSERT(requests[0].graphId == "qnn_bucket_plugin#0#0#graph_s128#graph_s8#graph_s1");
        MNNTEST_ASSERT(selectQnnBucketSize(qnn.qnn, 1) == 1);
        MNNTEST_ASSERT(selectQnnBucketSize(qnn.qnn, 2) == 8);
        MNNTEST_ASSERT(selectQnnBucketSize(qnn.qnn, 129) == -1);
        MNNTEST_ASSERT(selectQnnCompatibleBucketSize(snapshot, 2) == 8);
        MNNTEST_ASSERT(selectQnnCompatibleBucketSize(snapshot, 129) == -1);

        OpInfo second = qnn;
        second.opName = "qnn_bucket_plugin_2";
        second.qnn.bucketSizes[1] = 16;
        snapshot.ops.push_back(second);
        MNNTEST_ASSERT(selectQnnCompatibleBucketSize(snapshot, 2) == 128);
        return true;
    }
};

class DualPipelineGraphQnnResourceIdTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo first = makeOp("qnn_layer_0", "QNN");
        first.isPlugin = true;
        first.pluginType = "QNN";
        first.qnn.path = "/models/qnn/graph0.bin";
        first.qnn.allGraphName.push_back("graph_s8");
        snapshot.ops.push_back(first);

        OpInfo second = first;
        second.opName = "qnn_layer_1";
        second.qnn.path = "/models/qnn/graph1.bin";
        snapshot.ops.push_back(second);

        const auto requests = buildQnnGraphRequests(snapshot, 0, 2, 24);
        MNNTEST_ASSERT(requests.size() == 2);
        MNNTEST_ASSERT(requests[0].graphId != requests[1].graphId);
        return true;
    }
};

class DualPipelineGraphQnnMetadataWithoutTypeTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo qnn = makeOp("exported_qnn_plugin", "CPU");
        qnn.isPlugin = true;
        qnn.pluginType = "Plugin";
        qnn.qnn.path = "/models/qnn/graph0.bin";
        qnn.qnn.relativePath = "qnn/graph0.bin";
        qnn.qnn.allGraphName.push_back("graph0");
        snapshot.ops.push_back(qnn);

        auto requests = buildQnnGraphRequests(snapshot, 0, 1, 19);

        MNNTEST_ASSERT(requests.size() == 1);
        MNNTEST_ASSERT(requests[0].graphId == "/models/qnn/graph0.bin#0#0#graph0");
        MNNTEST_ASSERT(requests[0].graphPath == "/models/qnn/graph0.bin");
        MNNTEST_ASSERT(requests[0].allGraphName == std::vector<std::string>({"graph0"}));
        return true;
    }
};

class DualPipelineGraphExecutionOrderMergeTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot model;
        OpInfo modelA = makeOp("qnn_a", "QNN");
        modelA.isPlugin = true;
        modelA.pluginType = "QNN";
        modelA.qnn.path = "/models/a.bin";
        modelA.qnn.allGraphName.push_back("a");
        OpInfo modelB = modelA;
        modelB.opName = "qnn_b";
        modelB.qnn.path = "/models/b.bin";
        modelB.qnn.allGraphName[0] = "b";
        OpInfo modelC = modelA;
        modelC.opName = "qnn_c";
        modelC.qnn.path = "/models/c.bin";
        modelC.qnn.allGraphName[0] = "c";
        model.ops = {modelA, modelB, modelC};

        GraphSnapshot execution;
        execution.ops = {modelC, modelA};
        const GraphSnapshot merged = mergeQnnGraphSnapshotsInExecutionOrder(execution, model);
        MNNTEST_ASSERT(merged.ops.size() == 3);
        MNNTEST_ASSERT(merged.ops[0].opName == "qnn_c");
        MNNTEST_ASSERT(merged.ops[1].opName == "qnn_a");
        MNNTEST_ASSERT(merged.ops[2].opName == "qnn_b");
        MNNTEST_ASSERT(merged.ops[0].qnn.path == "/models/c.bin");
        return true;
    }
};

class DualPipelineGraphSkipsNonQnnPluginTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        GraphSnapshot snapshot;
        OpInfo plugin = makeOp("custom_plugin", "plugin");
        plugin.isPlugin = true;
        plugin.pluginType = "custom_cpu_plugin";
        plugin.qnn.path = "/models/not_qnn.bin";
        snapshot.ops.push_back(plugin);

        auto requests = buildQnnGraphRequests(snapshot, 0, 8, 17);

        MNNTEST_ASSERT(requests.empty());
        return true;
    }
};

MNNTestSuiteRegister(DualPipelineGraphQnnRequestTest, "llm/dual_pipeline_graph_qnn_request");
MNNTestSuiteRegister(DualPipelineGraphQnnDraftPinTest, "llm/dual_pipeline_graph_qnn_draft_pin");
MNNTestSuiteRegister(DualPipelineGraphQnnPerShapePathTest, "llm/dual_pipeline_graph_qnn_per_shape_path");
MNNTestSuiteRegister(DualPipelineGraphQnnBucketSelectionTest, "llm/dual_pipeline_graph_qnn_bucket_selection");
MNNTestSuiteRegister(DualPipelineGraphQnnResourceIdTest, "llm/dual_pipeline_graph_qnn_resource_id");
MNNTestSuiteRegister(DualPipelineGraphQnnMetadataWithoutTypeTest, "llm/dual_pipeline_graph_qnn_metadata_without_type");
MNNTestSuiteRegister(DualPipelineGraphSkipsNonQnnPluginTest, "llm/dual_pipeline_graph_skip_non_qnn_plugin");
MNNTestSuiteRegister(DualPipelineGraphExecutionOrderMergeTest, "llm/dual_pipeline_graph_execution_order_merge");
