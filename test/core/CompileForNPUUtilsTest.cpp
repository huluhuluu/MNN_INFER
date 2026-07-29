//
//  CompileForNPUUtilsTest.cpp
//  MNNTests
//

#include <MNN/MNNDefine.h>
#include "MNNTestSuite.h"
#include "../../tools/cpp/CompileForNPUUtils.hpp"

#include <memory>
#include <set>

namespace {

std::unique_ptr<MNN::OpT> makeOp(MNN::OpType type,
                                 const char* name,
                                 std::vector<int> inputs,
                                 std::vector<int> outputs) {
    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
    op->type = type;
    op->name = name;
    op->inputIndexes = std::move(inputs);
    op->outputIndexes = std::move(outputs);
    return op;
}

} // namespace

class CompileForNPUReorderOptionalInputTest : public MNNTestCase {
public:
    bool run(int precision) override {
        MNN::NetT net;
        net.oplists.emplace_back(makeOp(MNN::OpType_Input, "input", {}, {0}));
        net.oplists.emplace_back(makeOp(MNN::OpType_TanH, "consumer", {0, -1}, {1}));

        MNNTEST_ASSERT(MNN::Tools::reorderOps(&net));
        MNNTEST_ASSERT(net.oplists.size() == 2);
        MNNTEST_ASSERT(net.oplists[0]->name == "input");
        MNNTEST_ASSERT(net.oplists[1]->name == "consumer");
        return true;
    }
};

class CompileForNPUReorderMissingProducerTest : public MNNTestCase {
public:
    bool run(int precision) override {
        MNN::NetT net;
        net.oplists.emplace_back(makeOp(MNN::OpType_TanH, "orphan", {7}, {1}));

        MNNTEST_ASSERT(!MNN::Tools::reorderOps(&net));
        return true;
    }
};

class CompileForNPUNoImplicitAttentionSkipTest : public MNNTestCase {
public:
    bool run(int precision) override {
        MNN::NetT net;
        net.oplists.emplace_back(makeOp(MNN::OpType_Attention, "attention", {0}, {1}));

        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNN::Net::Pack(builder, &net));
        auto packedNet = MNN::GetNet(builder.GetBufferPointer());
        auto outputs = MNN::Tools::collectExplicitSkipOutputs(packedNet, {});

        MNNTEST_ASSERT(outputs.empty());
        return true;
    }
};

class CompileForNPUExplicitAttentionSkipTest : public MNNTestCase {
public:
    bool run(int precision) override {
        MNN::NetT net;
        net.oplists.emplace_back(makeOp(MNN::OpType_Attention, "attention", {0}, {1}));

        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(MNN::Net::Pack(builder, &net));
        auto packedNet = MNN::GetNet(builder.GetBufferPointer());
        auto outputs = MNN::Tools::collectExplicitSkipOutputs(packedNet, {"attention"});

        MNNTEST_ASSERT(outputs == std::set<int>({1}));
        return true;
    }
};

MNNTestSuiteRegister(CompileForNPUReorderOptionalInputTest,
                     "tools/compilefornpu/reorder_optional_input");
MNNTestSuiteRegister(CompileForNPUReorderMissingProducerTest,
                     "tools/compilefornpu/reorder_missing_producer");
MNNTestSuiteRegister(CompileForNPUNoImplicitAttentionSkipTest,
                     "tools/compilefornpu/no_implicit_attention_skip");
MNNTestSuiteRegister(CompileForNPUExplicitAttentionSkipTest,
                     "tools/compilefornpu/explicit_attention_skip");
