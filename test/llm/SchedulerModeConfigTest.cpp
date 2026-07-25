//
//  SchedulerModeConfigTest.cpp
//

#include "../MNNTestSuite.h"
#include "../../transformers/llm/engine/src/llmconfig.hpp"

using namespace MNN::Transformer;

class SchedulerModeLegacyConfigTest : public MNNTestCase {
public:
    bool run(int) override {
        LlmConfig config;
        config.config_ = rapid_json_wrapper::parse("{}");
        MNNTEST_ASSERT(config.scheduler_mode() == "single_request");
        MNNTEST_ASSERT(!config.dual_pipeline_mode());
        MNNTEST_ASSERT(!config.scheduler_requires_packed_attention());

        config.config_ = rapid_json_wrapper::parse("{\"dual_pipeline_mode\":true}");
        MNNTEST_ASSERT(config.scheduler_mode() == "dual_pipeline");
        MNNTEST_ASSERT(config.dual_pipeline_mode());
        MNNTEST_ASSERT(config.scheduler_requires_packed_attention());
        return true;
    }
};

class SchedulerModeExplicitConfigTest : public MNNTestCase {
public:
    bool run(int) override {
        LlmConfig config;
        config.config_ = rapid_json_wrapper::parse(
            "{\"dual_pipeline_mode\":true,\"scheduler_mode\":\"continuous_batch\"}");
        MNNTEST_ASSERT(config.scheduler_mode() == "continuous_batch");
        MNNTEST_ASSERT(!config.dual_pipeline_mode());
        MNNTEST_ASSERT(config.continuous_batch_mode());
        MNNTEST_ASSERT(config.scheduler_requires_packed_attention());

        config.config_ = rapid_json_wrapper::parse("{\"scheduler_mode\":\"single_request\"}");
        MNNTEST_ASSERT(config.valid_scheduler_mode());
        MNNTEST_ASSERT(!config.scheduler_requires_packed_attention());
        return true;
    }
};

class SchedulerModeInvalidConfigTest : public MNNTestCase {
public:
    bool run(int) override {
        LlmConfig config;
        config.config_ = rapid_json_wrapper::parse("{\"scheduler_mode\":\"invalid\"}");
        MNNTEST_ASSERT(!config.valid_scheduler_mode());

        config.config_ = rapid_json_wrapper::parse("{\"scheduler_mode\":1}");
        MNNTEST_ASSERT(config.scheduler_mode().empty());
        MNNTEST_ASSERT(!config.valid_scheduler_mode());
        return true;
    }
};

MNNTestSuiteRegister(SchedulerModeLegacyConfigTest, "llm/scheduler_mode_legacy_config");
MNNTestSuiteRegister(SchedulerModeExplicitConfigTest, "llm/scheduler_mode_explicit_config");
MNNTestSuiteRegister(SchedulerModeInvalidConfigTest, "llm/scheduler_mode_invalid_config");
