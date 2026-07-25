//
//  llm_manager.cpp
//
//  LLM model management and lifecycle implementation
//

#include "llm_manager.hpp"
#include "../../../transformers/llm/engine/include/llm/llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

namespace mnncli {

std::unique_ptr<MNN::Transformer::Llm> LLMManager::CreateLLM(
    const std::string& config_path, 
    bool use_template,
    const std::string& scheduler_mode
) {
    std::unique_ptr<MNN::Transformer::Llm> llm(MNN::Transformer::Llm::createLLM(config_path));
    std::string runtime_config = use_template ? "{\"tmp_path\":\"tmp\"}" :
                                                "{\"tmp_path\":\"tmp\",\"use_template\":false}";
    if (!scheduler_mode.empty()) {
        runtime_config = use_template ?
            "{\"tmp_path\":\"tmp\",\"scheduler_mode\":\"" + scheduler_mode + "\"}" :
            "{\"tmp_path\":\"tmp\",\"use_template\":false,\"scheduler_mode\":\"" + scheduler_mode + "\"}";
    }
    llm->set_config(runtime_config);
    
    {
        AUTOTIME;
        if (!llm->load()) {
            return nullptr;
        }
    }
    
    if (true) {
        AUTOTIME;
        TuningPrepare(llm.get());
    }
    
    return llm;
}

void LLMManager::PrepareTuning(MNN::Transformer::Llm* llm) {
    TuningPrepare(llm);
}

void LLMManager::TuningPrepare(MNN::Transformer::Llm* llm) {
    llm->tuning(MNN::Transformer::OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
}

} // namespace mnncli
