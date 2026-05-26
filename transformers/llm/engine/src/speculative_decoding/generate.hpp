//
//  generate.hpp
//
//  Created by MNN on 2025/06/09.
//

#ifndef SPEC_GENERATE_HPP
#define SPEC_GENERATE_HPP

#include <MNN/AutoTime.hpp>
#include "llm/llm.hpp"
#include "../llmconfig.hpp"
#include "../kvmeta.hpp"

//#define DUMP_PROFILE_INFO

namespace MNN {
namespace Transformer {
// ==================== Eagle Context ====================
// Statistics for Eagle speculative decoding
struct EagleContext {
    uint draft = 0;              // total draft tokens generated
    uint accepted = 0;           // total accepted tokens
    uint steps = 0;                    // total target verify calls
    uint64_t draft_time_us = 0;        // total draft model time (microseconds)
    uint64_t draft_prefill_time_us = 0; // initial draft tree build time (microseconds)
    uint64_t draft_decode_time_us = 0;  // iterative draft update time (microseconds)
    uint64_t target_time_us = 0;       // total target model time (microseconds)
    
    void reset() {
        draft = 0;
        accepted = 0;
        steps = 0;
        draft_time_us = 0;
        draft_prefill_time_us = 0;
        draft_decode_time_us = 0;
        target_time_us = 0;
    }
    
    float avgAcceptLen() const {
        return accepted==0 ? 0.0f : accepted / (steps * 1.0f); 
    }
    
    float acceptRate() const {
        return accepted==0 ? 0.0f : accepted / (draft * 1.0f);
    }
    
    float compressionRatio() const {
        return accepted==0 ? 0.0f : draft / (accepted * 1.0f);
    }
    
    float avgDraftTimeMs() const {
        return steps == 0 ? 0.0f : draft_time_us / 1000.0f / steps;
    }
    
    float avgTargetTimeMs() const {
        return steps == 0 ? 0.0f : target_time_us / 1000.0f / steps;
    }
    
    /**
     * Calculate theoretical speedup compared to non-speculative decoding.
     * 
     * Formula:
     *   - Without speculation: total_time = accepted * T_target_per_token
     *   - With speculation: total_time = draft_time + target_time
     *   - Speedup = (accepted * T_target_per_step / avg_accept_len) / (draft_time + target_time)
     * 
     * @return theoretical speedup ratio (>1 means faster)
     */
    float theoreticalSpeedup() const {
        if (draft_time_us == 0 && target_time_us == 0) return 0.0f;
        if (accepted == 0 || steps == 0) return 0.0f;
        
        // Total time with speculative decoding
        float total_time_us = static_cast<float>(draft_time_us + target_time_us);
        
        // Baseline: if we only use target model, we need 'accepted' forward passes
        // Each forward pass takes target_time_us / steps (average per step)
        float target_time_per_step_us = static_cast<float>(target_time_us) / steps;
        float baseline_time_us = accepted * target_time_per_step_us;
        
        return baseline_time_us / total_time_us;
    }
};

// ==================== Generation Params ====================

struct GenerationParams {
    int max_new_tokens;
    std::vector<int> input_ids;
    MNN::Express::VARP input_embeds;
    std::vector<MNN::Express::VARP> outputs;
    int validLogitStart = 0;
    int validLogitSize = 0;
};

class Generation {
public:
    Generation(Llm* llm, std::shared_ptr<LlmContext> context) {
        mLlm = llm;
        mContext = context;
    };
    virtual ~Generation() = default;
    virtual void load(Module::Config module_config) {
        // do nothing
    };
    virtual void generate(GenerationParams& param) = 0;
protected:
    int draftVerify(MNN::Express::VARP logits, const std::vector<int>& drafts, bool& stop);
    std::shared_ptr<LlmContext> mContext;
    Llm* mLlm;
};

class ArGeneration: public Generation {
public:
    ArGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    virtual ~ArGeneration() = default;
    virtual void generate(GenerationParams& param);
};

class LookaheadGeneration: public Generation {
public:
    LookaheadGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    virtual ~LookaheadGeneration() = default;
    virtual void generate(GenerationParams& param);
private:
    int mNgramKeyMaxLen = 4;
    MatchStrictLevel mStrictLevel;
    bool mUpdateNgram = false;
    NgramSelectRule mSelectRule;
};

class MtpGeneration: public Generation {
public:
    MtpGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    virtual ~MtpGeneration() = default;
    virtual void load(Module::Config module_config) override;
    virtual void generate(GenerationParams& param) override;
private:
    std::vector<MNN::Express::VARP> mtpForward(const std::vector<int>& input_ids, MNN::Express::VARP hidden_states);
    std::vector<MNN::Express::VARP> mtpForward(MNN::Express::VARP input_embeds, MNN::Express::VARP hidden_states);

    std::vector<std::shared_ptr<MNN::Express::Module>> mMtpModules;
    std::map<std::pair<int, bool>, std::shared_ptr<MNN::Express::Module>> mMtpModulePool;
    std::shared_ptr<KVMeta> mMtpMeta;
    int mHiddenStateIndex = -1;
};

class EagleGeneration: public Generation {
public:
    EagleGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    virtual ~EagleGeneration() = default;
    virtual void load(Module::Config module_config) override;
    virtual void generate(GenerationParams& param) override;
    
    // Eagle context interface
    EagleContext* getEagleContext() { return &mEagleContext; }
    const EagleContext* getEagleContext() const { return &mEagleContext; }
    void resetEagleContext() { mEagleContext.reset(); }
private: // For eagle_eval access
    struct DraftInfo {
        std::vector<int> draftTokens;
        std::vector<std::vector<int>> retrieveIndices;
        VARP attentionMask;
        VARP positionIds;
    };
    struct AcceptInfo {
        std::vector<int> sampleTokens;
        std::vector<int> acceptIndices;
        std::vector<int> acceptTokens;
    };
    MNN::Express::VARPS eagleForwardRaw(const MNN::Express::VARPS& inputs);
    MNN::Express::VARPS eagleForward(const std::vector<int>& inputEmbeds, MNN::Express::VARP hiddenStates, bool allLogits = false);
    MNN::Express::VARPS eagleForward(MNN::Express::VARP inputEmbeds, MNN::Express::VARP hiddenStates, bool allLogits = false);
    DraftInfo topkGenerate(const std::vector<int>& inputIds, MNN::Express::VARP hiddenStates, MNN::Express::VARP inputEmbeds = nullptr);
    VARPS treeDecoding(const DraftInfo& draftInfo);
    AcceptInfo evaluatePosterior(const DraftInfo& drafInfo, VARP logits);
    DraftInfo updateDraft(const AcceptInfo& accpetInfo, VARP hiddenStates);
    MNN::Express::VARP getMask(std::vector<std::vector<bool>> mask, int seqLen);
    bool processTokens(const std::vector<int>& accpetTokens);
    void setPosition(int position);
    std::string tokenStr(int token);
    
    EagleContext mEagleContext;
    std::vector<std::shared_ptr<MNN::Express::Module>> mEagleModules;
    std::shared_ptr<KVMeta> mEagleMeta;
    MNN::Express::VARP mD2t, mTreePosition;
    int mTopK, mDepth;
    int mEaglePastLen = 0, mEagleRemove = 0;
};


class GenerationStrategyFactory {
public:
    static std::shared_ptr<Generation> create(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, bool canSpec);
};


} // namespace Transformer
} // namespace MNN
#endif
