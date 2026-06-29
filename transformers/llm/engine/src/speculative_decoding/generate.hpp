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
#include <cstdint>
#include <map>

//#define DUMP_PROFILE_INFO

namespace MNN {
namespace Transformer {
struct SpecContext {
    unsigned int draft = 0;
    unsigned int accepted = 0;
    unsigned int steps = 0;
    std::map<int, unsigned int> accept_len_freq;
    uint64_t draft_time_us = 0;
    uint64_t draft_prefill_time_us = 0;
    uint64_t draft_decode_time_us = 0;
    uint64_t target_time_us = 0;

    void reset() {
        draft = 0;
        accepted = 0;
        steps = 0;
        accept_len_freq.clear();
        draft_time_us = 0;
        draft_prefill_time_us = 0;
        draft_decode_time_us = 0;
        target_time_us = 0;
    }

    float avgAcceptLen() const {
        return steps == 0 ? 0.0f : accepted / (steps * 1.0f);
    }

    float acceptRate() const {
        return draft == 0 ? 0.0f : accepted / (draft * 1.0f);
    }

    float compressionRatio() const {
        return accepted == 0 ? 0.0f : draft / (accepted * 1.0f);
    }

    float avgDraftTimeMs() const {
        return steps == 0 ? 0.0f : draft_time_us / 1000.0f / steps;
    }

    float avgTargetTimeMs() const {
        return steps == 0 ? 0.0f : target_time_us / 1000.0f / steps;
    }

    float theoreticalSpeedup() const {
        if ((draft_time_us == 0 && target_time_us == 0) || accepted == 0 || steps == 0) {
            return 0.0f;
        }
        float total_time_us = static_cast<float>(draft_time_us + target_time_us);
        float target_time_per_step_us = static_cast<float>(target_time_us) / steps;
        float baseline_time_us = accepted * target_time_per_step_us;
        return baseline_time_us / total_time_us;
    }
};

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
    virtual SpecContext* getSpecContext() { return nullptr; }
    virtual const SpecContext* getSpecContext() const { return nullptr; }
    virtual void resetSpecContext() {}
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
    virtual SpecContext* getSpecContext() override { return &mSpecContext; }
    virtual const SpecContext* getSpecContext() const override { return &mSpecContext; }
    virtual void resetSpecContext() override { mSpecContext.reset(); }
private:
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
    SpecContext mSpecContext;
    std::vector<std::shared_ptr<MNN::Express::Module>> mEagleModules;
    std::shared_ptr<KVMeta> mEagleMeta;
    MNN::Express::VARP mD2t, mTreePosition;
    int mTopK, mDepth;
    int mEaglePastLen = 0, mEagleRemove = 0;
};

class DFlashGeneration: public Generation {
public:
    DFlashGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    virtual ~DFlashGeneration() = default;
    virtual void load(Module::Config module_config) override;
    virtual void generate(GenerationParams& param) override;
    virtual SpecContext* getSpecContext() override { return &mDFlashContext; }
    virtual const SpecContext* getSpecContext() const override { return &mDFlashContext; }
    virtual void resetSpecContext() override { mDFlashContext.reset(); }
private:
    struct DFlashDraftInfo {
        std::vector<int> draftTokens;
        MNN::Express::VARP logits;
    };
    struct TreeInfo {
        std::vector<int> draftTokens;
        std::vector<std::map<int, int>> childMaps;
        std::vector<std::vector<int>> retrieveIndices;
        MNN::Express::VARP attentionMask;
        MNN::Express::VARP positionIds;
    };
    struct AcceptInfo {
        std::vector<int> sampleTokens;
        std::vector<int> acceptIndices;
        std::vector<int> acceptTokens;
    };
    MNN::Express::VARP buildAttentionMask(int draftLen, int targetLen);
    MNN::Express::VARP buildPositionIds(int start, int len);
    MNN::Express::VARP lastHidden(MNN::Express::VARP hidden);
    MNN::Express::VARP prefixHidden(MNN::Express::VARP hidden, int len);
    MNN::Express::VARP gatherHidden(MNN::Express::VARP hidden, const std::vector<int>& indices);
    DFlashDraftInfo sampleDraftWithLogits(MNN::Express::VARP targetHidden, const std::vector<int>& blockTokens);
    std::vector<int> sampleDraft(MNN::Express::VARP targetHidden, const std::vector<int>& blockTokens);
    TreeInfo buildDDTree(int anchorToken, MNN::Express::VARP logits, int startPosition, int pastLength);
    AcceptInfo evaluateTreePosterior(const TreeInfo& treeInfo, MNN::Express::VARP logits);
    bool processAcceptedTokens(const std::vector<int>& tokens, int& len, int maxToken);
    void compactTargetCache(const AcceptInfo& acceptInfo);
    int mHiddenStateIndex = -1;
    int mBlockSize = 16;
    int mMaskTokenId = -1;
    int mDDTreeTopK = 0;
    int mDDTreeBudget = 16;
    SpecContext mDFlashContext;
    std::shared_ptr<KVMeta> mDFlashMeta;
    std::shared_ptr<MNN::Express::Module> mDFlashModule;
};


class GenerationStrategyFactory {
public:
    static std::shared_ptr<Generation> create(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, bool canSpec);
};


} // namespace Transformer
} // namespace MNN
#endif
