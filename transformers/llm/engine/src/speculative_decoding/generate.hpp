//
//  generate.hpp
//
//  Created by MNN on 2025/06/09.
//

#ifndef SPEC_GENERATE_HPP
#define SPEC_GENERATE_HPP

#include <MNN/AutoTime.hpp>
#include <array>
#include <map>
#include <ostream>
#include "llm/llm.hpp"
#include "../llmconfig.hpp"
#include "../kvmeta.hpp"

//#define DUMP_PROFILE_INFO

namespace MNN {
namespace Transformer {
struct SpecContext {
    uint64_t draft = 0;
    uint64_t accepted = 0;
    uint64_t steps = 0;
    std::map<int, uint64_t> accept_len_freq;
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
    double avgAcceptLen() const {
        return steps == 0 ? 0.0 : static_cast<double>(accepted) / steps;
    }
    double acceptRate() const {
        return draft == 0 ? 0.0 : static_cast<double>(accepted) / draft;
    }
    double compressionRatio() const {
        return accepted == 0 ? 0.0 : static_cast<double>(draft) / accepted;
    }
    double avgDraftTimeMs() const {
        return steps == 0 ? 0.0 : static_cast<double>(draft_time_us) / 1000.0 / steps;
    }
    double avgTargetTimeMs() const {
        return steps == 0 ? 0.0 : static_cast<double>(target_time_us) / 1000.0 / steps;
    }
    double theoreticalSpeedup() const {
        if (accepted == 0 || steps == 0 || draft_time_us + target_time_us == 0) {
            return 0.0;
        }
        const double targetTimePerStep = static_cast<double>(target_time_us) / steps;
        return accepted * targetTimePerStep / (draft_time_us + target_time_us);
    }
};

struct GenerationParams {
    int max_new_tokens;
    int reqId = 0;
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
    virtual void prepare() {}
    virtual bool prefill(const std::vector<int>& inputIds, MNN::Express::VARP hiddenStates) {
        return true;
    }
    virtual void generate(GenerationParams& param) = 0;
    virtual std::vector<std::vector<int>> generateBatch(const std::vector<std::vector<int>>& inputIds, std::ostream* os, int maxNewTokens);
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
    virtual void generate(GenerationParams& param) override;
    virtual std::vector<std::vector<int>> generateBatch(const std::vector<std::vector<int>>& inputIds, std::ostream* os, int maxNewTokens) override;
};

class LookaheadGeneration: public Generation {
public:
    LookaheadGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config);
    virtual ~LookaheadGeneration() = default;
    virtual void generate(GenerationParams& param) override;
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
    virtual void prepare() override;
    virtual bool prefill(const std::vector<int>& inputIds, MNN::Express::VARP hiddenStates) override;
    virtual void generate(GenerationParams& param) override;
    virtual std::vector<std::vector<int>> generateBatch(const std::vector<std::vector<int>>& inputIds, std::ostream* os, int maxNewTokens) override;
    SpecContext* getSpecContext() override { return &mSpecContext; }
    const SpecContext* getSpecContext() const override { return &mSpecContext; }
    void resetSpecContext() override { mSpecContext.reset(); }
private:
    struct DraftInfo {
        int reqId = 0;
        int pipelineId = -1;
        std::vector<int> draftTokens;
        std::vector<std::vector<int>> retrieveIndices;
        VARP attentionMask;
        VARP positionIds;
    };
    struct AcceptInfo {
        int reqId = 0;
        std::vector<int> sampleTokens;
        std::vector<int> acceptIndices;
        std::vector<int> acceptTokens;
    };
    struct PendingBaseKV {
        size_t remove = 0;
        std::vector<int> reserveHost;
    };
    struct EagleState {
        int pastLen = 0;
        int remove = 0;
    };
    struct PackedDraftInput {
        int reqId = 0;
        int pipelineId = -1;
        EagleState* state = nullptr;
        std::vector<int> inputIds;
        MNN::Express::VARP hiddenStates;
        MNN::Express::VARP inputEmbeds;
    };
    struct PackedDraftKVInfo {
        int reqId = 0;
        size_t add = 0;
        size_t remove = 0;
    };
    MNN::Express::VARPS eagleForwardRaw(const MNN::Express::VARPS& inputs);
    static MNN::Express::VARP gatherHiddenRows(MNN::Express::VARP hiddenStates, const std::vector<int>& indices);
    static void waitModuleOutputs(const MNN::Express::VARPS& outputs);
    MNN::Express::VARPS eagleForward(const std::vector<int>& inputEmbeds, MNN::Express::VARP hiddenStates, bool allLogits = false);
    MNN::Express::VARPS eagleForward(MNN::Express::VARP inputEmbeds, MNN::Express::VARP hiddenStates, bool allLogits = false);
    void loadPackedDraftModule();
    MNN::Express::VARPS eagleForwardRawPacked(const std::vector<PackedDraftKVInfo>& kvInfos, const MNN::Express::VARPS& inputs, bool waitAllOutputs = true, int pipelineId = -1);
    MNN::Express::VARPS runDualPipelineComponent(int pipelineId,
                                                 const std::shared_ptr<MNN::Express::Module>& module,
                                                 const MNN::Express::VARPS& inputs,
                                                 const GraphSnapshot& graphSnapshot,
                                                 const std::vector<DualPipelineScheduler::GraphRequest>& graphRequests,
                                                 const std::unordered_map<std::string, int>& qnnOpIndices,
                                                 const std::vector<int>& ownerReqIds);
    void loadDualPipelineGraphInfo();
    MNN::Express::VARP eagleFCForward(const MNN::Express::VARPS& hiddenStates, int pipelineId = -1);
    DraftInfo topkGenerate(const std::vector<int>& inputIds, MNN::Express::VARP hiddenStates, MNN::Express::VARP inputEmbeds = nullptr, int reqId = 0);
    std::vector<DraftInfo> topkGeneratePacked(const std::vector<PackedDraftInput>& inputs, int pipelineId = -1);
    bool prefillDraftPacked(const std::vector<PackedDraftInput>& inputs, int pipelineId = -1);
    VARPS treeDecoding(const DraftInfo& draftInfo);
    VARPS treeDecodingPacked(const DraftInfo& draftInfo);
    AcceptInfo evaluatePosterior(const DraftInfo& drafInfo, VARP logits);
    DraftInfo updateDraft(const AcceptInfo& accpetInfo, VARP hiddenStates);
    void updatePackedBaseKV(const AcceptInfo& acceptInfo);
    MNN::Express::VARP getMask(std::vector<std::vector<bool>> mask, int seqLen);
    MNN::Express::VARP getPackedMask(const std::vector<DraftInfo>& draftInfos);
    bool processTokens(const std::vector<int>& accpetTokens);
    void setPosition(int position);
    std::vector<std::shared_ptr<MNN::Express::Module>> mEagleModules;
    Module::Config mEagleModuleConfig;
    std::shared_ptr<KVMeta> mEagleMeta;
    std::shared_ptr<BatchKVMeta> mEagleBatchMeta;
    std::shared_ptr<MNN::Express::Module> mEaglePackedRootModule;
    std::shared_ptr<MNN::Express::Module> mEaglePackedCacheOwner;
    std::map<std::pair<int, int>, std::shared_ptr<MNN::Express::Module>> mEaglePackedModulePool;
    struct PackedDraftRuntime {
        std::shared_ptr<BatchKVMeta> batchMeta;
        std::shared_ptr<MNN::Express::Module> rootModule;
        std::shared_ptr<MNN::Express::Module> cacheOwner;
        std::shared_ptr<MNN::Express::Module> fcModule;
        std::map<std::pair<int, int>, std::shared_ptr<MNN::Express::Module>> modulePool;
    };
    std::array<PackedDraftRuntime, 2> mEagleDualRuntimes;
    GraphSnapshot mEagleDraftGraphSnapshot;
    GraphSnapshot mEagleFCGraphSnapshot;
    std::vector<DualPipelineScheduler::GraphRequest> mEagleDraftGraphRequests;
    std::vector<DualPipelineScheduler::GraphRequest> mEagleFCGraphRequests;
    std::unordered_map<std::string, int> mEagleDraftQnnOpIndices;
    std::unordered_map<std::string, int> mEagleFCQnnOpIndices;
    MNN::Express::VARP mD2t, mTreePosition;
    int mTopK, mDepth;
    int mEaglePastLen = 0, mEagleRemove = 0;
    bool mEagleRequestPrepared = false;
    std::map<int, PendingBaseKV> mBasePendingKV;
    SpecContext mSpecContext;
};


class GenerationStrategyFactory {
public:
    static std::shared_ptr<Generation> create(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, bool canSpec);
};


} // namespace Transformer
} // namespace MNN
#endif
