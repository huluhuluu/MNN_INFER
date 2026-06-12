//
//  dflash.cpp
//

#include "generate.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

DFlashGeneration::DFlashGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) : Generation(llm, context) {
    mBlockSize = config->dflash_block_size();
    mMaskTokenId = config->dflash_mask_token_id();
}

void DFlashGeneration::load(Module::Config module_config) {
    mDFlashMeta.reset(new KVMeta);
    auto targetLayerIds = mLlm->mConfig->dflash_target_layer_ids();
    mDFlashMeta->layer_nums = static_cast<int>(targetLayerIds.size());
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mDFlashMeta.get());

    std::vector<std::string> inputNames{"noise_embedding", "target_hidden", "attention_mask", "position_ids"};
    std::vector<std::string> outputNames{"logits"};
    mDFlashModule.reset(Module::load(inputNames, outputNames, mLlm->mConfig->dflash_model().c_str(), mLlm->mRuntimeManager, &module_config));
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mLlm->mMeta.get());
    mHiddenStateIndex = mLlm->getOutputIndex("hidden_states");
    if (mHiddenStateIndex < 0) {
        MNN_ERROR("DFlash requires hidden_states output from target model.\n");
    }
    if (mMaskTokenId < 0) {
        MNN_ERROR("DFlash requires dflash_mask_token_id in config.\n");
    }
}

VARP DFlashGeneration::buildAttentionMask(int draftLen, int targetLen) {
    int queryLen = draftLen;
    int kvLen = targetLen + draftLen;
    auto attentionMask = _Input({1, 1, queryLen, kvLen}, NCHW, halide_type_of<float>());
    auto ptr = attentionMask->writeMap<float>();
    for (int i = 0; i < queryLen * kvLen; i++) {
        ptr[i] = 0.0f;
    }
    return attentionMask;
}

VARP DFlashGeneration::buildPositionIds(int start, int len) {
    auto positionIds = _Input({1, len}, NCHW, halide_type_of<int>());
    auto ptr = positionIds->writeMap<int>();
    for (int i = 0; i < len; i++) {
        ptr[i] = start + i;
    }
    return positionIds;
}

std::vector<int> DFlashGeneration::sampleDraft(VARP targetHidden, const std::vector<int>& blockTokens) {
    if (mDFlashModule == nullptr || blockTokens.size() <= 1) {
        return {};
    }
    std::vector<int> draftTokensInput(blockTokens.begin(), blockTokens.end());
    int draftLen = static_cast<int>(draftTokensInput.size());
    int targetLen = targetHidden->getInfo()->dim[0];
    auto noiseEmbedding = mLlm->embedding(draftTokensInput);
    auto attentionMask = buildAttentionMask(draftLen, targetLen);
    auto positionIds = buildPositionIds(0, targetLen + draftLen);
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mDFlashMeta.get());
    auto outputs = mDFlashModule->onForward({noiseEmbedding, targetHidden, attentionMask, positionIds});
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mLlm->mMeta.get());
    if (outputs.empty()) {
        return {};
    }
    auto logits = outputs[0];
    int sampleSize = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
    std::vector<int> draftTokens(blockTokens.size() - 1);
    for (int i = 1; i < static_cast<int>(blockTokens.size()); i++) {
        draftTokens[i - 1] = mLlm->sample(logits, i * sampleSize, sampleSize);
    }
    return draftTokens;
}

VARP DFlashGeneration::lastHidden(VARP hidden) {
    auto dims = hidden->getInfo()->dim;
    int targetLen = dims[0] == 1 && dims.size() > 1 ? dims[1] : dims[0];
    int hiddenSize = dims[dims.size() - 1];
    auto tail = _Input({1, 1, hiddenSize}, NCHW, halide_type_of<float>());
    ::memcpy(tail->writeMap<float>(), hidden->readMap<float>() + (targetLen - 1) * hiddenSize, hiddenSize * sizeof(float));
    return tail;
}

VARP DFlashGeneration::prefixHidden(VARP hidden, int len) {
    auto dims = hidden->getInfo()->dim;
    int seqLen = dims[0] == 1 && dims.size() > 1 ? dims[1] : dims[0];
    int hiddenSize = dims[dims.size() - 1];
    len = std::min(len, seqLen);
    auto prefix = _Input({len, 1, hiddenSize}, NCHW, halide_type_of<float>());
    ::memcpy(prefix->writeMap<float>(), hidden->readMap<float>(), len * hiddenSize * sizeof(float));
    return prefix;
}

void DFlashGeneration::generate(GenerationParams& param) {
    if (mHiddenStateIndex < 0 || mDFlashModule == nullptr || mMaskTokenId < 0) {
        ArGeneration(mLlm, mContext, mLlm->mConfig).generate(param);
        return;
    }

    int maxToken = param.max_new_tokens;
    int len = 0;
    bool draftEmptyWarned = false;
    bool invalidVerifyOutputWarned = false;
    auto targetHidden = prefixHidden(param.outputs[mHiddenStateIndex],
                                     param.outputs[mHiddenStateIndex]->getInfo()->dim[0] == 1 ?
                                     param.outputs[mHiddenStateIndex]->getInfo()->dim[1] :
                                     param.outputs[mHiddenStateIndex]->getInfo()->dim[0]);
    if (mDFlashMeta != nullptr) {
        mDFlashMeta->remove = mDFlashMeta->previous;
        mDFlashMeta->add = 0;
    }
    int sampleToken = mLlm->sample(param.outputs[0], param.validLogitStart, param.validLogitSize);
    mContext->current_token = sampleToken;

    while (len < maxToken) {
        if (mContext->status == LlmStatus::USER_CANCEL) {
            break;
        }

        if (mLlm->is_stop(mContext->current_token)) {
            mContext->history_tokens.push_back(mContext->current_token);
            mContext->output_tokens.push_back(mContext->current_token);
            mLlm->updateContext(0, 1);
            if (nullptr != mContext->os) {
                *mContext->os << mContext->end_with << std::flush;
            }
            break;
        }

        MNN::Timer _t;
        std::vector<int> blockTokens(mBlockSize, mMaskTokenId);
        blockTokens[0] = mContext->current_token;
        MNN::Timer draftTimer;
        auto draftTokens = sampleDraft(targetHidden, blockTokens);
        auto draftUs = draftTimer.durationInUs();
        if (draftTokens.empty()) {
            if (!draftEmptyWarned) {
                MNN_PRINT("Warning: DFlash draft returned empty tokens, fallback to autoregressive decoding for this step.\n");
                draftEmptyWarned = true;
            }
            auto outputs = mLlm->forwardVec({mContext->current_token});
            if (outputs.empty()) {
                break;
            }
            mLlm->updateContext(1, 1);
            mContext->history_tokens.push_back(mContext->current_token);
            mContext->output_tokens.push_back(mContext->current_token);
            auto tokenStr = mLlm->tokenizer_decode(mContext->current_token);
            mContext->generate_str += tokenStr;
            if (nullptr != mContext->os) {
                *mContext->os << tokenStr << std::flush;
            }
            mContext->current_token = mLlm->sample(outputs[0]);
            targetHidden = _Concat({targetHidden, lastHidden(outputs[mHiddenStateIndex])}, 0);
            len++;
            mContext->decode_us += _t.durationInUs();
            continue;
        }
        std::copy(draftTokens.begin(), draftTokens.end(), blockTokens.begin() + 1);

        MNN::Timer targetTimer;
        int savedGenSeqLen = mContext->gen_seq_len;
        if (savedGenSeqLen == 0) {
            mContext->gen_seq_len = 1;
        }
        auto outputs = mLlm->forwardVec(blockTokens);
        mContext->gen_seq_len = savedGenSeqLen;
        auto targetUs = targetTimer.durationInUs();
        if (outputs.size() <= mHiddenStateIndex || outputs[0]->getInfo()->size == 0) {
            if (!invalidVerifyOutputWarned) {
                MNN_PRINT("Warning: DFlash target verification output is invalid, stop DFlash generation.\n");
                invalidVerifyOutputWarned = true;
            }
            break;
        }

        auto logits = outputs[0];
        int sampleSize = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
        std::vector<int> posterior(blockTokens.size());
        for (int i = 0; i < blockTokens.size(); i++) {
            posterior[i] = mLlm->sample(logits, i * sampleSize, sampleSize);
        }

        int acceptLen = 1;
        while (acceptLen < blockTokens.size() && posterior[acceptLen - 1] == blockTokens[acceptLen]) {
            acceptLen++;
        }
        int emitLen = std::min(acceptLen, maxToken - len);
        bool stop = false;
        for (int i = 0; i < emitLen; i++, len++) {
            int token = blockTokens[i];
            mContext->history_tokens.push_back(token);
            mContext->output_tokens.push_back(token);
            auto tokenStr = mLlm->tokenizer_decode(token);
            mContext->generate_str += tokenStr;
            if (nullptr != mContext->os) {
                *mContext->os << tokenStr << std::flush;
            }
            if (mLlm->is_stop(token)) {
                emitLen = i + 1;
                stop = true;
                break;
            }
        }

        int removeLen = static_cast<int>(blockTokens.size()) - emitLen;
        mLlm->mMeta->remove = removeLen;
        mLlm->updateContext(emitLen, emitLen);
        auto acceptHidden = prefixHidden(outputs[mHiddenStateIndex], emitLen);
        targetHidden = _Concat({targetHidden, acceptHidden}, 0);
        mDFlashContext.steps += 1;
        mDFlashContext.draft += blockTokens.size();
        mDFlashContext.accepted += emitLen;
        mDFlashContext.accept_len_freq[emitLen] += 1;
        mDFlashContext.draft_time_us += draftUs;
        mDFlashContext.target_time_us += targetUs;
        mContext->decode_us += _t.durationInUs();

        if (stop) {
            if (nullptr != mContext->os) {
                *mContext->os << mContext->end_with << std::flush;
            }
            return;
        }
        if (emitLen < acceptLen) {
            break;
        }
        mContext->current_token = posterior[acceptLen - 1];
    }

    if(len >= maxToken) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }
}

} // namespace Transformer
} // namespace MNN
