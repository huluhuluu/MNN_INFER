//
//  mtp.cpp
//
//  Created by MNN on 2025/05/09.
//
//

#include "generate.hpp"
#include "tokentree.hpp"
#include <numeric>
#include <algorithm>

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

static VARP _gatherHiddenRows(VARP hiddenStates, const std::vector<int>& indices) {
    auto info = hiddenStates->getInfo();
    if (info == nullptr || info->dim.empty()) {
        return nullptr;
    }
    auto src = hiddenStates->readMap<float>();
    if (src == nullptr) {
        return nullptr;
    }
    int hiddenSize = info->dim.back();
    int seqLen = info->dim.size() == 3 && info->dim[0] == 1 ? info->dim[1] : info->dim[0];
    auto output = _Input({1, static_cast<int>(indices.size()), hiddenSize}, NCHW, halide_type_of<float>());
    auto dst = output->writeMap<float>();
    if (dst == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < indices.size(); ++i) {
        int index = indices[i];
        if (index < 0 || index >= seqLen) {
            return nullptr;
        }
        ::memcpy(dst + i * hiddenSize, src + index * hiddenSize, hiddenSize * sizeof(float));
    }
    return output;
}

static void _waitModuleOutputs(const std::vector<MNN::Express::VARP>& outputs) {
    for (auto& output : outputs) {
        ((MNN::Tensor*)(output->getTensor()))->wait(Tensor::MAP_TENSOR_READ, true);
    }
}

EagleGeneration::EagleGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) : Generation(llm, context) {
    // do nothing
}

void EagleGeneration::load(Module::Config module_config) {
    mEagleModuleConfig = module_config;
    mEagleMeta.reset(new KVMeta);
    mEagleBatchMeta.reset();
    mEaglePackedRootModule.reset();
    mEaglePackedModulePool.clear();
    bool packedMode = mLlm->mConfig->packed_attention();
    mLlm->mRuntimeManager->setHint(MNN::Interpreter::PACKED_ATTENTION_MODE, false);
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mEagleMeta.get());

    std::vector<std::string> inputNames{"input_embed", "hidden_states", "attention_mask", "position_ids", "logits_index"};
    std::vector<std::string> outputNames {"logits", "out_hidden_states"};
    mEagleModules.resize(2);
    mEagleModules[0].reset(Module::load(inputNames, outputNames, mLlm->mConfig->eagle_model().c_str(), mLlm->mRuntimeManager, &module_config));

    mEagleModules[1].reset(Module::load({"fc_hidden"}, {"hidden_states"}, mLlm->mConfig->eagle_fc().c_str(), mLlm->mRuntimeManager, &module_config));
    if (packedMode) {
        loadPackedDraftModule();
    }
    mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, packedMode);

    mD2t = Express::Variable::load(mLlm->mConfig->eagle_d2t().c_str())[0];

    // init
    mTopK = mLlm->mConfig->eagle_topk();
    mDepth = mLlm->mConfig->eagle_depth();
    mTreePosition = _Input({1, mTopK}, NCHW, halide_type_of<int>());
}

MNN::Express::VARP EagleGeneration::getMask(std::vector<std::vector<bool>> mask, int seqLen) {
    MNN::Express::VARP attentionMask;
    int row = static_cast<int>(mask.size());
    int col = static_cast<int>(mask[0].size());
    if (row == col) {
        attentionMask = _Input({1, 1, row, col}, NCHW, halide_type_of<float>());
        auto maskPtr  = attentionMask->writeMap<float>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                maskPtr[i * col + j] = mask[i][j] ? 0.0 : std::numeric_limits<float>::lowest();
            }
        }
    } else {
        attentionMask = _Input({1, 1, row, seqLen + col}, NCHW, halide_type_of<float>());
        auto maskPtr  = attentionMask->writeMap<float>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < seqLen; j++) {
                maskPtr[i * (seqLen + col) + j] = 0.0;
            }
            for (int j = 0; j < col; j++) {
                maskPtr[i * (seqLen + col) + seqLen + j] = mask[i][j] ? 0.0 : std::numeric_limits<float>::lowest();
            }
        }
    }
    return attentionMask;
}

void EagleGeneration::setPosition(int position) {
    auto positionPtr = mTreePosition->writeMap<int>();
    for (int i = 0; i < mTopK; i++) {
        positionPtr[i] = position;
    }
}

std::vector<MNN::Express::VARP> EagleGeneration::eagleForwardRaw(const std::vector<MNN::Express::VARP>& inputs) {
    int seq_len     = inputs[0]->getInfo()->dim[0];
    mEagleMeta->add = seq_len;
    mLlm->mRuntimeManager->setHint(MNN::Interpreter::PACKED_ATTENTION_MODE, false);
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mEagleMeta.get());
    auto outputs    = mEagleModules[0]->onForward(inputs);
    if (outputs.size() > 1) {
        _waitModuleOutputs(outputs);
    }
    mEagleMeta->sync();
    mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, mLlm->mConfig->packed_attention());
    return outputs;
}

std::vector<VARP> EagleGeneration::eagleForward(Express::VARP input_embeds, VARP hidden_states, bool all_logits) {
    int seq_len         = input_embeds->getInfo()->dim[0];
    auto attention_mask = mLlm->gen_attention_mask(seq_len);
    auto position_ids = _Input({1, seq_len}, NCHW, halide_type_of<int>());
    for (int i = 0; i < seq_len; i++) {
        position_ids->writeMap<int>()[i] = mEaglePastLen + i;
    }
    auto logits_index = all_logits ? mLlm->logitsAllIdx : mLlm->logitsLastIdx;
    std::vector<Express::VARP> inputs = {input_embeds, hidden_states, attention_mask, position_ids, logits_index};
    return eagleForwardRaw(inputs);
}

std::vector<VARP> EagleGeneration::eagleForward(const std::vector<int>& input_ids, VARP hidden_states, bool all_logits) {
    auto input_embeds = mLlm->embedding(input_ids);
    auto outputs = eagleForward(input_embeds, hidden_states, all_logits);
    return outputs;
}

EagleGeneration::DraftInfo EagleGeneration::topkGenerate(const std::vector<int>& inputIds, MNN::Express::VARP hiddenStates, MNN::Express::VARP inputEmbeds, int reqId) {
    auto d2tPtr = mD2t->readMap<int>();
    TokenTree tokenTree(mTopK, d2tPtr);
    int sampleToken = inputIds.back();
    if(inputEmbeds == nullptr) {
        inputEmbeds = mLlm->embedding(inputIds);
    }
    int seqLen       = mEaglePastLen + inputEmbeds->getInfo()->dim[0];
    auto inputHidden = mEagleModules[1]->forward(hiddenStates);
    // first token
    mEagleMeta->remove = mEagleRemove;
    auto outputs      = eagleForward(inputEmbeds, inputHidden);
    mEaglePastLen     = seqLen;
    mEagleRemove      = mTopK * (mDepth - 1);
    auto lastP        = outputs[0];
    auto lastHidden   = outputs[1];
    auto topKV = MNN::Express::_TopKV2(lastP, MNN::Express::_Scalar<int>(mTopK));
    auto scores = topKV[0]->readMap<float>();
    auto indices = topKV[1]->readMap<int>();
    tokenTree.init(indices, scores);
    inputHidden = MNN::Express::_Tile(lastHidden, _var<int>({1, mTopK, 1}, {3}));
    for (int d = 0; d < mDepth - 1; d++) {
        setPosition(seqLen + d);
        inputEmbeds   = mLlm->embedding(tokenTree.getIds());
        auto attentionMask = getMask(tokenTree.getMask(), seqLen);
        mEagleMeta->remove = 0;
        outputs = eagleForwardRaw({inputEmbeds, inputHidden, attentionMask, mTreePosition, mLlm->logitsAllIdx});
        lastP   = outputs[0];
        inputHidden  = outputs[1];
        auto topKV   = MNN::Express::_TopKV2(lastP, MNN::Express::_Scalar<int>(mTopK));
        auto scores  = topKV[0]->readMap<float>();
        auto indices = topKV[1]->readMap<int>();
        tokenTree.grow(indices, scores);
    }
    auto output = tokenTree.finalize(sampleToken, mLlm->mDraftLength);
    int inputLen = output.draftTokens.size();
    DraftInfo info;
    info.reqId = reqId;
    info.draftTokens = std::move(output.draftTokens);
    info.retrieveIndices = std::move(output.retrieveIndices);
    info.attentionMask = _Input({1, 1, inputLen, inputLen}, NCHW, halide_type_of<float>());
    for (int i = 0; i < inputLen; i++) {
        for (int j = 0; j < inputLen; j++) {
            info.attentionMask->writeMap<float>()[i * inputLen + j] = output.attentionMask[i][j] ? 0.0 : std::numeric_limits<float>::lowest();
        }
    }
    info.positionIds = _Input({1, inputLen}, NCHW, halide_type_of<int>());
    for (int i = 0; i < inputLen; i++) {
        info.positionIds->writeMap<int>()[i] = seqLen + output.positionIds[i];
    }
    return info;
}

VARPS EagleGeneration::treeDecoding(const EagleGeneration::DraftInfo& drafInfo) {
    if (mLlm->mConfig->packed_attention()) {
        return treeDecodingPacked(drafInfo);
    }
    auto inputEmbeds   = mLlm->embedding(drafInfo.draftTokens);
    int inputLen = drafInfo.draftTokens.size();
    mLlm->mMeta->add = inputLen;
    return mLlm->forwardRaw(inputEmbeds, drafInfo.attentionMask, drafInfo.positionIds);
}

EagleGeneration::AcceptInfo EagleGeneration::evaluatePosterior(const EagleGeneration::DraftInfo& drafInfo, VARP logits) {
    auto sampleTokens = MNN::Express::_ArgMax(logits, -1);
    std::vector<int> samples(drafInfo.draftTokens.size());
    ::memcpy(samples.data(), sampleTokens->readMap<int>(), samples.size() * sizeof(int));
    std::vector<int> bestCandidate;
    int nextSample = 0;
    for (auto indices : drafInfo.retrieveIndices) {
        std::vector<int> candidate;
        int next = -1;
        for (int i = 0; i < indices.size() - 1; i++) {
            int sampleIdx = indices[i];
            int draftIdx  = indices[i + 1];
            if (samples[sampleIdx] != drafInfo.draftTokens[draftIdx]) {
                break;
            }
            candidate.push_back(sampleIdx);
            next = draftIdx;
        }
        if (candidate.size() > bestCandidate.size()) {
            bestCandidate = candidate;
            nextSample = next;
        }
    }
    bestCandidate.push_back(nextSample);
    std::vector<int> acceptTokens(bestCandidate.size());
    for (int i = 0; i < bestCandidate.size(); i++) {
        acceptTokens[i] = samples[bestCandidate[i]];
    }
    AcceptInfo acceptInfo;
    acceptInfo.reqId = drafInfo.reqId;
    acceptInfo.sampleTokens  = std::move(samples);
    acceptInfo.acceptIndices = std::move(bestCandidate);
    acceptInfo.acceptTokens  = std::move(acceptTokens);
    return acceptInfo;
}

EagleGeneration::DraftInfo EagleGeneration::updateDraft(const AcceptInfo& acceptInfo, VARP hiddenStates) {
    int acceptLen = static_cast<int>(acceptInfo.acceptTokens.size());
    // update base model kv cache
    {
        mLlm->updateContext(acceptLen, acceptLen);
        if (mLlm->mConfig->packed_attention()) {
            updatePackedBaseKV(acceptInfo);
        } else {
            mLlm->mMeta->remove = acceptInfo.sampleTokens.size();
            mLlm->mMeta->n_reserve = acceptLen;
            mLlm->mMeta->reserveHost.resize(acceptLen * 2);
            mLlm->mMeta->reserve = mLlm->mMeta->reserveHost.data();
            for (int i = 0; i < acceptLen; i++) {
                mLlm->mMeta->reserve[2 * i] = acceptInfo.acceptIndices[i];
                mLlm->mMeta->reserve[2 * i + 1] = 1;
            }
        }
    }
    auto acceptHiddenState = _gatherHiddenRows(hiddenStates, acceptInfo.acceptIndices);
    if (acceptHiddenState == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    return topkGenerate(acceptInfo.acceptTokens, acceptHiddenState, nullptr, acceptInfo.reqId);
}

bool EagleGeneration::processTokens(const std::vector<int>& acceptTokens) {
    for (int i = 0; i < acceptTokens.size(); i++) {
        auto token = acceptTokens[i];
        if (mLlm->is_stop(token)) {
            return true;
        }
        if (nullptr != mContext->os) {
            auto tokenStr = mLlm->tokenizer_decode(token);
            *mContext->os << tokenStr << std::flush;
        }
    }
    return false;
}

void EagleGeneration::generate(GenerationParams& param) {
    int reqId = param.reqId;
    mEaglePastLen = 0;
    mEagleRemove  = mEagleMeta->previous;
    MNN::Timer _t;
    VARP inputEmbeds  = param.input_embeds;
    auto inputIds     = param.input_ids;
    auto sampleToken  = mLlm->sample(param.outputs[0]);
    mContext->current_token = sampleToken;
    mContext->history_tokens.push_back(mContext->current_token);
    mContext->output_tokens.push_back(mContext->current_token);
    mLlm->updateContext(0, 1);
    if (nullptr != mContext->os) {
        *mContext->os << mLlm->tokenizer_decode(sampleToken) << std::flush;
    }
    inputIds.push_back(sampleToken);
    VARP hiddenStates = param.outputs[1];
    // push sampleToken to inputEmbeds
    int seqLen      = inputEmbeds->getInfo()->dim[0];
    auto cur_embed  = mLlm->embedding({sampleToken});
    auto pre_embeds = _Split(inputEmbeds, {1, seqLen - 1}, 0);
    inputEmbeds     = _Concat({pre_embeds[1], cur_embed}, 0);
    // eagle generate
    auto draftInfo  = topkGenerate(inputIds, hiddenStates, inputEmbeds, reqId);
    auto newTokens = 0, steps = 0;
    while (true) {
        if(mContext->status == LlmStatus::USER_CANCEL) {
            break;
        }
        steps++;
        auto decodingInfo = treeDecoding(draftInfo);
        for (auto o : decodingInfo) {
            if(nullptr == o->readMap<float>()) {
                mContext->status = LlmStatus::INTERNAL_ERROR;
                break;
            }
        }
        if(decodingInfo.empty()) {
            break;
        }
        
        auto acceptInfo = evaluatePosterior(draftInfo, decodingInfo[0]);
        newTokens += acceptInfo.acceptTokens.size();
        {
            mContext->current_token = acceptInfo.acceptTokens.back();
            for (auto token : acceptInfo.acceptTokens) {
                mContext->history_tokens.push_back(token);
                mContext->output_tokens.push_back(token);
            }
        }
        bool stop = processTokens(acceptInfo.acceptTokens);
        if (stop || newTokens >= param.max_new_tokens) {
            mContext->output_tokens.push_back(steps);
            break;
        }
        draftInfo = updateDraft(acceptInfo, decodingInfo[1]);
    }
    mContext->decode_us += _t.durationInUs();
    if(newTokens >= param.max_new_tokens) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }
    mBasePendingKV.erase(reqId);
    return;
}

} // namespace Transformer
} // namespace MNN
