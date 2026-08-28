//
//  mtp.cpp
//
//  Created by MNN on 2025/05/09.
//
//

#include "generate.hpp"
#include "tokentree.hpp"
#include <cstring>
#include <limits>

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

static int _rowCount(VARP input) {
    auto info = input == nullptr ? nullptr : input->getInfo();
    if (info == nullptr || info->dim.empty()) {
        return 0;
    }
    if (info->dim.size() == 3 && info->dim[0] == 1) {
        return info->dim[1];
    }
    return info->dim[0];
}

static VARP _sliceRows(VARP input, int offset, int len) {
    auto info = input == nullptr ? nullptr : input->getInfo();
    if (info == nullptr || info->dim.empty() || offset < 0 || len <= 0 || offset + len > _rowCount(input)) {
        return nullptr;
    }
    if (info->dim.size() == 3) {
        if (info->dim[0] == 1) {
            return _Slice(input, _var<int>({0, offset, 0}, {3}), _var<int>({1, len, -1}, {3}));
        }
        return _Slice(input, _var<int>({offset, 0, 0}, {3}), _var<int>({len, 1, -1}, {3}));
    }
    return _Slice(input, _var<int>({offset, 0}, {2}), _var<int>({len, -1}, {2}));
}

static VARP _padFloatRows(VARP input, int paddedLen) {
    auto info = input == nullptr ? nullptr : input->getInfo();
    const int actualLen = _rowCount(input);
    if (info == nullptr || actualLen <= 0 || paddedLen < actualLen) {
        return nullptr;
    }
    const bool needsLayoutNormalization = info->dim.size() == 3 && info->dim[0] == 1;
    if (paddedLen == actualLen && !needsLayoutNormalization) {
        return input;
    }
    std::vector<int> dims = info->dim;
    if (dims.size() == 3) {
        dims[0] = paddedLen;
        dims[1] = 1;
    } else {
        dims[0] = paddedLen;
    }
    auto output = _Input(dims, NCHW, halide_type_of<float>());
    auto src = input->readMap<float>();
    auto dst = output->writeMap<float>();
    if (src == nullptr || dst == nullptr) {
        return nullptr;
    }
    ::memset(dst, 0, output->getInfo()->size * sizeof(float));
    ::memcpy(dst, src, info->size * sizeof(float));
    return output;
}

static VARP _padIntRows(VARP input, int paddedLen) {
    auto info = input == nullptr ? nullptr : input->getInfo();
    const int actualLen = info == nullptr ? 0 : info->size;
    if (info == nullptr || actualLen <= 0 || paddedLen <= actualLen) {
        return input;
    }
    std::vector<int> dims = info->dim;
    if (dims.size() > 1 && dims[0] == 1) {
        dims[1] = paddedLen;
    } else {
        dims[0] = paddedLen;
    }
    auto output = _Input(dims, NCHW, halide_type_of<int>());
    auto src = input->readMap<int>();
    auto dst = output->writeMap<int>();
    if (src == nullptr || dst == nullptr) {
        return nullptr;
    }
    ::memset(dst, 0, output->getInfo()->size * sizeof(int));
    ::memcpy(dst, src, info->size * sizeof(int));
    return output;
}

static VARP _padTreeMask(VARP input, int actualLen, int paddedLen) {
    auto info = input == nullptr ? nullptr : input->getInfo();
    if (info == nullptr || info->dim.size() != 4 || info->dim[2] != actualLen ||
        info->dim[3] != actualLen || paddedLen < actualLen) {
        return nullptr;
    }
    if (paddedLen == actualLen) {
        return input;
    }
    auto output = _Input({1, 1, paddedLen, paddedLen}, NCHW, halide_type_of<float>());
    auto src = input->readMap<float>();
    auto dst = output->writeMap<float>();
    if (src == nullptr || dst == nullptr) {
        return nullptr;
    }
    std::fill(dst, dst + output->getInfo()->size, std::numeric_limits<float>::lowest());
    for (int row = 0; row < actualLen; ++row) {
        ::memcpy(dst + row * paddedLen, src + row * actualLen, actualLen * sizeof(float));
    }
    return output;
}

VARP EagleGeneration::gatherHiddenRows(VARP hiddenStates, const std::vector<int>& indices) {
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

void EagleGeneration::waitModuleOutputs(const std::vector<MNN::Express::VARP>& outputs) {
    for (auto& output : outputs) {
        ((MNN::Tensor*)(output->getTensor()))->wait(Tensor::MAP_TENSOR_READ, true);
    }
}

EagleGeneration::EagleGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) : Generation(llm, context) {
    // do nothing
}

void EagleGeneration::load(Module::Config module_config) {
    mEagleModuleConfig = module_config;
    loadDualPipelineGraphInfo();
    mEagleMeta.reset(new KVMeta);
    mEagleBatchMeta.reset();
    mEaglePackedRootModule.reset();
    mEaglePackedModulePool.clear();
    for (auto& runtime : mEagleDualRuntimes) {
        runtime.batchMeta.reset();
        runtime.rootModule.reset();
        runtime.cacheOwner.reset();
        runtime.fcModule.reset();
        runtime.modulePool.clear();
    }
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
    mTreePosition = _Input({mTopK}, NCHW, halide_type_of<int>());
}

void EagleGeneration::prepare() {
    mEaglePastLen = 0;
    mEagleRemove = mEagleMeta == nullptr ? 0 : mEagleMeta->previous;
    mEagleRequestPrepared = true;
}

bool EagleGeneration::prefill(const std::vector<int>& inputIds, VARP hiddenStates) {
    if (!mEagleRequestPrepared || inputIds.empty() ||
        _rowCount(hiddenStates) != static_cast<int>(inputIds.size())) {
        return false;
    }
    MNN::Timer timer;
    auto inputHidden = eagleFCForward({hiddenStates});
    if (inputHidden == nullptr) {
        return false;
    }
    mEagleMeta->remove = mEagleRemove;
    auto outputs = eagleForward(inputIds, inputHidden);
    if (outputs.size() < 2 || outputs[0] == nullptr || outputs[1] == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    mEaglePastLen += static_cast<int>(inputIds.size());
    mEagleRemove = 0;
    const uint64_t elapsedUs = timer.durationInUs();
    mSpecContext.draft_time_us += elapsedUs;
    mSpecContext.draft_prefill_time_us += elapsedUs;
    return true;
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
    if (inputs.size() < 5) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    const int actualLen = _rowCount(inputs[0]);
    const int requiredLen = std::max(actualLen, _rowCount(inputs[1]));
    const int paddedLen = selectQnnCompatibleBucketSize(mEagleDraftGraphSnapshot, requiredLen);
    if (actualLen <= 0 || paddedLen < requiredLen) {
        MNN_ERROR("MNN_QNN: no Eagle draft graph bucket can hold single length %d "
                  "(embed rows %d, hidden rows %d, draft past %d, target block %d).\n",
                  requiredLen, actualLen, _rowCount(inputs[1]), mEaglePastLen, mLlm->mBlockSize);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    VARPS paddedInputs = inputs;
    paddedInputs[0] = _padFloatRows(inputs[0], paddedLen);
    paddedInputs[1] = _padFloatRows(inputs[1], paddedLen);
    paddedInputs[3] = _padIntRows(inputs[3], paddedLen);
    if (paddedInputs[0] == nullptr || paddedInputs[1] == nullptr || paddedInputs[3] == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    const int* logitsIndex = inputs[4]->readMap<int>();
    const bool allLogits = logitsIndex != nullptr && logitsIndex[0] == 0;
    if (paddedLen > actualLen) {
        paddedInputs[4] = mLlm->logitsAllIdx;
    }
    mEagleMeta->add = actualLen;
    mLlm->mRuntimeManager->setHint(MNN::Interpreter::PACKED_ATTENTION_MODE, false);
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mEagleMeta.get());
    auto outputs = mEagleModules[0]->onForward(paddedInputs);
    if (outputs.size() > 1) {
        waitModuleOutputs(outputs);
    }
    mEagleMeta->sync();
    mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, mLlm->mConfig->packed_attention());
    if (paddedLen > actualLen && outputs.size() > 1) {
        outputs[0] = _sliceRows(outputs[0], allLogits ? 0 : actualLen - 1, allLogits ? actualLen : 1);
        outputs[1] = _sliceRows(outputs[1], 0, actualLen);
        if (outputs[0] == nullptr || outputs[1] == nullptr) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return {};
        }
    }
    return outputs;
}

std::vector<VARP> EagleGeneration::eagleForward(Express::VARP input_embeds, VARP hidden_states, bool all_logits) {
    int seq_len         = input_embeds->getInfo()->dim[0];
    auto attention_mask = mLlm->gen_attention_mask(seq_len);
    auto position_ids = _Input({seq_len}, NCHW, halide_type_of<int>());
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
    auto inputHidden = eagleFCForward({hiddenStates});
    if (inputHidden == nullptr) {
        return {};
    }
    // first token
    mEagleMeta->remove = mEagleRemove;
    auto outputs      = eagleForward(inputEmbeds, inputHidden);
    if (outputs.size() < 2 || outputs[0] == nullptr || outputs[1] == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    mEaglePastLen     = seqLen;
    mEagleRemove      = mTopK * (mDepth - 1);
    auto lastP        = outputs[0];
    auto lastHidden = _sliceRows(outputs[1], _rowCount(outputs[1]) - 1, 1);
    if (lastHidden == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
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
        if (outputs.size() < 2 || outputs[0] == nullptr || outputs[1] == nullptr) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return {};
        }
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
    info.positionIds = _Input({inputLen}, NCHW, halide_type_of<int>());
    for (int i = 0; i < inputLen; i++) {
        info.positionIds->writeMap<int>()[i] = seqLen + output.positionIds[i];
    }
    return info;
}

VARPS EagleGeneration::treeDecoding(const EagleGeneration::DraftInfo& drafInfo) {
    if (mLlm->mConfig->packed_attention()) {
        return treeDecodingPacked(drafInfo);
    }
    int inputLen = drafInfo.draftTokens.size();
    const int paddedLen = mLlm->qnnPaddedCulLen(inputLen);
    if (paddedLen < inputLen) {
        MNN_ERROR("MNN_QNN: no target graph bucket can hold Eagle single tree length %d.\n", inputLen);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    auto inputEmbeds = _padFloatRows(mLlm->embedding(drafInfo.draftTokens), paddedLen);
    auto positionIds = _padIntRows(drafInfo.positionIds, paddedLen);
    auto attentionMask = _padTreeMask(drafInfo.attentionMask, inputLen, paddedLen);
    if (inputEmbeds == nullptr || positionIds == nullptr || attentionMask == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    mLlm->mMeta->add = inputLen;
    auto outputs = mLlm->forwardRaw(inputEmbeds, attentionMask, positionIds);
    if (paddedLen > inputLen && outputs.size() > 1) {
        outputs[0] = _sliceRows(outputs[0], 0, inputLen);
        outputs[1] = _sliceRows(outputs[1], 0, inputLen);
    }
    return outputs;
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

void EagleGeneration::commitAcceptedTokens(const AcceptInfo& acceptInfo) {
    int acceptLen = static_cast<int>(acceptInfo.acceptTokens.size());
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

EagleGeneration::DraftInfo EagleGeneration::updateDraft(const AcceptInfo& acceptInfo, VARP hiddenStates) {
    auto acceptHiddenState = gatherHiddenRows(hiddenStates, acceptInfo.acceptIndices);
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
    if (!mEagleRequestPrepared) {
        prepare();
    }
    mEagleRequestPrepared = false;
    MNN::Timer _t;
    VARP inputEmbeds  = param.input_embeds;
    auto inputIds     = param.input_ids;
    if (mLlm->cancelRequested()) {
        mBasePendingKV.erase(reqId);
        return;
    }
    auto sampleToken  = mLlm->sample(param.outputs[0], param.validLogitStart, param.validLogitSize);
    mContext->current_token = sampleToken;
    mContext->history_tokens.push_back(mContext->current_token);
    mContext->output_tokens.push_back(mContext->current_token);
    mLlm->updateContext(0, 1);
    const bool firstTokenStops = mLlm->is_stop(sampleToken);
    if (firstTokenStops && nullptr != mContext->os) {
        *mContext->os << mContext->end_with << std::flush;
    } else if (nullptr != mContext->os) {
        *mContext->os << mLlm->tokenizer_decode(sampleToken) << std::flush;
    }
    int newTokens = 1;
    if (firstTokenStops || newTokens >= param.max_new_tokens) {
        if (!firstTokenStops) {
            mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
        }
        mContext->decode_us += _t.durationInUs();
        mBasePendingKV.erase(reqId);
        return;
    }
    inputIds.push_back(sampleToken);
    VARP hiddenStates = param.outputs[1];
    if (param.validLogitSize > 0) {
        const int actualLen = param.validLogitStart / param.validLogitSize + 1;
        inputEmbeds = _sliceRows(inputEmbeds, 0, actualLen);
        hiddenStates = _sliceRows(hiddenStates, 0, actualLen);
        if (inputEmbeds == nullptr || hiddenStates == nullptr) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return;
        }
    }
    // push sampleToken to inputEmbeds
    int seqLen      = inputEmbeds->getInfo()->dim[0];
    auto cur_embed  = mLlm->embedding({sampleToken});
    auto pre_embeds = _Split(inputEmbeds, {1, seqLen - 1}, 0);
    inputEmbeds     = _Concat({pre_embeds[1], cur_embed}, 0);
    // eagle generate
    MNN::Timer draftTimer;
    auto draftInfo  = topkGenerate(inputIds, hiddenStates, inputEmbeds, reqId);
    const uint64_t draftPrefillUs = draftTimer.durationInUs();
    mSpecContext.draft_time_us += draftPrefillUs;
    mSpecContext.draft_prefill_time_us += draftPrefillUs;
    mSpecContext.draft += draftInfo.draftTokens.size();
    if (draftInfo.draftTokens.empty() || mContext->status == LlmStatus::INTERNAL_ERROR) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return;
    }
    while (true) {
        if (mLlm->cancelRequested()) {
            break;
        }
        mSpecContext.steps++;
        MNN::Timer targetTimer;
        auto decodingInfo = treeDecoding(draftInfo);
          if (decodingInfo.size() < 2 || decodingInfo[0] == nullptr || decodingInfo[1] == nullptr ||
              decodingInfo[0]->readMap<float>() == nullptr || decodingInfo[1]->readMap<float>() == nullptr) {
              mContext->status = LlmStatus::INTERNAL_ERROR;
              break;
          }
          if (mLlm->cancelRequested()) {
              break;
          }

          auto acceptInfo = evaluatePosterior(draftInfo, decodingInfo[0]);
        int acceptLimit = 0;
        bool stop = false;
        for (int token : acceptInfo.acceptTokens) {
            ++acceptLimit;
            if (mLlm->is_stop(token)) {
                stop = true;
                break;
            }
            if (newTokens + acceptLimit >= param.max_new_tokens) {
                break;
            }
        }
        if (acceptLimit < static_cast<int>(acceptInfo.acceptTokens.size())) {
            acceptInfo.acceptTokens.resize(acceptLimit);
            acceptInfo.acceptIndices.resize(acceptLimit);
        }
        mSpecContext.target_time_us += targetTimer.durationInUs();
        const int acceptLen = static_cast<int>(acceptInfo.acceptTokens.size());
        mSpecContext.accepted += acceptLen;
        mSpecContext.accept_len_freq[acceptLen]++;
        newTokens += acceptLen;
        {
            mContext->current_token = acceptInfo.acceptTokens.back();
            for (auto token : acceptInfo.acceptTokens) {
                mContext->history_tokens.push_back(token);
                mContext->output_tokens.push_back(token);
            }
        }
        processTokens(acceptInfo.acceptTokens);
        commitAcceptedTokens(acceptInfo);
        if (stop || newTokens >= param.max_new_tokens) {
            break;
        }
        draftTimer.reset();
        draftInfo = updateDraft(acceptInfo, decodingInfo[1]);
        if (draftInfo.draftTokens.empty() || mContext->status == LlmStatus::INTERNAL_ERROR) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            break;
        }
        const uint64_t draftDecodeUs = draftTimer.durationInUs();
        mSpecContext.draft_time_us += draftDecodeUs;
        mSpecContext.draft_decode_time_us += draftDecodeUs;
        mSpecContext.draft += draftInfo.draftTokens.size();
    }
    mContext->decode_us += _t.durationInUs();
    if (newTokens >= param.max_new_tokens && mContext->status == LlmStatus::RUNNING) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }
    mBasePendingKV.erase(reqId);
    return;
}

} // namespace Transformer
} // namespace MNN
