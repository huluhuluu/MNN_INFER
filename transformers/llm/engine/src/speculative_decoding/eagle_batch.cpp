//
//  eagle_batch.cpp
//
//  Batch Eagle3 generation for continuous batching.
//

#include "generate.hpp"
#include "tokentree.hpp"
#include <cstring>
#include <limits>
#include <set>

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

static constexpr size_t kMaxPackedDraftModulePool = 4;

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

static VARP _slicePackedRows(VARP hiddenStates, int offset, int len) {
    auto dims = hiddenStates->getInfo()->dim;
    if (dims.size() == 3) {
        if (dims[0] == 1) {
            return _Slice(hiddenStates, _var<int>({0, offset, 0}, {3}), _var<int>({1, len, -1}, {3}));
        }
        return _Slice(hiddenStates, _var<int>({offset, 0, 0}, {3}), _var<int>({len, 1, -1}, {3}));
    }
    return _Slice(hiddenStates, _var<int>({offset, 0}, {2}), _var<int>({len, -1}, {2}));
}

static int _packedSeqLen(VARP hiddenStates) {
    auto info = hiddenStates == nullptr ? nullptr : hiddenStates->getInfo();
    if (info == nullptr || info->dim.empty()) {
        return 0;
    }
    if (info->dim.size() == 3 && info->dim[0] == 1) {
        return info->dim[1];
    }
    return info->dim[0];
}

static int _varElementSize(VARP var) {
    auto info = var == nullptr ? nullptr : var->getInfo();
    return info == nullptr ? 0 : info->size;
}

static VARP _cloneHiddenToInput(VARP hiddenStates) {
    auto info = hiddenStates->getInfo();
    if (info == nullptr) {
        return nullptr;
    }
    std::vector<int> dims = info->dim;
    if (dims.size() == 3 && dims[0] == 1) {
        dims = {dims[1], 1, dims[2]};
    }
    auto input = _Input(dims, NCHW, halide_type_of<float>());
    auto src = hiddenStates->readMap<float>();
    auto dst = input->writeMap<float>();
    if (src == nullptr || dst == nullptr) {
        return nullptr;
    }
    ::memcpy(dst, src, info->size * sizeof(float));
    return input;
}

static VARP _makePositionIds(const std::vector<int>& positions) {
    auto positionIds = _Input({static_cast<int>(positions.size())}, NCHW, halide_type_of<int>());
    auto dst = positionIds->writeMap<int>();
    if (dst != nullptr && !positions.empty()) {
        ::memcpy(dst, positions.data(), positions.size() * sizeof(int));
    }
    return positionIds;
}

static VARP _packFlatFloatVars(const VARPS& vars) {
    int total = 0;
    for (auto& var : vars) {
        total += var->getInfo()->size;
    }
    auto packed = _Input({1, 1, 1, total}, NCHW, halide_type_of<float>());
    auto dst = packed->writeMap<float>();
    int offset = 0;
    for (auto& var : vars) {
        auto info = var->getInfo();
        auto src = var->readMap<float>();
        if (src == nullptr || dst == nullptr) {
            return nullptr;
        }
        ::memcpy(dst + offset, src, info->size * sizeof(float));
        offset += info->size;
    }
    return packed;
}

static std::shared_ptr<Module> _loadPackedEagleModule(const std::string& modelPath,
                                                      const std::shared_ptr<MNN::Express::Executor::RuntimeManager>& runtimeManager,
                                                      const Module::Config& moduleConfig) {
    std::vector<std::string> inputNames{"input_embed", "hidden_states", "attention_mask", "position_ids", "logits_index"};
    std::vector<std::string> outputNames{"logits", "out_hidden_states"};
    return std::shared_ptr<Module>(Module::load(inputNames, outputNames, modelPath.c_str(),
                                                runtimeManager, &moduleConfig));
}

void EagleGeneration::loadPackedDraftModule() {
    mEagleBatchMeta.reset(new BatchKVMeta);
    mEaglePackedRootModule.reset();
    mEaglePackedCacheOwner.reset();
    mEaglePackedModulePool.clear();
    mLlm->mRuntimeManager->setHint(MNN::Interpreter::PACKED_ATTENTION_MODE, true);
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mEagleBatchMeta.get());
    mEaglePackedRootModule = _loadPackedEagleModule(mLlm->mConfig->eagle_model(),
                                                    mLlm->mRuntimeManager,
                                                    mEagleModuleConfig);
}

std::vector<MNN::Express::VARP> EagleGeneration::eagleForwardRawPacked(const std::vector<PackedDraftKVInfo>& kvInfos, const std::vector<MNN::Express::VARP>& inputs, bool waitAllOutputs) {
    if (mEagleBatchMeta == nullptr) {
        mEagleBatchMeta.reset(new BatchKVMeta);
    }
    for (auto& kv : kvInfos) {
        mEagleBatchMeta->setKVCacheInfo(kv.reqId, kv.add, kv.remove, nullptr, 0);
        mEagleBatchMeta->setKVMetaInfo(kv.reqId, mLlm->mConfig->layer_nums(), 0, 0, "", KVMeta::NoChange);
    }
    mLlm->mRuntimeManager->setHint(MNN::Interpreter::PACKED_ATTENTION_MODE, true);
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mEagleBatchMeta.get());
    std::vector<MNN::Express::VARP> outputs;
    int totalLen = inputs.empty() || inputs[0] == nullptr ? 0 : _packedSeqLen(inputs[0]);
    int maskSize = inputs.size() > 2 ? _varElementSize(inputs[2]) : 0;
    if (totalLen <= 0 || maskSize <= 0) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, mLlm->mConfig->packed_attention());
        return outputs;
    }
    if (mEaglePackedRootModule == nullptr) {
        mEaglePackedRootModule = _loadPackedEagleModule(mLlm->mConfig->eagle_model(),
                                                        mLlm->mRuntimeManager,
                                                        mEagleModuleConfig);
    }
    if (mEaglePackedRootModule == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, mLlm->mConfig->packed_attention());
        return outputs;
    }
    // Clone shape-specialized draft modules from one root module so CPU packed
    // attention keeps a single KV cache manager across module keys.
    auto createPackedModule = [&]() {
        return std::shared_ptr<Module>(Module::clone(mEaglePackedRootModule.get()));
    };
    auto moduleKey = std::make_pair(totalLen, maskSize);
    auto iter = mEaglePackedModulePool.find(moduleKey);
    if (iter == mEaglePackedModulePool.end()) {
        if (mEaglePackedModulePool.size() >= kMaxPackedDraftModulePool) {
            mEaglePackedModulePool.erase(mEaglePackedModulePool.begin());
        }
        iter = mEaglePackedModulePool.emplace(moduleKey, createPackedModule()).first;
    }
    if (iter->second == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, mLlm->mConfig->packed_attention());
        return outputs;
    }
    // PackedAttention clones share KV managers. Keep the first executing
    // clone alive because those managers retain its backend pointer.
    if (mEaglePackedCacheOwner == nullptr) {
        mEaglePackedCacheOwner = iter->second;
    }
    outputs = iter->second->onForward(inputs);
    if (outputs.empty()) {
        auto failedModule = iter->second;
        mEaglePackedModulePool.erase(moduleKey);
        auto module = createPackedModule();
        if (module == nullptr) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, mLlm->mConfig->packed_attention());
            return outputs;
        }
        outputs = module->onForward(inputs);
        if (outputs.empty()) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, mLlm->mConfig->packed_attention());
            return outputs;
        }
        if (mEaglePackedCacheOwner == failedModule) {
            mEaglePackedCacheOwner = module;
        }
        mEaglePackedModulePool[moduleKey] = std::move(module);
    }
    if (outputs.size() > 1) {
        if (waitAllOutputs) {
            waitModuleOutputs(outputs);
        } else {
            outputs[0]->readMap<float>();
        }
    }
    mEagleBatchMeta->sync();
    mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, mLlm->mConfig->packed_attention());
    return outputs;
}

VARPS EagleGeneration::treeDecodingPacked(const EagleGeneration::DraftInfo& draftInfo) {
    auto inputEmbeds = mLlm->embedding(draftInfo.draftTokens);
    int inputLen = draftInfo.draftTokens.size();
    mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, true);
    auto pendingIter = mBasePendingKV.find(draftInfo.reqId);
    PendingBaseKV* pending = pendingIter == mBasePendingKV.end() ? nullptr : &pendingIter->second;
    size_t remove = pending == nullptr ? 0 : pending->remove;
    int* reserve = (pending == nullptr || pending->reserveHost.empty()) ? nullptr : pending->reserveHost.data();
    int reserveSize = pending == nullptr ? 0 : static_cast<int>(pending->reserveHost.size() / 2);
    mLlm->mBatchMeta->setKVCacheInfo(draftInfo.reqId, inputLen, remove, reserve, reserveSize);
    mLlm->mBatchMeta->setKVMetaInfo(draftInfo.reqId, mLlm->mConfig->layer_nums(), 0, 0, "", KVMeta::NoChange);
    auto treeMask = getPackedMask({draftInfo});
    auto moduleKey = std::make_pair(inputLen, true);
    if (mLlm->mModulePool.find(moduleKey) == mLlm->mModulePool.end()) {
        mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, true);
        mLlm->mModulePool[moduleKey].reset(Module::clone(mLlm->mModule.get()));
    }
    auto outputs = mLlm->mModulePool[moduleKey]->onForward({inputEmbeds, treeMask, draftInfo.positionIds, mLlm->logitsAllIdx});
    mLlm->mBatchMeta->sync();
    mBasePendingKV.erase(draftInfo.reqId);
    return outputs;
}

MNN::Express::VARP EagleGeneration::getPackedMask(const std::vector<DraftInfo>& draftInfos) {
    int total = 0;
    for (auto& info : draftInfos) {
        int len = static_cast<int>(info.draftTokens.size());
        total += len * len;
    }
    auto mask = _Input({1, 1, 1, total}, NCHW, halide_type_of<float>());
    auto dst = mask->writeMap<float>();
    int offset = 0;
    for (auto& info : draftInfos) {
        int len = static_cast<int>(info.draftTokens.size());
        auto src = info.attentionMask->readMap<float>();
        ::memcpy(dst + offset, src, len * len * sizeof(float));
        offset += len * len;
    }
    return mask;
}

MNN::Express::VARP EagleGeneration::eagleFCForward(const MNN::Express::VARPS& hiddenStates) {
    if (hiddenStates.empty()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return nullptr;
    }
    VARPS packedInputs;
    packedInputs.reserve(hiddenStates.size());
    for (auto& hidden : hiddenStates) {
        if (_packedSeqLen(hidden) <= 0) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return nullptr;
        }
        packedInputs.push_back(_cloneHiddenToInput(hidden));
    }
    auto packedHidden = packedInputs.size() == 1 ? packedInputs[0] : _Concat(packedInputs, 0);
    auto outputs = mEagleModules[1]->onForward({packedHidden});
    if (outputs.empty()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return nullptr;
    }
    return _cloneHiddenToInput(outputs[0]);
}

std::vector<EagleGeneration::DraftInfo> EagleGeneration::topkGeneratePacked(const std::vector<PackedDraftInput>& inputs) {
    std::vector<DraftInfo> results;
    if (inputs.empty()) {
        return results;
    }
    struct PackedWork {
        int reqId = 0;
        EagleState* state = nullptr;
        int sampleToken = 0;
        int seqLen = 0;
        int lastOffset = 0;
        std::unique_ptr<TokenTree> tokenTree;
        VARP inputHidden;
    };

    auto d2tPtr = mD2t->readMap<int>();
    std::vector<PackedWork> works;
    std::vector<VARP> inputEmbedsList;
    std::vector<VARP> fcInputList;
    std::vector<int> calLen;
    std::vector<int> positionHost;
    std::vector<PackedDraftKVInfo> kvInfos;
    int totalLen = 0;

    for (auto& input : inputs) {
        if (input.state == nullptr || input.inputIds.empty()) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return {};
        }
        VARP inputEmbeds = input.inputEmbeds;
        if (inputEmbeds == nullptr) {
            inputEmbeds = mLlm->embedding(input.inputIds);
        }
        int inputLen = inputEmbeds->getInfo()->dim[0];
        if (_packedSeqLen(input.hiddenStates) != inputLen) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return {};
        }
        int seqLen = input.state->pastLen + inputLen;
        PackedWork work;
        work.reqId = input.reqId;
        work.state = input.state;
        work.sampleToken = input.inputIds.back();
        work.seqLen = seqLen;
        work.lastOffset = totalLen + inputLen - 1;
        work.tokenTree.reset(new TokenTree(mTopK, d2tPtr));
        inputEmbedsList.push_back(inputEmbeds);
        fcInputList.push_back(input.hiddenStates);
        calLen.push_back(inputLen);
        for (int i = 0; i < inputLen; ++i) {
            positionHost.push_back(input.state->pastLen + i);
        }
        PackedDraftKVInfo kvInfo;
        kvInfo.reqId = input.reqId;
        kvInfo.add = static_cast<size_t>(inputLen);
        kvInfo.remove = static_cast<size_t>(input.state->remove);
        kvInfos.push_back(kvInfo);
        works.push_back(std::move(work));
        totalLen += inputLen;
    }

    auto packedFcHidden = eagleFCForward(fcInputList);
    if (packedFcHidden == nullptr || _packedSeqLen(packedFcHidden) < totalLen) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    auto inputEmbeds = inputEmbedsList.size() == 1 ? inputEmbedsList[0] : _Concat(inputEmbedsList, 0);
    auto inputHidden = packedFcHidden;
    auto attentionMask = mLlm->gen_attention_mask(calLen);
    auto positionIds = _makePositionIds(positionHost);
    auto outputs = eagleForwardRawPacked(kvInfos, {inputEmbeds, inputHidden, attentionMask, positionIds, mLlm->logitsAllIdx});
    if (outputs.size() < 2) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    auto logitsAll = _Squeeze(outputs[0], {0});
    auto hiddenAll = outputs[1];

    for (auto& work : works) {
        work.state->pastLen = work.seqLen;
        work.state->remove = mTopK * (mDepth - 1);
        auto lastP = _Gather(logitsAll, _Scalar<int>(work.lastOffset));
        auto lastHidden = _slicePackedRows(hiddenAll, work.lastOffset, 1);
        auto topKV = MNN::Express::_TopKV2(lastP, MNN::Express::_Scalar<int>(mTopK));
        auto scores = topKV[0]->readMap<float>();
        auto indices = topKV[1]->readMap<int>();
        work.tokenTree->init(indices, scores);
        work.inputHidden = _cloneHiddenToInput(MNN::Express::_Tile(lastHidden, _var<int>({1, mTopK, 1}, {3})));
    }

    for (int d = 0; d < mDepth - 1; d++) {
        std::vector<std::vector<int>> packedIds;
        std::vector<int> stepCalLen;
        std::vector<VARP> stepHiddenList;
        std::vector<VARP> stepMaskList;
        std::vector<int> stepPositionHost;
        std::vector<PackedDraftKVInfo> stepKVInfos;
        totalLen = 0;
        for (auto& work : works) {
            auto ids = work.tokenTree->getIds();
            int stepLen = static_cast<int>(ids.size());
            packedIds.push_back(std::move(ids));
            stepCalLen.push_back(stepLen);
            stepHiddenList.push_back(work.inputHidden);
            stepMaskList.push_back(getMask(work.tokenTree->getMask(), work.seqLen));
            for (int i = 0; i < stepLen; ++i) {
                stepPositionHost.push_back(work.seqLen + d);
            }
            PackedDraftKVInfo kvInfo;
            kvInfo.reqId = work.reqId;
            kvInfo.add = static_cast<size_t>(stepLen);
            stepKVInfos.push_back(kvInfo);
            totalLen += stepLen;
        }
        auto stepEmbeds = mLlm->embedding(packedIds, stepCalLen, totalLen);
        auto stepHidden = stepHiddenList.size() == 1 ? stepHiddenList[0] : _Concat(stepHiddenList, 0);
        auto stepMask = _packFlatFloatVars(stepMaskList);
        auto stepPositionIds = _makePositionIds(stepPositionHost);
        if (stepMask == nullptr) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return {};
        }
        outputs = eagleForwardRawPacked(stepKVInfos, {stepEmbeds, stepHidden, stepMask, stepPositionIds, mLlm->logitsAllIdx});
        if (outputs.size() < 2) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return {};
        }
        logitsAll = _Squeeze(outputs[0], {0});
        hiddenAll = outputs[1];
        int offset = 0;
        for (int i = 0; i < works.size(); ++i) {
            int stepLen = stepCalLen[i];
            auto logitsSlice = _Slice(logitsAll, _var<int>({offset, 0}, {2}), _var<int>({stepLen, -1}, {2}));
            auto topKV = MNN::Express::_TopKV2(logitsSlice, MNN::Express::_Scalar<int>(mTopK));
            auto scores = topKV[0]->readMap<float>();
            auto indices = topKV[1]->readMap<int>();
            works[i].tokenTree->grow(indices, scores);
            works[i].inputHidden = _cloneHiddenToInput(_slicePackedRows(hiddenAll, offset, stepLen));
            offset += stepLen;
        }
    }

    for (auto& work : works) {
        auto output = work.tokenTree->finalize(work.sampleToken, mLlm->mDraftLength);
        int inputLen = output.draftTokens.size();
        DraftInfo info;
        info.reqId = work.reqId;
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
            info.positionIds->writeMap<int>()[i] = work.seqLen + output.positionIds[i];
        }
        results.push_back(std::move(info));
    }
    return results;
}

bool EagleGeneration::prefillDraftPacked(const std::vector<PackedDraftInput>& inputs) {
    if (inputs.empty()) {
        return true;
    }
    std::vector<VARP> embedsList;
    std::vector<VARP> hiddenList;
    std::vector<int> calLen;
    std::vector<int> positionHost;
    std::vector<PackedDraftKVInfo> kvInfos;
    for (auto& input : inputs) {
        if (input.state == nullptr || input.inputIds.empty() || _packedSeqLen(input.hiddenStates) != input.inputIds.size()) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return false;
        }
        int inputLen = static_cast<int>(input.inputIds.size());
        embedsList.push_back(mLlm->embedding(input.inputIds));
        hiddenList.push_back(input.hiddenStates);
        calLen.push_back(inputLen);
        for (int i = 0; i < inputLen; ++i) {
            positionHost.push_back(input.state->pastLen + i);
        }
        PackedDraftKVInfo kvInfo;
        kvInfo.reqId = input.reqId;
        kvInfo.add = static_cast<size_t>(inputLen);
        kvInfo.remove = static_cast<size_t>(input.state->remove);
        kvInfos.push_back(kvInfo);
    }
    auto hidden = eagleFCForward(hiddenList);
    if (hidden == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    auto embeds = embedsList.size() == 1 ? embedsList[0] : _Concat(embedsList, 0);
    auto outputs = eagleForwardRawPacked(kvInfos, {embeds, hidden, mLlm->gen_attention_mask(calLen),
                                                  _makePositionIds(positionHost), mLlm->logitsLastIdx}, false);
    if (outputs.size() < 2) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    for (auto& input : inputs) {
        input.state->pastLen += static_cast<int>(input.inputIds.size());
        input.state->remove = 0;
    }
    return true;
}

void EagleGeneration::updatePackedBaseKV(const AcceptInfo& acceptInfo) {
    int acceptLen = static_cast<int>(acceptInfo.acceptTokens.size());
    auto& pending = mBasePendingKV[acceptInfo.reqId];
    pending.remove = acceptInfo.sampleTokens.size();
    pending.reserveHost.resize(acceptLen * 2);
    for (int i = 0; i < acceptLen; i++) {
        pending.reserveHost[2 * i] = acceptInfo.acceptIndices[i];
        pending.reserveHost[2 * i + 1] = 1;
    }
}

std::vector<std::vector<int>> EagleGeneration::generateBatch(const std::vector<std::vector<int>>& inputIds, std::ostream* os, int maxNewTokens) {
    if (!mLlm->mConfig->packed_attention()) {
        MNN_PRINT("EagleGeneration::generateBatch requires packed_attention to be enabled in the model config.\n");
        return {};
    }
    int bs = inputIds.size();
    std::vector<std::vector<int>> ret(bs, std::vector<int>{});
    int maxTokens = maxNewTokens > 0 ? maxNewTokens : mLlm->mConfig->max_new_tokens();
    mContext->prompt_len = 0;
    mContext->gen_seq_len = 0;
    mContext->all_seq_len = 0;

    std::vector<int> reqIds = mLlm->mScheduler->addRequest(inputIds);
    std::map<int, int> inputIndexByReqId;
    for (int i = 0; i < reqIds.size(); ++i) {
        inputIndexByReqId[reqIds[i]] = i;
    }
    mLlm->mScheduler->setMaxNewTokens(maxTokens);
    mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, true);

    std::map<int, EagleState> eagleStates;
    std::map<int, DraftInfo> draftInfos;
    if (mEagleBatchMeta != nullptr) {
        mEagleBatchMeta->reset();
    }

    struct BatchBaseForward {
        VARP logits;
        VARP hiddenStates;
        VARP inputEmbeds;
        std::vector<int> draftOffsets;
        std::vector<int> chunkOffsets;
        std::vector<int> chunkDraftIndices;
    };

    auto releaseRequestState = [&](int id) {
        mLlm->mScheduler->releaseKVCache(id);
        eagleStates.erase(id);
        mBasePendingKV.erase(id);
        if (mEagleBatchMeta != nullptr) {
            mEagleBatchMeta->releaseKV(id);
        }
    };

    auto runBatchBaseForward = [&](const std::vector<DraftInfo>& activeDrafts,
                                   const std::shared_ptr<BatchScheduler::Chunk>& chunk) -> BatchBaseForward {
        BatchBaseForward result;
        std::vector<std::vector<int>> packedTokens;
        std::vector<int> calLen;
        std::vector<int> positionHost;
        int culLen = 0;
        int maskSize = 0;
        int chunkMaskSize = 0;
        std::vector<int> chunkCalLenOrdered;
        std::vector<int> chunkOrder;

        for (auto& info : activeDrafts) {
            int len = static_cast<int>(info.draftTokens.size());
            result.draftOffsets.push_back(culLen);
            packedTokens.push_back(info.draftTokens);
            calLen.push_back(len);
            culLen += len;
            maskSize += len * len;

            auto positionPtr = info.positionIds->readMap<int>();
            if (positionPtr == nullptr) {
                mContext->status = LlmStatus::INTERNAL_ERROR;
                return result;
            }
            positionHost.insert(positionHost.end(), positionPtr, positionPtr + len);

            auto pendingIter = mBasePendingKV.find(info.reqId);
            PendingBaseKV* pending = pendingIter == mBasePendingKV.end() ? nullptr : &pendingIter->second;
            size_t remove = pending == nullptr ? 0 : pending->remove;
            int* reserve = (pending == nullptr || pending->reserveHost.empty()) ? nullptr : pending->reserveHost.data();
            int reserveSize = pending == nullptr ? 0 : static_cast<int>(pending->reserveHost.size() / 2);
            mLlm->mBatchMeta->setKVCacheInfo(info.reqId, len, remove, reserve, reserveSize);
            mLlm->mBatchMeta->setKVMetaInfo(info.reqId, mLlm->mConfig->layer_nums(), 0, 0, "", KVMeta::NoChange);
        }

        if (chunk != nullptr) {
            result.chunkOffsets.resize(chunk->inputs.size(), 0);
            for (int i = 0; i < chunk->inputs.size(); ++i) {
                int state = mLlm->mScheduler->state(chunk->reqId[i]);
                if (BatchScheduler::judgeState(state, BatchScheduler::RequestState::DECODE)) {
                    result.chunkDraftIndices.push_back(i);
                    chunkOrder.push_back(i);
                }
            }
            for (int i = 0; i < chunk->inputs.size(); ++i) {
                int state = mLlm->mScheduler->state(chunk->reqId[i]);
                if (!BatchScheduler::judgeState(state, BatchScheduler::RequestState::DECODE)) {
                    chunkOrder.push_back(i);
                }
            }
            for (int index : chunkOrder) {
                int len = chunk->calLen[index];
                result.chunkOffsets[index] = culLen;
                packedTokens.push_back(chunk->inputs[index]);
                calLen.push_back(len);
                chunkCalLenOrdered.push_back(len);
                culLen += len;
                maskSize += len * len;
                chunkMaskSize += len * len;
                for (int j = 0; j < len; ++j) {
                    positionHost.push_back(chunk->pos[index] + j);
                }
                int reqId = chunk->reqId[index];
                mLlm->mBatchMeta->setKVCacheInfo(reqId, len, 0, nullptr, 0);
                mLlm->mBatchMeta->setKVMetaInfo(reqId, mLlm->mConfig->layer_nums(), 0, 0, "", KVMeta::NoChange);
            }
        }

        if (culLen <= 0) {
            return result;
        }

        VARP attentionMask;
        if (activeDrafts.empty() && chunk != nullptr) {
            attentionMask = mLlm->gen_attention_mask(chunkCalLenOrdered);
        } else {
            attentionMask = _Input({1, 1, 1, maskSize}, NCHW, halide_type_of<float>());
            auto maskDst = attentionMask->writeMap<float>();
            int maskOffset = 0;
            for (auto& info : activeDrafts) {
                int len = static_cast<int>(info.draftTokens.size());
                auto maskSrc = info.attentionMask->readMap<float>();
                if (maskDst == nullptr || maskSrc == nullptr) {
                    mContext->status = LlmStatus::INTERNAL_ERROR;
                    return result;
                }
                ::memcpy(maskDst + maskOffset, maskSrc, len * len * sizeof(float));
                maskOffset += len * len;
            }
            if (chunk != nullptr) {
                auto chunkMask = mLlm->gen_attention_mask(chunkCalLenOrdered);
                auto chunkMaskInfo = chunkMask == nullptr ? nullptr : chunkMask->getInfo();
                if (chunkMaskInfo != nullptr && !chunkMaskInfo->dim.empty() && chunkMaskInfo->size == chunkMaskSize) {
                    if (chunkMaskInfo->type == halide_type_of<float>()) {
                        auto chunkMaskSrc = chunkMask->readMap<float>();
                        if (chunkMaskSrc == nullptr) {
                            mContext->status = LlmStatus::INTERNAL_ERROR;
                            return result;
                        }
                        ::memcpy(maskDst + maskOffset, chunkMaskSrc, chunkMaskSize * sizeof(float));
                    } else if (chunkMaskInfo->type.code == halide_type_int) {
                        auto chunkMaskSrc = chunkMask->readMap<int>();
                        if (chunkMaskSrc == nullptr) {
                            mContext->status = LlmStatus::INTERNAL_ERROR;
                            return result;
                        }
                        float minVal = std::numeric_limits<float>::lowest();
                        for (int i = 0; i < chunkMaskSize; ++i) {
                            maskDst[maskOffset + i] = chunkMaskSrc[i] == 0 ? minVal : 0.0f;
                        }
                    } else {
                        mContext->status = LlmStatus::INTERNAL_ERROR;
                        return result;
                    }
                } else {
                    float minVal = std::numeric_limits<float>::lowest();
                    for (int len : chunkCalLenOrdered) {
                        for (int r = 0; r < len; ++r) {
                            for (int c = 0; c < len; ++c) {
                                maskDst[maskOffset + r * len + c] = c > r ? minVal : 0.0f;
                            }
                        }
                        maskOffset += len * len;
                    }
                }
            }
        }

        mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, true);
        auto inputEmbeds = mLlm->embedding(packedTokens, calLen, culLen);
        auto positionIds = _makePositionIds(positionHost);
        auto moduleKey = std::make_pair(culLen, true);
        std::shared_ptr<Module> selectModule = mLlm->mModule;
        if(mLlm->mModulePool.find(moduleKey) == mLlm->mModulePool.end()) {
            mLlm->mModulePool[moduleKey].reset(Module::clone(mLlm->mModule.get()));
        }
        selectModule = mLlm->mModulePool[moduleKey];
        auto outputs = selectModule->onForward({inputEmbeds, attentionMask, positionIds, mLlm->logitsAllIdx});
        if (outputs.size() < 2) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return result;
        }
        mLlm->mBatchMeta->sync();
        for (auto& info : activeDrafts) {
            mBasePendingKV.erase(info.reqId);
        }
        result.logits = _Squeeze(outputs[0], {0});
        result.hiddenStates = outputs[1];
        result.inputEmbeds = inputEmbeds;
        return result;
    };

    while (mLlm->mScheduler->hasValidWork() || !draftInfos.empty()) {
        std::set<int> skipReqIds;
        std::vector<DraftInfo> activeDrafts;
        for (auto& kv : draftInfos) {
            if (!mLlm->mScheduler->isFinished(kv.first)) {
                skipReqIds.insert(kv.first);
                activeDrafts.push_back(kv.second);
            }
        }
        auto chunk = mLlm->mScheduler->schedule(-1, 4, skipReqIds);
        if (activeDrafts.empty() && chunk == nullptr) {
            break;
        }

        // One base forward can contain previous draft verification plus fresh
        // scheduler work. Prefill chunks stop here unless they become DECODE
        // after this forward; only DECODE requests enter the draft model.
        auto base = runBatchBaseForward(activeDrafts, chunk);
        if (base.logits == nullptr || base.hiddenStates == nullptr) {
            if (mContext->status != LlmStatus::INTERNAL_ERROR) {
                mContext->status = LlmStatus::INTERNAL_ERROR;
            }
            break;
        }

        std::map<int, DraftInfo> nextDrafts;
        std::vector<PackedDraftInput> nextDraftInputs;
        for (int i = 0; i < activeDrafts.size(); ++i) {
            auto& draft = activeDrafts[i];
            int id = draft.reqId;
            int len = static_cast<int>(draft.draftTokens.size());
            auto logitsSlice = _Slice(base.logits, _var<int>({base.draftOffsets[i], 0}, {2}), _var<int>({len, -1}, {2}));
            auto acceptInfo = evaluatePosterior(draft, logitsSlice);
            std::vector<int> accepted;
            const size_t generatedSize = mLlm->mScheduler->getResultSize(id);
            bool stop = false;
            int acceptLimit = 0;
            for (auto token : acceptInfo.acceptTokens) {
                accepted.push_back(token);
                acceptLimit++;
                if (mLlm->is_stop(token)) {
                    stop = true;
                    break;
                }
                if (generatedSize + accepted.size() >= maxTokens) {
                    break;
                }
            }
            if (acceptLimit < acceptInfo.acceptTokens.size()) {
                acceptInfo.acceptTokens.resize(acceptLimit);
                acceptInfo.acceptIndices.resize(acceptLimit);
            }
            mLlm->mScheduler->update(id, accepted, 0, stop);
            mLlm->updateContext(static_cast<int>(accepted.size()), static_cast<int>(accepted.size()));
            if (mLlm->mScheduler->isFinished(id)) {
                releaseRequestState(id);
                continue;
            }
            if (accepted.empty()) {
                mContext->status = LlmStatus::INTERNAL_ERROR;
                continue;
            }
            auto hiddenSlice = _slicePackedRows(base.hiddenStates, base.draftOffsets[i], len);
            updatePackedBaseKV(acceptInfo);
            auto acceptHiddenState = gatherHiddenRows(hiddenSlice, acceptInfo.acceptIndices);
            if (acceptHiddenState == nullptr) {
                mContext->status = LlmStatus::INTERNAL_ERROR;
                continue;
            }
            PackedDraftInput draftInput;
            draftInput.reqId = id;
            draftInput.state = &eagleStates[id];
            draftInput.inputIds = acceptInfo.acceptTokens;
            draftInput.hiddenStates = acceptHiddenState;
            nextDraftInputs.push_back(std::move(draftInput));
        }

        if (chunk != nullptr) {
            std::set<int> finalChunkIndices(base.chunkDraftIndices.begin(), base.chunkDraftIndices.end());
            std::vector<PackedDraftInput> draftPrefillInputs;
            for (int i = 0; i < chunk->pos.size(); ++i) {
                int reqLen = chunk->calLen[i];
                if(reqLen > 1) {
                    mLlm->updateContext(reqLen, 0);
                    mContext->prompt_len += reqLen;
                }
                if (finalChunkIndices.find(i) == finalChunkIndices.end()) {
                    int id = chunk->reqId[i];
                    auto inputIter = inputIndexByReqId.find(id);
                    int pos = chunk->pos[i];
                    if (inputIter == inputIndexByReqId.end() || pos < 0 || reqLen <= 0 ||
                        pos + reqLen >= inputIds[inputIter->second].size()) {
                        mContext->status = LlmStatus::INTERNAL_ERROR;
                        break;
                    }
                    int inputIndex = inputIter->second;
                    PackedDraftInput draftInput;
                    draftInput.reqId = id;
                    draftInput.state = &eagleStates[id];
                    draftInput.inputIds.assign(inputIds[inputIndex].begin() + pos + 1,
                                               inputIds[inputIndex].begin() + pos + reqLen + 1);
                    draftInput.hiddenStates = _slicePackedRows(base.hiddenStates, base.chunkOffsets[i], reqLen);
                    draftPrefillInputs.push_back(std::move(draftInput));
                }
            }
            if (mContext->status == LlmStatus::INTERNAL_ERROR) {
                break;
            }
            if (!prefillDraftPacked(draftPrefillInputs)) {
                break;
            }
            for (int i : base.chunkDraftIndices) {
                int id = chunk->reqId[i];
                int reqLen = chunk->calLen[i];
                int offset = base.chunkOffsets[i];
                auto logit = MNN::Express::_Gather(base.logits, _Scalar(offset + reqLen - 1));
                int sampleToken = mLlm->sample(logit);
                bool stop = mLlm->is_stop(sampleToken);
                std::vector<int> firstToken{sampleToken};
                mLlm->mScheduler->update(id, firstToken, 0, stop);
                mLlm->updateContext(1, 1);
                if (stop || mLlm->mScheduler->isFinished(id)) {
                    releaseRequestState(id);
                    continue;
                }

                auto hiddenSlice = _slicePackedRows(base.hiddenStates, offset, reqLen);
                auto inputEmbedSlice = _Slice(base.inputEmbeds, _var<int>({offset, 0, 0}, {3}), _var<int>({reqLen, 1, -1}, {3}));
                auto curEmbed = mLlm->embedding({sampleToken});
                VARP eagleInputEmbed = curEmbed;
                if (reqLen > 1) {
                    auto preEmbeds = _Split(inputEmbedSlice, {1, reqLen - 1}, 0);
                    eagleInputEmbed = _Concat({preEmbeds[1], curEmbed}, 0);
                }
                std::vector<int> draftInputIds = chunk->inputs[i];
                draftInputIds.push_back(sampleToken);
                auto& stateInfo = eagleStates[id];
                stateInfo.remove = 0;
                PackedDraftInput draftInput;
                draftInput.reqId = id;
                draftInput.state = &stateInfo;
                draftInput.inputIds = std::move(draftInputIds);
                draftInput.hiddenStates = hiddenSlice;
                draftInput.inputEmbeds = eagleInputEmbed;
                nextDraftInputs.push_back(std::move(draftInput));
            }
        }

        if (!nextDraftInputs.empty()) {
            auto packedDrafts = topkGeneratePacked(nextDraftInputs);
            for (auto& draft : packedDrafts) {
                nextDrafts[draft.reqId] = std::move(draft);
            }
        }
        draftInfos.swap(nextDrafts);
        if (mContext->status == LlmStatus::INTERNAL_ERROR) {
            break;
        }
    }
    for(int id: reqIds){
        const auto result = mLlm->mScheduler->getResult(id);
        for(int j = 0; j < bs; j++) {
            if(reqIds[j] == id) {
                ret[j] = result;
                break;
            }
        }
        if(os!= nullptr){
            *os<<"\n=============================\nReqId: "<<id<<"\n";
            for(int token: result){
                *os<<mLlm->tokenizer_decode(token);
            }
        }
        mLlm->mScheduler->releaseReq(id);
        mBasePendingKV.erase(id);
        if (mEagleBatchMeta != nullptr) {
            mEagleBatchMeta->releaseKV(id);
        }
    }
    return ret;
}

} // namespace Transformer
} // namespace MNN
