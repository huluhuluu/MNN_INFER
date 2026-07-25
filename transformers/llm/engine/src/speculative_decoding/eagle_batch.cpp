//
//  eagle_batch.cpp
//
//  Batch Eagle3 generation for continuous batching.
//

#include "generate.hpp"
#include "tokentree.hpp"
#include <MNN/expr/ExecutorScope.hpp>
#include <cstring>
#include <limits>
#include <set>
#include <thread>

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

static constexpr size_t kMaxPackedDraftModulePool = 4;

static void _copyBatchKVMetaEntry(BatchKVMeta* dst, const BatchKVMeta* src, int reqId, bool appendCalId) {
    if (dst == nullptr || src == nullptr) {
        return;
    }
    auto srcIter = src->mMetas.find(reqId);
    if (srcIter == src->mMetas.end() || srcIter->second == nullptr) {
        return;
    }
    KVMeta*& dstMeta = dst->mMetas[reqId];
    if (dstMeta == nullptr) {
        dstMeta = new KVMeta;
    }
    *dstMeta = *srcIter->second;
    if (appendCalId) {
        dst->calId.push_back(reqId);
    }
}

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

static VARP _cloneHiddenRowsToInput(VARP hiddenStates, int offset, int len) {
    auto info = hiddenStates == nullptr ? nullptr : hiddenStates->getInfo();
    const int seqLen = _packedSeqLen(hiddenStates);
    if (info == nullptr || seqLen <= 0 || offset < 0 || len <= 0 || offset + len > seqLen ||
        info->type != halide_type_of<float>() || info->size % seqLen != 0) {
        return nullptr;
    }
    std::vector<int> dims = info->dim;
    if (dims.size() == 3) {
        dims = {len, 1, dims[2]};
    } else if (!dims.empty()) {
        dims[0] = len;
    }
    auto input = _Input(dims, NCHW, halide_type_of<float>());
    auto src = hiddenStates->readMap<float>();
    auto dst = input->writeMap<float>();
    if (src == nullptr || dst == nullptr) {
        return nullptr;
    }
    const int rowSize = info->size / seqLen;
    ::memcpy(dst, src + offset * rowSize, static_cast<size_t>(len) * rowSize * sizeof(float));
    return input;
}

static VARP _cloneHiddenToInput(VARP hiddenStates) {
    return _cloneHiddenRowsToInput(hiddenStates, 0, _packedSeqLen(hiddenStates));
}

static bool _captureFloatRows(VARP input, int rowCount,
                              std::vector<int>& dims, std::vector<float>& values) {
    auto info = input == nullptr ? nullptr : input->getInfo();
    const int seqLen = _packedSeqLen(input);
    if (info == nullptr || seqLen <= 0 || rowCount <= 0 || rowCount > seqLen ||
        info->type != halide_type_of<float>() || info->size % seqLen != 0) {
        return false;
    }
    auto src = input->readMap<float>();
    if (src == nullptr) {
        return false;
    }
    dims = info->dim;
    if (dims.size() == 3 && dims[0] == 1) {
        dims[1] = rowCount;
    } else if (!dims.empty()) {
        dims[0] = rowCount;
    }
    const size_t size = static_cast<size_t>(rowCount) * info->size / seqLen;
    values.assign(src, src + size);
    return true;
}

static VARP _makeFloatInput(const std::vector<int>& dims, const std::vector<float>& values) {
    if (dims.empty() || values.empty()) {
        return nullptr;
    }
    auto input = _Input(dims, NCHW, halide_type_of<float>());
    auto dst = input->writeMap<float>();
    if (dst == nullptr || input->getInfo()->size != values.size()) {
        return nullptr;
    }
    ::memcpy(dst, values.data(), values.size() * sizeof(float));
    return input;
}

static VARP _padPackedFloatRows(VARP input, int paddedLen) {
    auto info = input == nullptr ? nullptr : input->getInfo();
    const int actualLen = _packedSeqLen(input);
    if (info == nullptr || actualLen <= 0 || paddedLen <= actualLen) {
        return input;
    }
    std::vector<int> dims = info->dim;
    if (dims.size() == 3 && dims[0] == 1) {
        dims[1] = paddedLen;
    } else if (!dims.empty()) {
        dims[0] = paddedLen;
    }
    auto output = _Input(dims, NCHW, halide_type_of<float>());
    auto src = input->readMap<float>();
    auto dst = output->writeMap<float>();
    if (src == nullptr || dst == nullptr) {
        MNN_ERROR("MNN_DUAL_PIPELINE: packed float padding map failed "
                  "(actual=%d padded=%d src=%d dst=%d).\n",
                  actualLen, paddedLen, src != nullptr, dst != nullptr);
        return nullptr;
    }
    ::memset(dst, 0, output->getInfo()->size * sizeof(float));
    ::memcpy(dst, src, info->size * sizeof(float));
    return output;
}

static VARP _padPositionIds(VARP positionIds, int paddedLen) {
    auto info = positionIds == nullptr ? nullptr : positionIds->getInfo();
    if (info == nullptr || info->size <= 0 || paddedLen <= info->size) {
        return positionIds;
    }
    auto output = _Input({paddedLen}, NCHW, halide_type_of<int>());
    auto src = positionIds->readMap<int>();
    auto dst = output->writeMap<int>();
    if (src == nullptr || dst == nullptr) {
        MNN_ERROR("MNN_DUAL_PIPELINE: position padding map failed "
                  "(actual=%zu padded=%d src=%d dst=%d).\n",
                  info->size, paddedLen, src != nullptr, dst != nullptr);
        return nullptr;
    }
    ::memset(dst, 0, paddedLen * sizeof(int));
    ::memcpy(dst, src, info->size * sizeof(int));
    return output;
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

void EagleGeneration::loadDualPipelineGraphInfo() {
    mEagleDraftGraphSnapshot = buildQnnGraphSnapshotFromModel(
        mLlm->mConfig->eagle_model(), mLlm->mConfig->base_dir_, mLlm->mConfig->npu_model_dir());
    mEagleFCGraphSnapshot = buildQnnGraphSnapshotFromModel(
        mLlm->mConfig->eagle_fc(), mLlm->mConfig->base_dir_, mLlm->mConfig->npu_model_dir());
    mEagleDraftGraphRequests = buildQnnGraphRequests(
        mEagleDraftGraphSnapshot, 0, static_cast<int>(mEagleDraftGraphSnapshot.ops.size()), -1);
    mEagleFCGraphRequests = buildQnnGraphRequests(
        mEagleFCGraphSnapshot, 0, static_cast<int>(mEagleFCGraphSnapshot.ops.size()), -1);
    mEagleDraftQnnOpIndices.clear();
    mEagleFCQnnOpIndices.clear();
    auto initializeComponentGraphs = [](const GraphSnapshot& snapshot,
                                        std::vector<DualPipelineScheduler::GraphRequest>& requests,
                                        std::unordered_map<std::string, int>& opIndices) {
        int graphIndex = 0;
        for (auto& request : requests) {
            request.draftGraph = true;
            request.pinResident = true;
        }
        for (const auto& op : snapshot.ops) {
            if (isQnnPluginOp(op) && graphIndex < static_cast<int>(requests.size())) {
                opIndices.emplace(op.opName, graphIndex++);
            }
        }
    };
    initializeComponentGraphs(mEagleDraftGraphSnapshot, mEagleDraftGraphRequests, mEagleDraftQnnOpIndices);
    initializeComponentGraphs(mEagleFCGraphSnapshot, mEagleFCGraphRequests, mEagleFCQnnOpIndices);
}

VARPS EagleGeneration::runDualPipelineComponent(
    int pipelineId,
    const std::shared_ptr<Module>& module,
    const VARPS& inputs,
    const GraphSnapshot& graphSnapshot,
    const std::vector<DualPipelineScheduler::GraphRequest>& graphRequests,
    const std::unordered_map<std::string, int>& qnnOpIndices,
    const std::vector<int>& ownerReqIds) {
    VARPS outputs;
    if (pipelineId < 0 || pipelineId >= static_cast<int>(mEagleDualRuntimes.size()) ||
        !module || !mLlm->mDualPipelineScheduler) {
        return outputs;
    }
    auto& runtime = mLlm->mDualPipelineRuntimes[pipelineId];
    runtime.activeQnnOpIndices = qnnOpIndices;
    DualPipelineScheduler::PipelineGraphWave graphWave;
    graphWave.pipelineId = pipelineId;
    graphWave.ownerRequestIds = ownerReqIds;
    const int paddedLen = inputs.empty() ? 0 : _packedSeqLen(inputs[0]);
    graphWave.graphs = buildQnnGraphRequestsForSize(
        graphSnapshot, 0, static_cast<int>(graphRequests.size()), -1, paddedLen);
    if (graphWave.graphs.size() != graphRequests.size()) {
        MNN_ERROR("MNN_DUAL_PIPELINE: component graph bucket selection failed for pipeline %d, bucket %d.\n",
                  pipelineId, paddedLen);
        runtime.activeQnnOpIndices = mLlm->mDualPipelineQnnOpIndices;
        return outputs;
    }
    for (auto& request : graphWave.graphs) {
        request.draftGraph = true;
        request.pinResident = true;
    }
    if (!mLlm->mDualPipelineScheduler->beginGraphPrefetchWave({graphWave})) {
        MNN_ERROR("MNN_DUAL_PIPELINE: component graph prefetch failed for pipeline %d.\n", pipelineId);
        runtime.activeQnnOpIndices = mLlm->mDualPipelineQnnOpIndices;
        return outputs;
    }
    if (!mLlm->mDualPipelineScheduler->beginStageWave({pipelineId})) {
        MNN_ERROR("MNN_DUAL_PIPELINE: component stage wave failed for pipeline %d.\n", pipelineId);
        mLlm->mDualPipelineScheduler->cancelGraphPrefetchWave();
        mLlm->mDualPipelineScheduler->finishGraphPrefetchWave();
        runtime.activeQnnOpIndices = mLlm->mDualPipelineQnnOpIndices;
        return outputs;
    }
    {
        Express::ExecutorScope scope(runtime.executor);
        outputs = module->onForward(inputs);
        if (!outputs.empty()) {
            waitModuleOutputs(outputs);
        }
    }
    if (outputs.empty()) {
        mLlm->mDualPipelineScheduler->cancelStageWave();
        mLlm->mDualPipelineScheduler->cancelGraphPrefetchWave();
    }
    const auto stageSnapshot = mLlm->mDualPipelineScheduler->stageSnapshot();
    const bool callbacksExecuted = stageSnapshot.completedStages > 0 &&
        (qnnOpIndices.empty() || stageSnapshot.completedQnnStages > 0);
    const bool stageSucceeded = mLlm->mDualPipelineScheduler->finishStageWave() && callbacksExecuted;
    const bool graphSucceeded = mLlm->mDualPipelineScheduler->finishGraphPrefetchWave();
    runtime.activeQnnOpIndices = mLlm->mDualPipelineQnnOpIndices;
    if (!stageSucceeded || !graphSucceeded) {
        MNN_ERROR("MNN_DUAL_PIPELINE: component wave completion failed for pipeline %d (stage=%d graph=%d).\n",
                  pipelineId, stageSucceeded, graphSucceeded);
        outputs.clear();
    }
    return outputs;
}

void EagleGeneration::loadPackedDraftModule() {
    if (mLlm->mConfig->dual_pipeline_mode()) {
        if (!mLlm->prepareDualPipelineExecutionState()) {
            MNN_ERROR("MNN_DUAL_PIPELINE: failed to prepare Eagle dual execution state.\n");
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return;
        }
        for (int pipelineId = 0; pipelineId < static_cast<int>(mEagleDualRuntimes.size()); ++pipelineId) {
            auto& runtime = mEagleDualRuntimes[pipelineId];
            auto& targetRuntime = mLlm->mDualPipelineRuntimes[pipelineId];
            runtime.batchMeta.reset(new BatchKVMeta);
            runtime.rootModule.reset();
            runtime.cacheOwner.reset();
            runtime.fcModule.reset();
            runtime.modulePool.clear();
            if (!targetRuntime.executor || !targetRuntime.runtimeManager) {
                mContext->status = LlmStatus::INTERNAL_ERROR;
                return;
            }
            Express::ExecutorScope scope(targetRuntime.executor);
            mLlm->applyKVCacheRuntimeHint(targetRuntime.runtimeManager, true, runtime.batchMeta.get());
            runtime.rootModule = _loadPackedEagleModule(mLlm->mConfig->eagle_model(),
                                                        targetRuntime.runtimeManager,
                                                        mEagleModuleConfig);
            runtime.fcModule.reset(Module::load({"fc_hidden"}, {"hidden_states"},
                                                mLlm->mConfig->eagle_fc().c_str(),
                                                targetRuntime.runtimeManager,
                                                &mEagleModuleConfig));
            if (!runtime.rootModule || !runtime.fcModule) {
                MNN_ERROR("MNN_DUAL_PIPELINE: failed to load Eagle modules for pipeline %d.\n", pipelineId);
                mContext->status = LlmStatus::INTERNAL_ERROR;
                return;
            }
        }
        return;
    }
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

std::vector<MNN::Express::VARP> EagleGeneration::eagleForwardRawPacked(const std::vector<PackedDraftKVInfo>& kvInfos, const std::vector<MNN::Express::VARP>& inputs, bool waitAllOutputs, int pipelineId) {
    const bool dualPipeline = pipelineId >= 0;
    if (dualPipeline && (pipelineId >= static_cast<int>(mEagleDualRuntimes.size()) ||
                         !mLlm->mConfig->dual_pipeline_mode())) {
        MNN_ERROR("MNN_DUAL_PIPELINE: Eagle draft has invalid pipeline %d (runtimes=%d enabled=%d).\n",
                  pipelineId, static_cast<int>(mEagleDualRuntimes.size()),
                  mLlm->mConfig->dual_pipeline_mode());
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    auto runtimeManager = mLlm->mRuntimeManager;
    std::shared_ptr<Express::Executor> executor;
    std::shared_ptr<BatchKVMeta>* batchMeta = &mEagleBatchMeta;
    std::shared_ptr<Module>* rootModule = &mEaglePackedRootModule;
    std::shared_ptr<Module>* cacheOwner = &mEaglePackedCacheOwner;
    std::map<std::pair<int, int>, std::shared_ptr<Module>>* modulePool = &mEaglePackedModulePool;
    if (dualPipeline) {
        auto& runtime = mEagleDualRuntimes[pipelineId];
        auto& targetRuntime = mLlm->mDualPipelineRuntimes[pipelineId];
        runtimeManager = targetRuntime.runtimeManager;
        executor = targetRuntime.executor;
        batchMeta = &runtime.batchMeta;
        rootModule = &runtime.rootModule;
        cacheOwner = &runtime.cacheOwner;
        modulePool = &runtime.modulePool;
    }
    if (!runtimeManager) {
        MNN_ERROR("MNN_DUAL_PIPELINE: Eagle draft runtime manager is missing for pipeline %d.\n",
                  pipelineId);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    if (*batchMeta == nullptr) {
        batchMeta->reset(new BatchKVMeta);
    }
    for (auto& kv : kvInfos) {
        (*batchMeta)->setKVCacheInfo(kv.reqId, kv.add, kv.remove, nullptr, 0);
        (*batchMeta)->setKVMetaInfo(kv.reqId, mLlm->mConfig->layer_nums(), 0, 0, "", KVMeta::NoChange);
    }
    mLlm->applyKVCacheRuntimeHint(runtimeManager, true, batchMeta->get());
    std::vector<MNN::Express::VARP> outputs;
    int totalLen = inputs.empty() || inputs[0] == nullptr ? 0 : _packedSeqLen(inputs[0]);
    int maskSize = inputs.size() > 2 ? _varElementSize(inputs[2]) : 0;
    if (totalLen <= 0 || maskSize <= 0) {
        MNN_ERROR("MNN_DUAL_PIPELINE: Eagle draft has invalid packed inputs for pipeline %d "
                  "(inputs=%d total=%d mask=%d).\n",
                  pipelineId, static_cast<int>(inputs.size()), totalLen, maskSize);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        if (!dualPipeline) {
            mLlm->applyKVCacheRuntimeHint(runtimeManager, mLlm->mConfig->packed_attention());
        }
        return outputs;
    }
    int requiredLen = totalLen;
    if (inputs.size() > 1) {
        requiredLen = std::max(requiredLen, _packedSeqLen(inputs[1]));
    }
    const int paddedLen = selectQnnCompatibleBucketSize(mEagleDraftGraphSnapshot, requiredLen);
    if (paddedLen < requiredLen) {
        MNN_ERROR("MNN_QNN: no Eagle draft graph bucket can hold packed length %d.\n", requiredLen);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return outputs;
    }
    VARPS paddedInputs = inputs;
    // Disk embeddings belong to the caller's executor. Materialize their
    // padded input before switching to the lane executor used by draft KV.
    paddedInputs[0] = _padPackedFloatRows(inputs[0], paddedLen);
    if (paddedInputs[0] == nullptr) {
        MNN_ERROR("MNN_DUAL_PIPELINE: Eagle draft embed padding failed for pipeline %d "
                  "(required=%d padded=%d).\n",
                  pipelineId, requiredLen, paddedLen);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return outputs;
    }
    std::unique_ptr<Express::ExecutorScope> executorScope;
    if (executor) {
        executorScope.reset(new Express::ExecutorScope(executor));
    }
    paddedInputs[1] = _padPackedFloatRows(inputs[1], paddedLen);
    if (paddedInputs.size() > 3) {
        paddedInputs[3] = _padPositionIds(inputs[3], paddedLen);
    }
    if (paddedInputs[1] == nullptr ||
        (paddedInputs.size() > 3 && paddedInputs[3] == nullptr)) {
        MNN_ERROR("MNN_DUAL_PIPELINE: Eagle draft padding failed for pipeline %d "
                  "(required=%d padded=%d embeds=%d hidden=%d position=%d).\n",
                  pipelineId, requiredLen, paddedLen, paddedInputs[0] == nullptr ? 0 : 1,
                  paddedInputs[1] == nullptr ? 0 : 1,
                  paddedInputs.size() <= 3 || paddedInputs[3] == nullptr ? 0 : 1);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return outputs;
    }
    if (*rootModule == nullptr) {
        *rootModule = _loadPackedEagleModule(mLlm->mConfig->eagle_model(),
                                             runtimeManager,
                                             mEagleModuleConfig);
    }
    if (*rootModule == nullptr) {
        MNN_ERROR("MNN_DUAL_PIPELINE: Eagle draft root module is missing for pipeline %d.\n",
                  pipelineId);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        if (!dualPipeline) {
            mLlm->applyKVCacheRuntimeHint(runtimeManager, mLlm->mConfig->packed_attention());
        }
        return outputs;
    }
    // Clone shape-specialized draft modules from one root module so CPU packed
    // attention keeps a single KV cache manager across module keys.
    auto createPackedModule = [&]() {
        return std::shared_ptr<Module>(Module::clone(rootModule->get()));
    };
    auto moduleKey = std::make_pair(paddedLen, maskSize);
    auto iter = modulePool->find(moduleKey);
    if (iter == modulePool->end()) {
        if (modulePool->size() >= kMaxPackedDraftModulePool) {
            modulePool->erase(modulePool->begin());
        }
        iter = modulePool->emplace(moduleKey, createPackedModule()).first;
    }
    if (iter->second == nullptr) {
        MNN_ERROR("MNN_DUAL_PIPELINE: Eagle draft clone failed for pipeline %d "
                  "(padded=%d mask=%d).\n",
                  pipelineId, paddedLen, maskSize);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        if (!dualPipeline) {
            mLlm->applyKVCacheRuntimeHint(runtimeManager, mLlm->mConfig->packed_attention());
        }
        return outputs;
    }
    // PackedAttention clones share KV managers. Keep the first executing
    // clone alive because those managers retain its backend pointer.
    if (*cacheOwner == nullptr) {
        *cacheOwner = iter->second;
    }
    std::vector<int> ownerReqIds;
    ownerReqIds.reserve(kvInfos.size());
    for (auto& kv : kvInfos) {
        ownerReqIds.push_back(kv.reqId);
    }
    auto runPackedModule = [&](const std::shared_ptr<Module>& module) {
        if (dualPipeline) {
            return runDualPipelineComponent(pipelineId, module, paddedInputs,
                                            mEagleDraftGraphSnapshot,
                                            mEagleDraftGraphRequests,
                                            mEagleDraftQnnOpIndices,
                                            ownerReqIds);
        }
        return module->onForward(paddedInputs);
    };
    outputs = runPackedModule(iter->second);
    if (outputs.empty()) {
        auto failedModule = iter->second;
        modulePool->erase(moduleKey);
        auto module = createPackedModule();
        if (module == nullptr) {
            MNN_ERROR("MNN_DUAL_PIPELINE: Eagle draft retry clone failed for pipeline %d "
                      "(padded=%d mask=%d).\n",
                      pipelineId, paddedLen, maskSize);
            mContext->status = LlmStatus::INTERNAL_ERROR;
            if (!dualPipeline) {
                mLlm->applyKVCacheRuntimeHint(runtimeManager, mLlm->mConfig->packed_attention());
            }
            return outputs;
        }
        outputs = runPackedModule(module);
        if (outputs.empty()) {
            MNN_ERROR("MNN_DUAL_PIPELINE: Eagle draft component retry returned no outputs for pipeline %d "
                      "(padded=%d mask=%d).\n",
                      pipelineId, paddedLen, maskSize);
            mContext->status = LlmStatus::INTERNAL_ERROR;
            if (!dualPipeline) {
                mLlm->applyKVCacheRuntimeHint(runtimeManager, mLlm->mConfig->packed_attention());
            }
            return outputs;
        }
        if (*cacheOwner == failedModule) {
            *cacheOwner = module;
        }
        (*modulePool)[moduleKey] = std::move(module);
    }
    if (outputs.size() > 1) {
        if (waitAllOutputs) {
            waitModuleOutputs(outputs);
        } else {
            outputs[0]->readMap<float>();
        }
    }
    (*batchMeta)->sync();
    if (!dualPipeline) {
        mLlm->applyKVCacheRuntimeHint(runtimeManager, mLlm->mConfig->packed_attention());
    }
    return outputs;
}

VARPS EagleGeneration::treeDecodingPacked(const EagleGeneration::DraftInfo& draftInfo) {
    auto inputEmbeds = mLlm->embedding(draftInfo.draftTokens);
    int inputLen = draftInfo.draftTokens.size();
    const int paddedLen = mLlm->qnnPaddedCulLen(inputLen);
    if (paddedLen < inputLen) {
        MNN_ERROR("MNN_QNN: no target graph bucket can hold Eagle single tree length %d.\n", inputLen);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    inputEmbeds = _padPackedFloatRows(inputEmbeds, paddedLen);
    auto positionIds = _padPositionIds(draftInfo.positionIds, paddedLen);
    if (inputEmbeds == nullptr || positionIds == nullptr) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, true);
    auto pendingIter = mBasePendingKV.find(draftInfo.reqId);
    PendingBaseKV* pending = pendingIter == mBasePendingKV.end() ? nullptr : &pendingIter->second;
    size_t remove = pending == nullptr ? 0 : pending->remove;
    int* reserve = (pending == nullptr || pending->reserveHost.empty()) ? nullptr : pending->reserveHost.data();
    int reserveSize = pending == nullptr ? 0 : static_cast<int>(pending->reserveHost.size() / 2);
    mLlm->mBatchMeta->setKVCacheInfo(draftInfo.reqId, inputLen, remove, reserve, reserveSize);
    mLlm->mBatchMeta->setKVMetaInfo(draftInfo.reqId, mLlm->mConfig->layer_nums(), 0, 0, "", KVMeta::NoChange);
    auto treeMask = getPackedMask({draftInfo});
    auto moduleKey = std::make_pair(paddedLen, true);
    if (mLlm->mModulePool.find(moduleKey) == mLlm->mModulePool.end()) {
        mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, true);
        mLlm->mModulePool[moduleKey].reset(Module::clone(mLlm->mModule.get()));
    }
    auto outputs = mLlm->mModulePool[moduleKey]->onForward({inputEmbeds, treeMask, positionIds, mLlm->logitsAllIdx});
    if (outputs.size() > 1) {
        waitModuleOutputs(outputs);
    }
    mLlm->mBatchMeta->sync();
    mBasePendingKV.erase(draftInfo.reqId);
    if (paddedLen > inputLen && outputs.size() > 1) {
        outputs[0] = _slicePackedRows(outputs[0], 0, inputLen);
        outputs[1] = _slicePackedRows(outputs[1], 0, inputLen);
    }
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

MNN::Express::VARP EagleGeneration::eagleFCForward(const MNN::Express::VARPS& hiddenStates, int pipelineId) {
    if (hiddenStates.empty()) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return nullptr;
    }
    VARPS packedInputs;
    packedInputs.reserve(hiddenStates.size());
    for (auto& hidden : hiddenStates) {
        if (_packedSeqLen(hidden) <= 0) {
            MNN_ERROR("MNN_DUAL_PIPELINE: Eagle FC received empty hidden state.\n");
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return nullptr;
        }
        packedInputs.push_back(_cloneHiddenToInput(hidden));
    }
    auto packedHidden = packedInputs.size() == 1 ? packedInputs[0] : _Concat(packedInputs, 0);
    const int actualLen = _packedSeqLen(packedHidden);
    const int paddedLen = selectQnnCompatibleBucketSize(mEagleFCGraphSnapshot, actualLen);
    if (paddedLen < actualLen) {
        MNN_ERROR("MNN_QNN: no Eagle FC graph bucket can hold packed length %d.\n", actualLen);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return nullptr;
    }
    packedHidden = _padPackedFloatRows(packedHidden, paddedLen);
    if (packedHidden == nullptr) {
        MNN_ERROR("MNN_DUAL_PIPELINE: Eagle FC failed to pad hidden state.\n");
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return nullptr;
    }
    std::shared_ptr<Module> module = mEagleModules[1];
    std::unique_ptr<Express::ExecutorScope> executorScope;
    if (pipelineId >= 0) {
        if (pipelineId >= static_cast<int>(mEagleDualRuntimes.size()) ||
            !mLlm->mConfig->dual_pipeline_mode()) {
            MNN_ERROR("MNN_DUAL_PIPELINE: Eagle FC has invalid pipeline %d.\n", pipelineId);
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return nullptr;
        }
        auto& targetRuntime = mLlm->mDualPipelineRuntimes[pipelineId];
        module = mEagleDualRuntimes[pipelineId].fcModule;
        if (targetRuntime.executor) {
            executorScope.reset(new Express::ExecutorScope(targetRuntime.executor));
        }
    }
    if (!module) {
        MNN_ERROR("MNN_DUAL_PIPELINE: Eagle FC module is missing for pipeline %d.\n", pipelineId);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return nullptr;
    }
    VARPS outputs;
    if (pipelineId >= 0) {
        outputs = runDualPipelineComponent(pipelineId, module, {packedHidden},
                                           mEagleFCGraphSnapshot,
                                           mEagleFCGraphRequests,
                                           mEagleFCQnnOpIndices,
                                           {});
    } else {
        outputs = module->onForward({packedHidden});
    }
    if (outputs.empty()) {
        MNN_ERROR("MNN_DUAL_PIPELINE: Eagle FC forward returned no outputs for pipeline %d.\n", pipelineId);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return nullptr;
    }
    auto cloned = _cloneHiddenRowsToInput(outputs[0], 0, actualLen);
    if (cloned == nullptr) {
        MNN_ERROR("MNN_DUAL_PIPELINE: Eagle FC output rows %d cannot provide %d rows for pipeline %d.\n",
                  _packedSeqLen(outputs[0]), actualLen, pipelineId);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return nullptr;
    }
    return cloned;
}

std::vector<EagleGeneration::DraftInfo> EagleGeneration::topkGeneratePacked(const std::vector<PackedDraftInput>& inputs, int pipelineId) {
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
        if (input.state == nullptr || input.inputIds.empty() || input.pipelineId != pipelineId) {
            MNN_ERROR("MNN_DUAL_PIPELINE: invalid Eagle draft input for pipeline %d.\n", pipelineId);
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return {};
        }
        VARP inputEmbeds = input.inputEmbeds;
        if (inputEmbeds == nullptr) {
            inputEmbeds = mLlm->embedding(input.inputIds);
        }
        int inputLen = inputEmbeds->getInfo()->dim[0];
        if (_packedSeqLen(input.hiddenStates) != inputLen) {
            MNN_ERROR("MNN_DUAL_PIPELINE: Eagle draft hidden/input length mismatch (%d vs %d).\n",
                      _packedSeqLen(input.hiddenStates), inputLen);
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

    auto packedFcHidden = eagleFCForward(fcInputList, pipelineId);
    if (packedFcHidden == nullptr || _packedSeqLen(packedFcHidden) < totalLen) {
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return {};
    }
    auto inputEmbeds = inputEmbedsList.size() == 1 ? inputEmbedsList[0] : _Concat(inputEmbedsList, 0);
    auto inputHidden = packedFcHidden;
    auto attentionMask = mLlm->gen_attention_mask(calLen);
    auto positionIds = _makePositionIds(positionHost);
    auto outputs = eagleForwardRawPacked(kvInfos, {inputEmbeds, inputHidden, attentionMask, positionIds, mLlm->logitsAllIdx}, true, pipelineId);
    if (outputs.size() < 2) {
        MNN_ERROR("MNN_DUAL_PIPELINE: Eagle draft forward returned %d outputs for pipeline %d.\n",
                  static_cast<int>(outputs.size()), pipelineId);
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
        outputs = eagleForwardRawPacked(stepKVInfos, {stepEmbeds, stepHidden, stepMask, stepPositionIds, mLlm->logitsAllIdx}, true, pipelineId);
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
        info.pipelineId = pipelineId;
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

bool EagleGeneration::prefillDraftPacked(const std::vector<PackedDraftInput>& inputs, int pipelineId) {
    if (inputs.empty()) {
        return true;
    }
    std::vector<VARP> embedsList;
    std::vector<VARP> hiddenList;
    std::vector<int> calLen;
    std::vector<int> positionHost;
    std::vector<PackedDraftKVInfo> kvInfos;
    for (auto& input : inputs) {
        if (input.state == nullptr || input.inputIds.empty() || input.pipelineId != pipelineId ||
            _packedSeqLen(input.hiddenStates) != input.inputIds.size()) {
            MNN_ERROR("MNN_DUAL_PIPELINE: invalid draft prefill input for pipeline %d "
                      "(input pipeline=%d ids=%d hidden=%d state=%d).\n",
                      pipelineId, input.pipelineId, static_cast<int>(input.inputIds.size()),
                      _packedSeqLen(input.hiddenStates), input.state != nullptr);
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
    auto hidden = eagleFCForward(hiddenList, pipelineId);
    if (hidden == nullptr) {
        MNN_ERROR("MNN_DUAL_PIPELINE: draft prefill FC failed for pipeline %d.\n", pipelineId);
        mContext->status = LlmStatus::INTERNAL_ERROR;
        return false;
    }
    auto embeds = embedsList.size() == 1 ? embedsList[0] : _Concat(embedsList, 0);
    auto outputs = eagleForwardRawPacked(kvInfos, {embeds, hidden, mLlm->gen_attention_mask(calLen),
                                                  _makePositionIds(positionHost), mLlm->logitsLastIdx}, false, pipelineId);
    if (outputs.size() < 2) {
        MNN_ERROR("MNN_DUAL_PIPELINE: draft prefill returned %d outputs for pipeline %d.\n",
                  static_cast<int>(outputs.size()), pipelineId);
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
    const bool dualPipelineMode = mLlm->mConfig->dual_pipeline_mode();
    if (dualPipelineMode) {
        for (auto& runtime : mEagleDualRuntimes) {
            if (runtime.batchMeta) {
                runtime.batchMeta->reset();
            }
        }
    } else if (mEagleBatchMeta != nullptr) {
        mEagleBatchMeta->reset();
    }

    struct BatchBaseForward {
        int pipelineId = -1;
        int actualCulLen = 0;
        int paddedCulLen = 0;
        std::shared_ptr<BatchScheduler::Chunk> chunk;
        std::shared_ptr<BatchKVMeta> batchMeta;
        std::shared_ptr<Module> module;
        VARPS outputs;
        VARP logits;
        VARP hiddenStates;
        VARP inputEmbeds;
        VARP attentionMask;
        VARP positionIds;
        std::vector<int> outputLogitsDims;
        std::vector<int> outputHiddenDims;
        std::vector<float> outputLogitsHost;
        std::vector<float> outputHiddenHost;
        std::vector<int> draftOffsets;
        std::vector<int> chunkOffsets;
        std::vector<int> chunkDraftIndices;
        std::vector<int> ownerReqIds;
    };

    std::map<int, int> requestPipelines;
    auto releaseRequestState = [&](int id) {
        mLlm->mScheduler->releaseKVCache(id);
        eagleStates.erase(id);
        mBasePendingKV.erase(id);
        auto pipelineIter = requestPipelines.find(id);
        int pipelineId = pipelineIter == requestPipelines.end() ? -1 : pipelineIter->second;
        if (pipelineId >= 0 && pipelineId < static_cast<int>(mEagleDualRuntimes.size())) {
            auto& runtime = mEagleDualRuntimes[pipelineId];
            if (runtime.batchMeta) {
                runtime.batchMeta->releaseKV(id);
            }
            mLlm->releaseDualPipelineRequestExecution(id);
        } else if (mEagleBatchMeta != nullptr) {
            mEagleBatchMeta->releaseKV(id);
        }
        requestPipelines.erase(id);
    };

    auto buildBatchBaseForward = [&](const std::vector<DraftInfo>& activeDrafts,
                                     const std::shared_ptr<BatchScheduler::Chunk>& chunk,
                                     int pipelineId) -> BatchBaseForward {
        BatchBaseForward result;
        result.pipelineId = pipelineId;
        result.chunk = chunk;
        BatchKVMeta* targetMeta = mLlm->mBatchMeta.get();
        if (pipelineId >= 0) {
            if (pipelineId >= static_cast<int>(mLlm->mDualPipelineRuntimes.size()) ||
                !mLlm->mDualPipelineRuntimes[pipelineId].batchMeta) {
                mContext->status = LlmStatus::INTERNAL_ERROR;
                return result;
            }
            result.batchMeta = mLlm->mDualPipelineRuntimes[pipelineId].batchMeta;
            result.batchMeta->calId.clear();
            targetMeta = result.batchMeta.get();
        } else {
            result.batchMeta = mLlm->mBatchMeta;
        }
        auto setTargetKV = [&](int reqId, int add, size_t remove, int* reserve, int reserveSize) {
            targetMeta->setKVCacheInfo(reqId, add, remove, reserve, reserveSize);
            targetMeta->setKVMetaInfo(reqId, mLlm->mConfig->layer_nums(), 0, 0, "", KVMeta::NoChange);
            result.ownerReqIds.push_back(reqId);
        };
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
            setTargetKV(info.reqId, len, remove, reserve, reserveSize);
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
                setTargetKV(reqId, len, 0, nullptr, 0);
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

        result.actualCulLen = culLen;
        const int requiredSize = std::max(culLen, static_cast<int>(result.ownerReqIds.size()));
        result.paddedCulLen = mLlm->qnnPaddedCulLen(requiredSize);
        if (result.paddedCulLen < culLen) {
            MNN_ERROR("MNN_QNN: no target graph bucket can hold Eagle target length %d.\n", culLen);
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return result;
        }
        if (result.paddedCulLen > static_cast<int>(positionHost.size())) {
            positionHost.resize(result.paddedCulLen, 0);
        }
        result.inputEmbeds = mLlm->embedding(packedTokens, calLen, result.paddedCulLen);
        result.attentionMask = attentionMask;
        result.positionIds = _makePositionIds(positionHost);
        auto moduleKey = std::make_pair(result.paddedCulLen, true);
        if (pipelineId < 0) {
            mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, true);
            if (mLlm->mModulePool.find(moduleKey) == mLlm->mModulePool.end()) {
                mLlm->mModulePool[moduleKey].reset(Module::clone(mLlm->mModule.get()));
            }
            result.module = mLlm->mModulePool[moduleKey];
        }
        return result;
    };

    auto executeBatchBaseWave = [&](std::vector<BatchBaseForward>& tasks) -> bool {
        if (tasks.empty()) {
            return false;
        }
        const bool dualPipeline = tasks[0].pipelineId >= 0;
        if (!dualPipeline) {
            auto& task = tasks[0];
            if (!task.module) {
                return false;
            }
            task.outputs = task.module->onForward(
                {task.inputEmbeds, task.attentionMask, task.positionIds, mLlm->logitsAllIdx});
            if (task.outputs.size() < 2) {
                return false;
            }
        } else {
            std::vector<DualPipelineScheduler::PipelineGraphWave> graphWaves;
            graphWaves.reserve(tasks.size());
            for (auto& task : tasks) {
                DualPipelineScheduler::PipelineGraphWave graphWave;
                graphWave.pipelineId = task.pipelineId;
                graphWave.ownerRequestIds = task.ownerReqIds;
                graphWave.graphs = buildQnnGraphRequestsForSize(
                    mLlm->mDualPipelineGraphSnapshot,
                    0,
                    static_cast<int>(mLlm->mDualPipelineQnnGraphRequests.size()),
                    -1,
                    task.paddedCulLen);
                if (graphWave.graphs.size() != mLlm->mDualPipelineQnnGraphRequests.size()) {
                    MNN_ERROR("MNN_DUAL_PIPELINE: target graph bucket selection failed for pipeline %d, bucket %d.\n",
                              task.pipelineId, task.paddedCulLen);
                    return false;
                }
                graphWaves.push_back(std::move(graphWave));
            }
            if (!mLlm->mDualPipelineScheduler ||
                !mLlm->mDualPipelineScheduler->beginGraphPrefetchWave(graphWaves)) {
                MNN_ERROR("MNN_DUAL_PIPELINE: target graph prefetch wave failed.\n");
                return false;
            }
            bool buildFailed = false;
            for (auto& task : tasks) {
                task.module = mLlm->getDualPipelineModule(
                    task.pipelineId, std::make_pair(task.paddedCulLen, true));
                if (!task.module) {
                    MNN_ERROR("MNN_DUAL_PIPELINE: target module creation failed for pipeline %d, bucket %d.\n",
                              task.pipelineId, task.paddedCulLen);
                    buildFailed = true;
                    break;
                }
            }
            if (buildFailed) {
                mLlm->mDualPipelineScheduler->cancelGraphPrefetchWave();
                mLlm->mDualPipelineScheduler->finishGraphPrefetchWave();
                return false;
            }
            std::vector<int> pipelineIds;
            pipelineIds.reserve(tasks.size());
            for (auto& task : tasks) {
                pipelineIds.push_back(task.pipelineId);
            }
            if (!mLlm->mDualPipelineScheduler->beginStageWave(pipelineIds)) {
                MNN_ERROR("MNN_DUAL_PIPELINE: target stage wave failed.\n");
                mLlm->mDualPipelineScheduler->cancelGraphPrefetchWave();
                mLlm->mDualPipelineScheduler->finishGraphPrefetchWave();
                return false;
            }
            std::vector<std::thread> workers;
            workers.reserve(tasks.size());
            for (size_t i = 0; i < tasks.size(); ++i) {
                workers.emplace_back([&, i]() {
                    auto& task = tasks[i];
                    auto& runtime = mLlm->mDualPipelineRuntimes[task.pipelineId];
                    Express::ExecutorScope scope(runtime.executor);
                    mLlm->applyKVCacheRuntimeHint(runtime.runtimeManager, true, task.batchMeta.get());
                    runtime.activeQnnOpIndices = mLlm->mDualPipelineQnnOpIndices;
                    task.outputs = task.module->onForward(
                        {task.inputEmbeds, task.attentionMask, task.positionIds, mLlm->logitsAllIdx});
                    if (!task.outputs.empty()) {
                        waitModuleOutputs(task.outputs);
                    }
                    const bool outputsCaptured = task.outputs.size() >= 2 &&
                        _captureFloatRows(task.outputs[0], task.actualCulLen,
                                          task.outputLogitsDims, task.outputLogitsHost) &&
                        _captureFloatRows(task.outputs[1], task.actualCulLen,
                                          task.outputHiddenDims, task.outputHiddenHost);
                    if (!outputsCaptured) {
                        mLlm->mDualPipelineScheduler->cancelStageWave();
                        mLlm->mDualPipelineScheduler->cancelGraphPrefetchWave();
                    }
                });
            }
            for (auto& worker : workers) {
                worker.join();
            }
            const auto stageSnapshot = mLlm->mDualPipelineScheduler->stageSnapshot();
            const bool callbacksExecuted = stageSnapshot.completedStages > 0 &&
                (mLlm->mDualPipelineQnnOpIndices.empty() || stageSnapshot.completedQnnStages > 0);
            const bool stageSucceeded = mLlm->mDualPipelineScheduler->finishStageWave() && callbacksExecuted;
            const bool graphSucceeded = mLlm->mDualPipelineScheduler->finishGraphPrefetchWave();
            if (!stageSucceeded || !graphSucceeded) {
                MNN_ERROR("MNN_DUAL_PIPELINE: target wave completion failed (stage=%d graph=%d).\n",
                          stageSucceeded, graphSucceeded);
                return false;
            }
            for (auto& task : tasks) {
                if (task.outputs.size() < 2) {
                    return false;
                }
                auto logits = _makeFloatInput(task.outputLogitsDims, task.outputLogitsHost);
                auto hidden = _makeFloatInput(task.outputHiddenDims, task.outputHiddenHost);
                if (logits == nullptr || hidden == nullptr) {
                    return false;
                }
                task.outputs = {logits, hidden};
            }
        }

        for (auto& task : tasks) {
            task.batchMeta->sync();
            if (dualPipeline) {
                for (int reqId : task.ownerReqIds) {
                    _copyBatchKVMetaEntry(mLlm->mBatchMeta.get(), task.batchMeta.get(), reqId, false);
                }
            }
            for (int reqId : task.ownerReqIds) {
                mBasePendingKV.erase(reqId);
            }
            task.logits = _Squeeze(task.outputs[0], {0});
            task.hiddenStates = task.outputs[1];
        }
        if (mLlm->mBatchMeta) {
            mLlm->mBatchMeta->calId.clear();
        }
        return true;
    };

    while (mLlm->mScheduler->hasValidWork() || !draftInfos.empty()) {
        std::set<int> skipReqIds;
        std::vector<DraftInfo> ordinaryDrafts;
        std::array<std::vector<DraftInfo>, 2> laneDrafts;
        for (auto& kv : draftInfos) {
            if (mLlm->mScheduler->isFinished(kv.first)) {
                continue;
            }
            skipReqIds.insert(kv.first);
            if (dualPipelineMode) {
                if (kv.second.pipelineId < 0 || kv.second.pipelineId >= static_cast<int>(laneDrafts.size())) {
                    mContext->status = LlmStatus::INTERNAL_ERROR;
                    break;
                }
                laneDrafts[kv.second.pipelineId].push_back(kv.second);
            } else {
                ordinaryDrafts.push_back(kv.second);
            }
        }
        if (mContext->status == LlmStatus::INTERNAL_ERROR) {
            break;
        }

        std::vector<std::shared_ptr<BatchScheduler::Chunk>> wave;
        if (dualPipelineMode) {
            wave = mLlm->mScheduler->scheduleWave(-1, 4, skipReqIds);
        } else {
            auto chunk = mLlm->mScheduler->schedule(-1, 4, skipReqIds);
            if (chunk) {
                wave.push_back(chunk);
            }
        }
        std::array<std::shared_ptr<BatchScheduler::Chunk>, 2> laneChunks;
        for (auto& chunk : wave) {
            int pipelineId = dualPipelineMode ? chunk->pipelineId : -1;
            if (dualPipelineMode && (pipelineId < 0 || pipelineId >= static_cast<int>(laneChunks.size()))) {
                mContext->status = LlmStatus::INTERNAL_ERROR;
                break;
            }
            if (dualPipelineMode) {
                laneChunks[pipelineId] = chunk;
            }
            for (int reqId : chunk->reqId) {
                requestPipelines[reqId] = pipelineId;
            }
        }
        if (mContext->status == LlmStatus::INTERNAL_ERROR) {
            break;
        }

        std::vector<BatchBaseForward> bases;
        if (dualPipelineMode) {
            for (int pipelineId = 0; pipelineId < static_cast<int>(laneDrafts.size()); ++pipelineId) {
                if (laneDrafts[pipelineId].empty() && !laneChunks[pipelineId]) {
                    continue;
                }
                bases.push_back(buildBatchBaseForward(laneDrafts[pipelineId], laneChunks[pipelineId], pipelineId));
            }
        } else if (!ordinaryDrafts.empty() || !wave.empty()) {
            bases.push_back(buildBatchBaseForward(ordinaryDrafts, wave.empty() ? nullptr : wave[0], -1));
        }
        if (bases.empty()) {
            MNN_ERROR("MNN_DUAL_PIPELINE: no base forward tasks were scheduled.\n");
            break;
        }
        bool baseBuildFailed = false;
        int activeVerificationCount = 0;
        for (auto& base : bases) {
            baseBuildFailed = baseBuildFailed || base.inputEmbeds == nullptr || base.batchMeta == nullptr;
            activeVerificationCount += dualPipelineMode
                ? static_cast<int>(laneDrafts[base.pipelineId].size())
                : static_cast<int>(ordinaryDrafts.size());
        }
        MNN::Timer targetWaveTimer;
        const bool baseForwardOk = !baseBuildFailed && executeBatchBaseWave(bases);
        if (!baseForwardOk) {
            MNN_ERROR("MNN_DUAL_PIPELINE: base forward failed (build=%d, pipelines=%d).\n",
                      baseBuildFailed, static_cast<int>(bases.size()));
            mContext->status = LlmStatus::INTERNAL_ERROR;
            break;
        }
        if (activeVerificationCount > 0) {
            mSpecContext.target_time_us += targetWaveTimer.durationInUs();
        }

        std::map<int, DraftInfo> nextDrafts;
        std::vector<PackedDraftInput> nextDraftInputs;
        for (auto& base : bases) {
            const std::vector<DraftInfo>& activeDrafts = dualPipelineMode
                ? laneDrafts[base.pipelineId] : ordinaryDrafts;
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
                mSpecContext.steps++;
                mSpecContext.accepted += acceptInfo.acceptTokens.size();
                mSpecContext.accept_len_freq[static_cast<int>(acceptInfo.acceptTokens.size())]++;
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
                draftInput.pipelineId = base.pipelineId;
                draftInput.state = &eagleStates[id];
                draftInput.inputIds = acceptInfo.acceptTokens;
                draftInput.hiddenStates = acceptHiddenState;
                nextDraftInputs.push_back(std::move(draftInput));
            }

            auto chunk = base.chunk;
            if (chunk == nullptr) {
                continue;
            }
            std::set<int> finalChunkIndices(base.chunkDraftIndices.begin(), base.chunkDraftIndices.end());
            std::vector<PackedDraftInput> draftPrefillInputs;
            for (int i = 0; i < chunk->pos.size(); ++i) {
                int reqLen = chunk->calLen[i];
                if (reqLen > 1) {
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
                    draftInput.pipelineId = base.pipelineId;
                    draftInput.state = &eagleStates[id];
                    draftInput.inputIds.assign(inputIds[inputIndex].begin() + pos + 1,
                                               inputIds[inputIndex].begin() + pos + reqLen + 1);
                    draftInput.hiddenStates = _slicePackedRows(base.hiddenStates, base.chunkOffsets[i], reqLen);
                    draftPrefillInputs.push_back(std::move(draftInput));
                }
            }
            MNN::Timer draftPrefillTimer;
            if (mContext->status == LlmStatus::INTERNAL_ERROR ||
                !prefillDraftPacked(draftPrefillInputs, base.pipelineId)) {
                break;
            }
            if (!draftPrefillInputs.empty()) {
                const uint64_t draftPrefillUs = draftPrefillTimer.durationInUs();
                mSpecContext.draft_time_us += draftPrefillUs;
                mSpecContext.draft_prefill_time_us += draftPrefillUs;
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
                draftInput.pipelineId = base.pipelineId;
                draftInput.state = &stateInfo;
                draftInput.inputIds = std::move(draftInputIds);
                draftInput.hiddenStates = hiddenSlice;
                draftInput.inputEmbeds = eagleInputEmbed;
                nextDraftInputs.push_back(std::move(draftInput));
            }
        }
        if (mContext->status == LlmStatus::INTERNAL_ERROR) {
            break;
        }

        MNN::Timer draftDecodeTimer;
        if (dualPipelineMode) {
            for (int pipelineId = 0; pipelineId < static_cast<int>(laneDrafts.size()); ++pipelineId) {
                std::vector<PackedDraftInput> laneInputs;
                for (auto& input : nextDraftInputs) {
                    if (input.pipelineId == pipelineId) {
                        laneInputs.push_back(input);
                    }
                }
                auto packedDrafts = topkGeneratePacked(laneInputs, pipelineId);
                for (auto& draft : packedDrafts) {
                    mSpecContext.draft += draft.draftTokens.size();
                    nextDrafts[draft.reqId] = std::move(draft);
                }
            }
        } else if (!nextDraftInputs.empty()) {
            auto packedDrafts = topkGeneratePacked(nextDraftInputs, -1);
            for (auto& draft : packedDrafts) {
                mSpecContext.draft += draft.draftTokens.size();
                nextDrafts[draft.reqId] = std::move(draft);
            }
        }
        if (!nextDraftInputs.empty()) {
            const uint64_t draftDecodeUs = draftDecodeTimer.durationInUs();
            mSpecContext.draft_time_us += draftDecodeUs;
            mSpecContext.draft_decode_time_us += draftDecodeUs;
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
        releaseRequestState(id);
        mLlm->mScheduler->releaseReq(id);
    }
    return ret;
}

} // namespace Transformer
} // namespace MNN
