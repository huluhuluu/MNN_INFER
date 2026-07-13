//
//  PackedAttentionBufExecution.cpp
//  MNN
//
//  Created by MNN on 2025/03/20.
//  Copyright © 2018, Alibaba Group Holding Limited.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "PackedAttentionBufExecution.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace MNN {
namespace OpenCL {

namespace {

static inline int reserveLength(const KVMeta* meta) {
    if (meta == nullptr || meta->reserve == nullptr || meta->n_reserve <= 0) {
        return 0;
    }
    int total = 0;
    for (int i = 0; i < meta->n_reserve; ++i) {
        total += meta->reserve[2 * i + 1];
    }
    return total;
}

static inline int retainedLength(const KVMeta* meta, int currentPast) {
    if (meta == nullptr) {
        return currentPast;
    }
    int remove = std::min<int>(meta->remove, currentPast);
    int kept = currentPast - remove;
    if (meta->n_reserve > 0) {
        kept += reserveLength(meta);
    }
    return std::max(kept, 0);
}

static inline int packedSoftmaxLocalSize(int maxWorkGroupSize) {
    return std::max(std::min<int>(maxWorkGroupSize, 64), 1);
}

static inline void acquireReleasePackedAttentionTemps(OpenCLBackend* backend, Tensor* q, Tensor* qk,
                                                      Tensor* softmax, Tensor* mask) {
    if (backend == nullptr) {
        return;
    }
    if (q != nullptr) {
        backend->onAcquireBuffer(q, Backend::DYNAMIC_IN_EXECUTION);
    }
    if (qk != nullptr) {
        backend->onAcquireBuffer(qk, Backend::DYNAMIC_IN_EXECUTION);
    }
    if (softmax != nullptr) {
        backend->onAcquireBuffer(softmax, Backend::DYNAMIC_IN_EXECUTION);
    }
    if (mask != nullptr) {
        backend->onAcquireBuffer(mask, Backend::DYNAMIC_IN_EXECUTION);
    }
    if (q != nullptr) {
        backend->onReleaseBuffer(q, Backend::DYNAMIC_IN_EXECUTION);
    }
    if (qk != nullptr) {
        backend->onReleaseBuffer(qk, Backend::DYNAMIC_IN_EXECUTION);
    }
    if (softmax != nullptr) {
        backend->onReleaseBuffer(softmax, Backend::DYNAMIC_IN_EXECUTION);
    }
    if (mask != nullptr) {
        backend->onReleaseBuffer(mask, Backend::DYNAMIC_IN_EXECUTION);
    }
}

} // namespace

// ============================================================================
// PackedAttentionBufExecution
// ============================================================================

void PackedAttentionBufExecution::registerResetCallback() {
    if (mMeta == nullptr || mBatchKVCacheManager == nullptr) {
        return;
    }
    std::weak_ptr<BatchKVCacheCLManager> weakManager = mBatchKVCacheManager;
    mMeta->registerResetCallback(mBatchKVCacheManager.get(), [weakManager]() {
        auto manager = weakManager.lock();
        if (manager != nullptr) {
            manager->onClear();
        }
    });
}

void PackedAttentionBufExecution::registerReleaseCallback() {
    if (mMeta == nullptr || mBatchKVCacheManager == nullptr) {
        return;
    }
    std::weak_ptr<BatchKVCacheCLManager> weakManager = mBatchKVCacheManager;
    mMeta->registerReleaseCallback(mBatchKVCacheManager.get(), [weakManager](int reqId) {
        auto manager = weakManager.lock();
        if (manager != nullptr) {
            manager->release(reqId);
        }
    });
}

PackedAttentionBufExecution::PackedAttentionBufExecution(const MNN::Op *op, Backend *backend, bool kv_cache)
    : CommonExecution(backend, op), mMeta((BatchKVMeta*)(backend->getMetaPtr())),
      mOpenCLBackend(static_cast<OpenCLBackend*>(backend)),
      mBatchKVCacheManager(new BatchKVCacheCLManager(backend, kv_cache)),
      mNeedKvCache(kv_cache) {
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("softmax_buf", "softmax_buf", {"-DSOFTMAX_LOCAL_SIZE=512"}, mOpenCLBackend->getPrecision());
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
    registerResetCallback();
    registerReleaseCallback();
}

PackedAttentionBufExecution::PackedAttentionBufExecution(std::shared_ptr<BatchKVCacheCLManager> manager,
                                                         const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op), mMeta((BatchKVMeta*)(backend->getMetaPtr())),
      mOpenCLBackend(static_cast<OpenCLBackend*>(backend)),
      mBatchKVCacheManager(std::move(manager)) {
    mNeedKvCache = mBatchKVCacheManager != nullptr && mBatchKVCacheManager->needKVCache();
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("softmax_buf", "softmax_buf", {"-DSOFTMAX_LOCAL_SIZE=512"}, mOpenCLBackend->getPrecision());
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
    registerResetCallback();
    registerReleaseCallback();
}

ErrorCode PackedAttentionBufExecution::init() {
    return NO_ERROR;
}

ErrorCode PackedAttentionBufExecution::onResize(const std::vector<Tensor *> &inputs,
                                                const std::vector<Tensor *> &outputs) {
    (void)outputs;
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    mMeta = (BatchKVMeta*)(mOpenCLBackend->getMetaPtr());

    auto query = inputs[0];
    auto key = inputs[1];
    mNumHead = query->length(2);
    mHeadDim = query->length(3);
    mKvNumHead = key->length(2);
    mGroupSize = mNumHead / mKvNumHead;
    mScale = 1.0f / std::sqrt((float)mHeadDim);
    mHasMask = inputs.size() >= 4;
    mIsAddMask = mHasMask && inputs[3]->getType() == halide_type_of<float>();

    std::vector<int> reqIds = (mMeta != nullptr && !mMeta->calId.empty()) ? mMeta->calId : std::vector<int>{0};
    size_t totalQElems = 0;
    size_t totalQkElems = 0;
    size_t totalMaskElems = 0;
    int maskElementSize = (mHasMask && !inputs[3]->shape().empty()) ? inputs[3]->elementSize() : 0;
    int squareMaskSize = 0;
    int fullMaskSize = 0;

    for (int reqId : reqIds) {
        const KVMeta* meta = nullptr;
        if (mMeta != nullptr) {
            auto it = mMeta->mMetas.find(reqId);
            if (it != mMeta->mMetas.end()) {
                meta = it->second;
            }
        }
        int seqLen = meta != nullptr ? static_cast<int>(meta->add) : query->length(1);
        int previous = meta != nullptr ? static_cast<int>(meta->previous) : 0;
        int kvSeqLen = retainedLength(meta, previous) + seqLen;
        int seqLen4 = ROUND_UP(seqLen, 4);
        totalQElems += (size_t)mNumHead * mHeadDim * seqLen4;
        totalQkElems += (size_t)mNumHead * kvSeqLen * seqLen4;
        if (mHasMask) {
            squareMaskSize += seqLen * seqLen;
            fullMaskSize += seqLen * kvSeqLen;
        }
    }
    const bool useFullMask = maskElementSize > 0 && maskElementSize == fullMaskSize && fullMaskSize != squareMaskSize;
    if (mHasMask) {
        for (int reqId : reqIds) {
            const KVMeta* meta = nullptr;
            if (mMeta != nullptr) {
                auto it = mMeta->mMetas.find(reqId);
                if (it != mMeta->mMetas.end()) {
                    meta = it->second;
                }
            }
            int seqLen = meta != nullptr ? static_cast<int>(meta->add) : query->length(1);
            int previous = meta != nullptr ? static_cast<int>(meta->previous) : 0;
            int kvSeqLen = retainedLength(meta, previous) + seqLen;
            int maskStride = useFullMask ? kvSeqLen : seqLen;
            totalMaskElems += (size_t)ROUND_UP(seqLen, 4) * ROUND_UP(maskStride, 4);
        }
    }

    mBatchKVCacheManager->setArgs(mNumHead, mKvNumHead, mHeadDim);

    mTempQ.reset(Tensor::createDevice<float>({(int)std::max<size_t>(totalQElems, 1)}));
    mTempQK.reset(Tensor::createDevice<float>({(int)std::max<size_t>(totalQkElems, 1)}));
    mTempSoftMax.reset(Tensor::createDevice<float>({(int)std::max<size_t>(totalQkElems, 1)}));

    if (mHasMask) {
        if (mIsAddMask) {
            mTempMask.reset(Tensor::createDevice<float>({(int)std::max<size_t>(totalMaskElems, 1)}));
        } else {
            mTempMask.reset(Tensor::createDevice<int>({(int)std::max<size_t>(totalMaskElems, 1)}));
        }
    } else {
        mTempMask.reset();
    }
    acquireReleasePackedAttentionTemps(mOpenCLBackend, mTempQ.get(), mTempQK.get(),
                                       mTempSoftMax.get(), mTempMask.get());

    if (reqIds.size() > mReqMetaCapacity) {
        mReqMetaCapacity = reqIds.size();
        size_t bytes = mReqMetaCapacity * 4 * sizeof(int);
        mReqMeta0.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr));
        mReqMeta1.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr));
        mReqMeta2.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr));
    }

    std::set<std::string> buildOptions;
    if (mOpenCLBackend->getPrecision() != BackendConfig::Precision_High) {
        buildOptions.insert("-DOPENCL_ENABLE_FP16");
        buildOptions.insert("-DUSE_FP16=1");
    }
    if (mGroupSize > 1) {
        buildOptions.insert("-DNUMHEAD_GROUP_SIZE=" + std::to_string(mGroupSize));
        buildOptions.insert("-DGROUP_SIZE=" + std::to_string(mGroupSize));
    }

    mKernelRearrangeQ = runtime->buildKernel("attention_buf", "rearrange_q_packed", buildOptions, mOpenCLBackend->getPrecision());
    mKernelRearrangeK = runtime->buildKernel("attention_buf", "rearrange_k_packed", buildOptions, mOpenCLBackend->getPrecision());
    mKernelRearrangeV = runtime->buildKernel("attention_buf", "rearrange_v_packed", buildOptions, mOpenCLBackend->getPrecision());
    if (mKernelRearrangeQ == nullptr || mKernelRearrangeK == nullptr || mKernelRearrangeV == nullptr) {
        return INVALID_VALUE;
    }

    if (mHasMask) {
        std::set<std::string> maskBuildOptions = buildOptions;
        maskBuildOptions.insert(mIsAddMask ? "-DADD_MASK" : "-DSET_MASK");
        mKernelRearrangeMask = runtime->buildKernel("attention_buf", "rearrange_mask_shortprefill_packed",
                                                    maskBuildOptions, mOpenCLBackend->getPrecision());
        if (mKernelRearrangeMask == nullptr) {
            return INVALID_VALUE;
        }
    } else {
        mKernelRearrangeMask = nullptr;
    }

    std::set<std::string> qkBuildOptions = buildOptions;
    if (mHasMask) {
        qkBuildOptions.insert(mIsAddMask ? "-DADD_MASK" : "-DSET_MASK");
    }
    mKernelQk = runtime->buildKernel("attention_buf", "matmul_qk_div_mask_prefill_packed",
                                     qkBuildOptions, mOpenCLBackend->getPrecision());
    if (mKernelQk == nullptr) {
        return INVALID_VALUE;
    }

    int softmaxLocalSize = packedSoftmaxLocalSize(mMaxWorkGroupSize);
    std::set<std::string> softmaxBuildOptions;
    softmaxBuildOptions.insert("-DSOFTMAX_LOCAL_SIZE=" + std::to_string(softmaxLocalSize));
    mKernelSoftmax = runtime->buildKernel("attention_buf", "softmax_v4_buf_packed",
                                          softmaxBuildOptions, mOpenCLBackend->getPrecision());
    if (mKernelSoftmax == nullptr) {
        return INVALID_VALUE;
    }

    mKernelQkv = runtime->buildKernel("attention_buf", "matmul_qkv_prefill_packed",
                                      buildOptions, mOpenCLBackend->getPrecision());
    if (mKernelQkv == nullptr) {
        return INVALID_VALUE;
    }

    std::set<std::string> decodeBuildOptions = buildOptions;
    decodeBuildOptions.insert("-DNUMHEAD_GROUP_SIZE=" + std::to_string(mGroupSize));
    mKernelQkDecode = runtime->buildKernel("attention_buf", "matmul_qk_decode_packed",
                                           decodeBuildOptions, mOpenCLBackend->getPrecision());
    if (mKernelQkDecode == nullptr) {
        return INVALID_VALUE;
    }

    mKernelSoftmaxDecode = runtime->buildKernel("attention_buf", "softmax_in1_buf_packed",
                                                softmaxBuildOptions, mOpenCLBackend->getPrecision());
    if (mKernelSoftmaxDecode == nullptr) {
        return INVALID_VALUE;
    }

    std::set<std::string> qkvDecodeBuildOptions = decodeBuildOptions;
    qkvDecodeBuildOptions.insert("-DLOOP_UNROLL_4");
    mKernelQkvDecode = runtime->buildKernel("attention_buf", "matmul_qkv_decode_b8_packed",
                                            qkvDecodeBuildOptions, mOpenCLBackend->getPrecision());
    if (mKernelQkvDecode == nullptr) {
        return INVALID_VALUE;
    }

    return NO_ERROR;
}

ErrorCode PackedAttentionBufExecution::onExecute(const std::vector<Tensor *> &inputs,
                                                 const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto& queue = runtime->commandQueue();
    mMeta = (BatchKVMeta*)(mOpenCLBackend->getMetaPtr());

    if (mKernelRearrangeQ == nullptr || mKernelRearrangeK == nullptr || mKernelRearrangeV == nullptr ||
        mKernelQk == nullptr || mKernelSoftmax == nullptr || mKernelQkv == nullptr ||
        mKernelQkDecode == nullptr || mKernelSoftmaxDecode == nullptr || mKernelQkvDecode == nullptr ||
        mReqMeta0 == nullptr || mReqMeta1 == nullptr || mReqMeta2 == nullptr) {
        MNN_ERROR("PackedAttentionBufExecution kernel or metadata buffer is not initialized\n");
        return NO_EXECUTION;
    }

    if (mNeedKvCache && mMeta != nullptr && !mBatchKVCacheManager->ensureForExecute(mMeta)) {
        MNN_ERROR("PackedAttentionBufExecution failed to prepare shared KV arena\n");
        return NO_EXECUTION;
    }

    std::vector<int> reqIds = (mMeta != nullptr && !mMeta->calId.empty()) ? mMeta->calId : std::vector<int>{0};
    if (reqIds.empty()) {
        return NO_ERROR;
    }

    const cl::Buffer* sharedKeyBuffer = mBatchKVCacheManager->key();
    const cl::Buffer* sharedValueBuffer = mBatchKVCacheManager->value();
    if ((sharedKeyBuffer == nullptr || sharedValueBuffer == nullptr) && mNeedKvCache) {
        MNN_ERROR("PackedAttentionBufExecution shared KV buffers are null\n");
        return NO_EXECUTION;
    }

    std::vector<int> meta0(reqIds.size() * 4, 0);
    std::vector<int> meta1(reqIds.size() * 4, 0);
    std::vector<int> meta2(reqIds.size() * 4, 0);

    int maxSeqLen = 1;
    int maxKvSeqLen = 1;
    int totalSeqOffset = 0;
    int maskInputOffset = 0;
    int qBase = 0;
    int qkBase = 0;
    int maskBase = 0;
    int maxMaskStride = 1;
    std::vector<int> seqLens(reqIds.size(), 0);
    std::vector<int> pastLens(reqIds.size(), 0);
    std::vector<int> maxLens(reqIds.size(), 0);
    std::vector<int> kvSeqLens(reqIds.size(), 0);
    bool allDecode = mNeedKvCache && mHeadDim % 8 == 0;

    for (size_t i = 0; i < reqIds.size(); ++i) {
        const int reqId = reqIds[i];
        const KVMeta* meta = nullptr;
        if (mMeta != nullptr) {
            auto it = mMeta->mMetas.find(reqId);
            if (it != mMeta->mMetas.end()) {
                meta = it->second;
            }
        }
        const int seqLen = meta != nullptr ? static_cast<int>(meta->add) : inputs[0]->length(1);
        const int pastLen = mNeedKvCache ? mBatchKVCacheManager->pastKvLength(reqId) : 0;
        const int maxLen = mNeedKvCache ? mBatchKVCacheManager->maxLength(reqId) : ROUND_UP(seqLen, 4);
        const int kvSeqLen = pastLen + seqLen;
        seqLens[i] = seqLen;
        pastLens[i] = pastLen;
        maxLens[i] = maxLen;
        kvSeqLens[i] = kvSeqLen;
        allDecode = allDecode && seqLen == 1;
        maxSeqLen = std::max(maxSeqLen, seqLen);
        maxKvSeqLen = std::max(maxKvSeqLen, kvSeqLen);
    }

    int maskElementSize = (mHasMask && !inputs[3]->shape().empty()) ? inputs[3]->elementSize() : 0;
    int squareMaskSize = 0;
    int fullMaskSize = 0;
    if (maskElementSize > 0) {
        for (size_t i = 0; i < reqIds.size(); ++i) {
            squareMaskSize += seqLens[i] * seqLens[i];
            fullMaskSize += seqLens[i] * kvSeqLens[i];
        }
    }
    const bool useFullMask = maskElementSize > 0 && maskElementSize == fullMaskSize && fullMaskSize != squareMaskSize;

    for (size_t i = 0; i < reqIds.size(); ++i) {
        const int reqId = reqIds[i];
        const int seqLen = seqLens[i];
        const int pastLen = pastLens[i];
        const int maxLen = maxLens[i];
        const int kvSeqLen = kvSeqLens[i];
        const int seqLen4 = ROUND_UP(seqLen, 4);
        const int maskStride = useFullMask ? kvSeqLen : seqLen;
        meta0[i * 4 + 0] = totalSeqOffset;
        meta0[i * 4 + 1] = seqLen;
        meta0[i * 4 + 2] = pastLen;
        meta0[i * 4 + 3] = maxLen;

        meta1[i * 4 + 0] = qBase;
        meta1[i * 4 + 1] = qkBase;
        meta1[i * 4 + 2] = mNeedKvCache ? mBatchKVCacheManager->keyOffset(reqId) : 0;
        meta1[i * 4 + 3] = mNeedKvCache ? mBatchKVCacheManager->valueOffset(reqId) : 0;

        meta2[i * 4 + 0] = maskInputOffset;
        meta2[i * 4 + 1] = maskBase;
        meta2[i * 4 + 2] = kvSeqLen;
        meta2[i * 4 + 3] = maskStride;

        totalSeqOffset += seqLen;
        maskInputOffset += seqLen * maskStride;
        qBase += mNumHead * mHeadDim * seqLen4;
        qkBase += mNumHead * kvSeqLen * (allDecode ? 1 : seqLen4);
        maskBase += seqLen4 * ROUND_UP(maskStride, 4);
        maxMaskStride = std::max(maxMaskStride, maskStride);
    }

    bool tempResized = false;
    auto ensureFloatTemp = [&](std::shared_ptr<Tensor>& tensor, size_t elems) -> bool {
        elems = std::max<size_t>(elems, 1);
        if (elems > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return false;
        }
        if (tensor != nullptr && tensor->elementSize() >= elems) {
            return true;
        }
        tensor.reset(Tensor::createDevice<float>({static_cast<int>(elems)}));
        tempResized = true;
        return true;
    };
    auto ensureMaskTemp = [&](size_t elems) -> bool {
        if (!mHasMask) {
            return true;
        }
        elems = std::max<size_t>(elems, 1);
        if (elems > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return false;
        }
        if (mTempMask != nullptr && mTempMask->elementSize() >= elems) {
            return true;
        }
        if (mIsAddMask) {
            mTempMask.reset(Tensor::createDevice<float>({static_cast<int>(elems)}));
        } else {
            mTempMask.reset(Tensor::createDevice<int>({static_cast<int>(elems)}));
        }
        tempResized = true;
        return true;
    };
    if (!ensureFloatTemp(mTempQ, qBase) ||
        !ensureFloatTemp(mTempQK, qkBase) ||
        !ensureFloatTemp(mTempSoftMax, qkBase) ||
        !ensureMaskTemp(maskBase)) {
        MNN_ERROR("PackedAttentionBufExecution failed to grow temporary buffers\n");
        return OUT_OF_MEMORY;
    }
    if (tempResized) {
        acquireReleasePackedAttentionTemps(mOpenCLBackend, mTempQ.get(), mTempQK.get(),
                                           mTempSoftMax.get(), mTempMask.get());
    }

    auto ensureReqMeta = [&]() -> bool {
        if (reqIds.size() <= mReqMetaCapacity) {
            return true;
        }
        mReqMetaCapacity = reqIds.size();
        size_t bytes = mReqMetaCapacity * 4 * sizeof(int);
        mReqMeta0.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr));
        mReqMeta1.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr));
        mReqMeta2.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr));
        return mReqMeta0 != nullptr && mReqMeta1 != nullptr && mReqMeta2 != nullptr;
    };
    if (!ensureReqMeta()) {
        MNN_ERROR("PackedAttentionBufExecution failed to grow request metadata buffers\n");
        return OUT_OF_MEMORY;
    }

    cl_int err = CL_SUCCESS;
    err = queue.enqueueWriteBuffer(*mReqMeta0, CL_TRUE, 0, meta0.size() * sizeof(int), meta0.data());
    err |= queue.enqueueWriteBuffer(*mReqMeta1, CL_TRUE, 0, meta1.size() * sizeof(int), meta1.data());
    err |= queue.enqueueWriteBuffer(*mReqMeta2, CL_TRUE, 0, meta2.size() * sizeof(int), meta2.data());
    if (err != CL_SUCCESS) {
        MNN_ERROR("PackedAttentionBufExecution failed to upload request metadata: %d\n", err);
        return INVALID_VALUE;
    }

    cl::Buffer& queryBuffer = openCLBuffer(inputs[0]);
    cl::Buffer& keyBuffer = openCLBuffer(inputs[1]);
    cl::Buffer& valueBuffer = openCLBuffer(inputs[2]);
    cl::Buffer& outputBuffer = openCLBuffer(outputs[0]);
    cl::Buffer& tempQBuffer = openCLDeferBuffer(mTempQ.get());
    cl::Buffer& tempQKBuffer = openCLDeferBuffer(mTempQK.get());
    cl::Buffer& tempSoftmaxBuffer = openCLDeferBuffer(mTempSoftMax.get());
    cl::Buffer* tempMaskBuffer = mHasMask ? &openCLDeferBuffer(mTempMask.get()) : nullptr;
    cl::Buffer* rawMaskBuffer = mHasMask ? &openCLBuffer(inputs[3]) : nullptr;

    cl_int ret = CL_SUCCESS;
    if (!allDecode) {
        std::vector<uint32_t> gws = {static_cast<uint32_t>(UP_DIV(maxSeqLen, 4)),
                                     static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
                                     static_cast<uint32_t>(reqIds.size() * mNumHead)};
        uint32_t index = 0;
        ret = CL_SUCCESS;
        ret |= mKernelRearrangeQ->get().setArg(index++, gws[0]);
        ret |= mKernelRearrangeQ->get().setArg(index++, gws[1]);
        ret |= mKernelRearrangeQ->get().setArg(index++, gws[2]);
        ret |= mKernelRearrangeQ->get().setArg(index++, queryBuffer);
        ret |= mKernelRearrangeQ->get().setArg(index++, tempQBuffer);
        ret |= mKernelRearrangeQ->get().setArg(index++, *mReqMeta0);
        ret |= mKernelRearrangeQ->get().setArg(index++, *mReqMeta1);
        ret |= mKernelRearrangeQ->get().setArg(index++, mHeadDim);
        ret |= mKernelRearrangeQ->get().setArg(index++, mNumHead);
        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_q_packed");
        run3DKernelDefault(mKernelRearrangeQ, gws, {1, 1, 1}, runtime);
    }

    {
        std::vector<uint32_t> kvGws = {static_cast<uint32_t>(UP_DIV(maxSeqLen, 4)),
                                       static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
                                       static_cast<uint32_t>(reqIds.size() * mKvNumHead)};
        uint32_t index = 0;
        ret = CL_SUCCESS;
        ret |= mKernelRearrangeK->get().setArg(index++, kvGws[0]);
        ret |= mKernelRearrangeK->get().setArg(index++, kvGws[1]);
        ret |= mKernelRearrangeK->get().setArg(index++, kvGws[2]);
        ret |= mKernelRearrangeK->get().setArg(index++, keyBuffer);
        ret |= mKernelRearrangeK->get().setArg(index++, const_cast<cl::Buffer&>(*sharedKeyBuffer));
        ret |= mKernelRearrangeK->get().setArg(index++, *mReqMeta0);
        ret |= mKernelRearrangeK->get().setArg(index++, *mReqMeta1);
        ret |= mKernelRearrangeK->get().setArg(index++, mKvNumHead);
        ret |= mKernelRearrangeK->get().setArg(index++, mNumHead);
        ret |= mKernelRearrangeK->get().setArg(index++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_k_packed");
        run3DKernelDefault(mKernelRearrangeK, kvGws, {1, 1, 1}, runtime);

        std::vector<uint32_t> valueGws = {static_cast<uint32_t>(UP_DIV(mHeadDim, 4)),
                                          static_cast<uint32_t>(UP_DIV(maxSeqLen, 4)),
                                          static_cast<uint32_t>(reqIds.size() * mKvNumHead)};
        index = 0;
        ret = CL_SUCCESS;
        ret |= mKernelRearrangeV->get().setArg(index++, valueGws[0]);
        ret |= mKernelRearrangeV->get().setArg(index++, valueGws[1]);
        ret |= mKernelRearrangeV->get().setArg(index++, valueGws[2]);
        ret |= mKernelRearrangeV->get().setArg(index++, valueBuffer);
        ret |= mKernelRearrangeV->get().setArg(index++, const_cast<cl::Buffer&>(*sharedValueBuffer));
        ret |= mKernelRearrangeV->get().setArg(index++, *mReqMeta0);
        ret |= mKernelRearrangeV->get().setArg(index++, *mReqMeta1);
        ret |= mKernelRearrangeV->get().setArg(index++, mKvNumHead);
        ret |= mKernelRearrangeV->get().setArg(index++, mHeadDim);
        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_v_packed");
        run3DKernelDefault(mKernelRearrangeV, valueGws, {1, 1, 1}, runtime);
    }

    if (mHasMask && !allDecode) {
        std::vector<uint32_t> maskGws = {static_cast<uint32_t>(UP_DIV(maxSeqLen, 4)),
                                         static_cast<uint32_t>(UP_DIV(maxMaskStride, 4)),
                                         static_cast<uint32_t>(reqIds.size())};
        uint32_t index = 0;
        ret = CL_SUCCESS;
        ret |= mKernelRearrangeMask->get().setArg(index++, maskGws[0]);
        ret |= mKernelRearrangeMask->get().setArg(index++, maskGws[1]);
        ret |= mKernelRearrangeMask->get().setArg(index++, maskGws[2]);
        ret |= mKernelRearrangeMask->get().setArg(index++, *rawMaskBuffer);
        ret |= mKernelRearrangeMask->get().setArg(index++, *tempMaskBuffer);
        ret |= mKernelRearrangeMask->get().setArg(index++, *mReqMeta0);
        ret |= mKernelRearrangeMask->get().setArg(index++, *mReqMeta2);
        MNN_CHECK_CL_SUCCESS(ret, "setArg rearrange_mask_shortprefill_packed");
        run3DKernelDefault(mKernelRearrangeMask, maskGws, {1, 1, 1}, runtime);
    }

    if (allDecode) {
        {
            std::vector<uint32_t> qkGws = {static_cast<uint32_t>(UP_DIV(maxKvSeqLen, 4)),
                                           static_cast<uint32_t>(reqIds.size() * mNumHead)};
            uint32_t index = 0;
            ret = CL_SUCCESS;
            ret |= mKernelQkDecode->get().setArg(index++, qkGws[0]);
            ret |= mKernelQkDecode->get().setArg(index++, qkGws[1]);
            ret |= mKernelQkDecode->get().setArg(index++, queryBuffer);
            ret |= mKernelQkDecode->get().setArg(index++, const_cast<cl::Buffer&>(*sharedKeyBuffer));
            ret |= mKernelQkDecode->get().setArg(index++, tempQKBuffer);
            ret |= mKernelQkDecode->get().setArg(index++, *mReqMeta0);
            ret |= mKernelQkDecode->get().setArg(index++, *mReqMeta1);
            ret |= mKernelQkDecode->get().setArg(index++, *mReqMeta2);
            ret |= mKernelQkDecode->get().setArg(index++, mScale);
            ret |= mKernelQkDecode->get().setArg(index++, mNumHead);
            ret |= mKernelQkDecode->get().setArg(index++, mHeadDim);
            MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qk_decode_packed");
            runKernel2D(mKernelQkDecode, qkGws, {0, 0}, runtime);
        }

        {
            int localSize = packedSoftmaxLocalSize(mMaxWorkGroupSize);
            std::vector<uint32_t> softmaxGws = {static_cast<uint32_t>(localSize),
                                                1,
                                                static_cast<uint32_t>(reqIds.size() * mNumHead)};
            std::vector<uint32_t> softmaxLws = {static_cast<uint32_t>(localSize), 1, 1};
            uint32_t index = 0;
            ret = CL_SUCCESS;
            ret |= mKernelSoftmaxDecode->get().setArg(index++, softmaxGws[0]);
            ret |= mKernelSoftmaxDecode->get().setArg(index++, softmaxGws[1]);
            ret |= mKernelSoftmaxDecode->get().setArg(index++, softmaxGws[2]);
            ret |= mKernelSoftmaxDecode->get().setArg(index++, tempQKBuffer);
            ret |= mKernelSoftmaxDecode->get().setArg(index++, tempSoftmaxBuffer);
            ret |= mKernelSoftmaxDecode->get().setArg(index++, *mReqMeta1);
            ret |= mKernelSoftmaxDecode->get().setArg(index++, *mReqMeta2);
            ret |= mKernelSoftmaxDecode->get().setArg(index++, mNumHead);
            MNN_CHECK_CL_SUCCESS(ret, "setArg softmax_in1_buf_packed");
            run3DKernelDefault(mKernelSoftmaxDecode, softmaxGws, softmaxLws, runtime);
        }

        {
            std::vector<uint32_t> qkvGws = {static_cast<uint32_t>(std::max(UP_DIV(mHeadDim, 8), 1)),
                                            static_cast<uint32_t>(reqIds.size() * mNumHead)};
            uint32_t index = 0;
            ret = CL_SUCCESS;
            ret |= mKernelQkvDecode->get().setArg(index++, qkvGws[0]);
            ret |= mKernelQkvDecode->get().setArg(index++, qkvGws[1]);
            ret |= mKernelQkvDecode->get().setArg(index++, tempSoftmaxBuffer);
            ret |= mKernelQkvDecode->get().setArg(index++, const_cast<cl::Buffer&>(*sharedValueBuffer));
            ret |= mKernelQkvDecode->get().setArg(index++, outputBuffer);
            ret |= mKernelQkvDecode->get().setArg(index++, *mReqMeta0);
            ret |= mKernelQkvDecode->get().setArg(index++, *mReqMeta1);
            ret |= mKernelQkvDecode->get().setArg(index++, *mReqMeta2);
            ret |= mKernelQkvDecode->get().setArg(index++, mNumHead);
            ret |= mKernelQkvDecode->get().setArg(index++, mHeadDim);
            MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qkv_decode_b8_packed");
            runKernel2D(mKernelQkvDecode, qkvGws, {0, 0}, runtime);
        }
    } else {
        {
            std::vector<uint32_t> qkGws = {static_cast<uint32_t>(UP_DIV(maxSeqLen, 4)),
                                           static_cast<uint32_t>(UP_DIV(maxKvSeqLen, 4)),
                                           static_cast<uint32_t>(reqIds.size() * mNumHead)};
            uint32_t index = 0;
            ret = CL_SUCCESS;
            ret |= mKernelQk->get().setArg(index++, qkGws[0]);
            ret |= mKernelQk->get().setArg(index++, qkGws[1]);
            ret |= mKernelQk->get().setArg(index++, qkGws[2]);
            ret |= mKernelQk->get().setArg(index++, tempQBuffer);
            ret |= mKernelQk->get().setArg(index++, const_cast<cl::Buffer&>(*sharedKeyBuffer));
            if (mHasMask) {
                ret |= mKernelQk->get().setArg(index++, *tempMaskBuffer);
            }
            ret |= mKernelQk->get().setArg(index++, tempQKBuffer);
            ret |= mKernelQk->get().setArg(index++, *mReqMeta0);
            ret |= mKernelQk->get().setArg(index++, *mReqMeta1);
            ret |= mKernelQk->get().setArg(index++, *mReqMeta2);
            ret |= mKernelQk->get().setArg(index++, mScale);
            ret |= mKernelQk->get().setArg(index++, mNumHead);
            ret |= mKernelQk->get().setArg(index++, mHeadDim);
            MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qk_div_mask_prefill_packed");
            run3DKernelDefault(mKernelQk, qkGws, {1, 1, 1}, runtime);
        }

        {
            int localSize = packedSoftmaxLocalSize(mMaxWorkGroupSize);
            std::vector<uint32_t> softmaxGws = {static_cast<uint32_t>(localSize),
                                                static_cast<uint32_t>(UP_DIV(maxSeqLen, 4)),
                                                static_cast<uint32_t>(reqIds.size() * mNumHead)};
            std::vector<uint32_t> softmaxLws = {static_cast<uint32_t>(localSize), 1, 1};
            uint32_t index = 0;
            ret = CL_SUCCESS;
            ret |= mKernelSoftmax->get().setArg(index++, softmaxGws[0]);
            ret |= mKernelSoftmax->get().setArg(index++, softmaxGws[1]);
            ret |= mKernelSoftmax->get().setArg(index++, softmaxGws[2]);
            ret |= mKernelSoftmax->get().setArg(index++, tempQKBuffer);
            ret |= mKernelSoftmax->get().setArg(index++, tempSoftmaxBuffer);
            ret |= mKernelSoftmax->get().setArg(index++, *mReqMeta0);
            ret |= mKernelSoftmax->get().setArg(index++, *mReqMeta1);
            ret |= mKernelSoftmax->get().setArg(index++, *mReqMeta2);
            ret |= mKernelSoftmax->get().setArg(index++, mNumHead);
            MNN_CHECK_CL_SUCCESS(ret, "setArg softmax_v4_buf_packed");
            run3DKernelDefault(mKernelSoftmax, softmaxGws, softmaxLws, runtime);
        }

        {
            std::vector<uint32_t> qkvGws = {static_cast<uint32_t>(std::max(UP_DIV(mHeadDim, 8), 1)),
                                            static_cast<uint32_t>(UP_DIV(maxSeqLen, 4)),
                                            static_cast<uint32_t>(reqIds.size() * mNumHead)};
            uint32_t index = 0;
            ret = CL_SUCCESS;
            ret |= mKernelQkv->get().setArg(index++, qkvGws[0]);
            ret |= mKernelQkv->get().setArg(index++, qkvGws[1]);
            ret |= mKernelQkv->get().setArg(index++, qkvGws[2]);
            ret |= mKernelQkv->get().setArg(index++, tempSoftmaxBuffer);
            ret |= mKernelQkv->get().setArg(index++, const_cast<cl::Buffer&>(*sharedValueBuffer));
            ret |= mKernelQkv->get().setArg(index++, outputBuffer);
            ret |= mKernelQkv->get().setArg(index++, *mReqMeta0);
            ret |= mKernelQkv->get().setArg(index++, *mReqMeta1);
            ret |= mKernelQkv->get().setArg(index++, *mReqMeta2);
            ret |= mKernelQkv->get().setArg(index++, mNumHead);
            ret |= mKernelQkv->get().setArg(index++, mKvNumHead);
            ret |= mKernelQkv->get().setArg(index++, mHeadDim);
            MNN_CHECK_CL_SUCCESS(ret, "setArg matmul_qkv_prefill_packed");
            run3DKernelDefault(mKernelQkv, qkvGws, {1, 1, 1}, runtime);
        }
    }

    if (mNeedKvCache) {
        for (size_t i = 0; i < reqIds.size(); ++i) {
            mBatchKVCacheManager->addKvLength(reqIds[i], seqLens[i]);
        }
    }
    return NO_ERROR;
}

bool PackedAttentionBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (dst == nullptr) {
        return true;
    }
    *dst = new PackedAttentionBufExecution(mBatchKVCacheManager, op, bn);
    return true;
}

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
