//
//  PackedAttentionBufExecution.cpp
//  MNN
//
//  Created by MNN on 2025/03/20.
//  Copyright © 2018, Alibaba Group Holding Limited.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "PackedAttentionBufExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

// ============================================================================
// BatchKVCacheCLManager Implementation
// ============================================================================

BatchKVCacheCLManager::BatchKVCacheCLManager(Backend *backend, bool kv_cache)
    : mBackend(backend), mKVCache(kv_cache) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    mByte = mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16() ? 2 : 4;
}

void BatchKVCacheCLManager::allocKVCache(const BatchKVMeta* batchMeta) {
    if (!mKVCache || batchMeta == nullptr) return;
    
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    
    for (int reqId : batchMeta->calId) {
        auto it = batchMeta->mMetas.find(reqId);
        const KVMeta* meta = (it != batchMeta->mMetas.end()) ? it->second : nullptr;
        int seqlen = (meta != nullptr) ? meta->add : 1;
        
        // Initialize per-request KV cache if not exists
        if (mPastKeys.find(reqId) == mPastKeys.end()) {
            int maxSize = std::max(seqlen, mExpandChunk);
            int allocSize = ((maxSize + mExpandChunk - 1) / mExpandChunk) * mExpandChunk;
            
            std::shared_ptr<cl::Buffer> keyBuffer(new cl::Buffer(
                runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                allocSize * mByte * mKvNumHead * mHeadDim, nullptr));
            std::shared_ptr<cl::Buffer> valueBuffer(new cl::Buffer(
                runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                allocSize * mByte * mKvNumHead * mHeadDim, nullptr));
            
            mPastKeys[reqId] = keyBuffer;
            mPastValues[reqId] = valueBuffer;
            mPastLengths[reqId] = 0;
            mMaxLengths[reqId] = allocSize;
        }
        
        // Set past length from meta
        if (meta != nullptr) {
            mPastLengths[reqId] = meta->previous;
        }
    }
}

bool BatchKVCacheCLManager::reallocKVCacheForReq(int reqId, const KVMeta* meta, bool isExecute) {
    if (!mKVCache) return false;
    if (mPastKeys.find(reqId) == mPastKeys.end()) return false;
    
    int seqlen = (meta != nullptr) ? meta->add : 0;
    
    int pastLen = mPastLengths[reqId];
    int newLen = pastLen + seqlen;
    int maxLen = mMaxLengths[reqId];
    
    // Need to expand?
    if (newLen > maxLen) {
        int newMaxLen = ((newLen + mExpandChunk - 1) / mExpandChunk) * mExpandChunk;
        
        auto runtime = mOpenCLBackend->getOpenCLRuntime();
        
        std::shared_ptr<cl::Buffer> newKeyBuffer(new cl::Buffer(
            runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            newMaxLen * mByte * mKvNumHead * mHeadDim, nullptr));
        std::shared_ptr<cl::Buffer> newValueBuffer(new cl::Buffer(
            runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            newMaxLen * mByte * mKvNumHead * mHeadDim, nullptr));
        
        if (isExecute && pastLen > 0) {
            // Copy old data to new buffer
            auto queue = runtime->commandQueue();
            queue.enqueueCopyBuffer(*mPastKeys[reqId], *newKeyBuffer, 0, 0, pastLen * mByte * mKvNumHead * mHeadDim);
            queue.enqueueCopyBuffer(*mPastValues[reqId], *newValueBuffer, 0, 0, pastLen * mByte * mKvNumHead * mHeadDim);
        }
        
        mPastKeys[reqId] = newKeyBuffer;
        mPastValues[reqId] = newValueBuffer;
        mMaxLengths[reqId] = newMaxLen;
    }
    
    return true;
}

bool BatchKVCacheCLManager::reallocKVCache(const BatchKVMeta* batchMeta, bool isExecute) {
    if (!mKVCache || batchMeta == nullptr) return false;
    
    for (int reqId : batchMeta->calId) {
        auto it = batchMeta->mMetas.find(reqId);
        const KVMeta* meta = (it != batchMeta->mMetas.end()) ? it->second : nullptr;
        reallocKVCacheForReq(reqId, meta, isExecute);
    }
    return true;
}

bool BatchKVCacheCLManager::remove(const BatchKVMeta* batchMeta) {
    if (!mKVCache || batchMeta == nullptr) return false;
    
    for (int reqId : batchMeta->calId) {
        auto it = batchMeta->mMetas.find(reqId);
        if (it != batchMeta->mMetas.end() && it->second->remove > 0) {
            mPastLengths[reqId] -= it->second->remove;
        }
    }
    return true;
}

int BatchKVCacheCLManager::pastKvLength(int reqId) const {
    auto it = mPastLengths.find(reqId);
    return (it != mPastLengths.end()) ? it->second : 0;
}

void BatchKVCacheCLManager::addKvLength(int reqId, int seqLen) {
    mPastLengths[reqId] += seqLen;
}

int BatchKVCacheCLManager::maxLength(int reqId) const {
    auto it = mMaxLengths.find(reqId);
    return (it != mMaxLengths.end()) ? it->second : 0;
}

const cl::Buffer* BatchKVCacheCLManager::key(int reqId) const {
    auto it = mPastKeys.find(reqId);
    return (it != mPastKeys.end()) ? it->second.get() : nullptr;
}

const cl::Buffer* BatchKVCacheCLManager::value(int reqId) const {
    auto it = mPastValues.find(reqId);
    return (it != mPastValues.end()) ? it->second.get() : nullptr;
}

// ============================================================================
// PackedAttentionBufExecution Implementation
// ============================================================================

PackedAttentionBufExecution::PackedAttentionBufExecution(const MNN::Op *op, Backend *backend, bool kv_cache)
    : CommonExecution(backend, op) {
    mMeta = nullptr;
    mBatchKVCacheManager.reset(new BatchKVCacheCLManager(backend, kv_cache));
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("softmax_buf", "softmax_buf", {"-DSOFTMAX_LOCAL_SIZE=512"}, mOpenCLBackend->getPrecision());
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

PackedAttentionBufExecution::PackedAttentionBufExecution(std::shared_ptr<BatchKVCacheCLManager> manager,
                                                         const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op), mBatchKVCacheManager(manager) {
    mMeta = nullptr;
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    auto kernel = mOpenCLBackend->getOpenCLRuntime()->buildKernel("softmax_buf", "softmax_buf", {"-DSOFTMAX_LOCAL_SIZE=512"}, mOpenCLBackend->getPrecision());
    mMaxWorkGroupSize = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(kernel));
}

ErrorCode PackedAttentionBufExecution::init() {
    return NO_ERROR;
}

int PackedAttentionBufExecution::getLocalSize(int size, int maxGroupSize) {
    int local_size = 1;
    while (local_size * 2 <= std::min(size, (int)maxGroupSize)) {
        local_size *= 2;
    }
    return local_size;
}

ErrorCode PackedAttentionBufExecution::onResize(const std::vector<Tensor *> &inputs,
                                                 const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    
    // Get BatchKVMeta from backend meta pointer
    mMeta = (BatchKVMeta*)(mOpenCLBackend->getMetaPtr());
    
    // Get dimensions
    auto query = inputs[0];
    mTotalSeqLen = query->length(0);
    mNumHead = query->length(1);
    mHeadDim = query->length(2);
    
    // Calculate scale
    mScale = 1.0f / sqrt((float)mHeadDim);
    
    // Get KV shape
    auto key = inputs[1];
    mKvNumHead = key->length(1);
    mGroupSize = mNumHead / mKvNumHead;
    
    mBatchKVCacheManager->setArgs(mNumHead, mKvNumHead, mHeadDim);
    
    // Has mask?
    mHasMask = (inputs.size() >= 4);
    
    // Compute query offsets for packed tensor
    mQueryOffsets.clear();
    if (mMeta != nullptr) {
        int offset = 0;
        for (int reqId : mMeta->calId) {
            mQueryOffsets.push_back(offset);
            auto it = mMeta->mMetas.find(reqId);
            int seqLen = (it != mMeta->mMetas.end() && it->second != nullptr) ? it->second->add : 1;
            offset += seqLen;
        }
    } else {
        mQueryOffsets.push_back(0);
    }
    
    int numReqs = mMeta ? mMeta->calId.size() : 1;
    
    // Alloc KV cache
    if (mMeta != nullptr) {
        mBatchKVCacheManager->allocKVCache(mMeta);
    }
    
    // Create per-request kernels and work sizes
    mKernelRearrangeQVec.resize(numReqs);
    mKernelRearrangeKVec.resize(numReqs);
    mKernelRearrangeVVec.resize(numReqs);
    mKernelQkVec.resize(numReqs);
    mKernelSoftmaxVec.resize(numReqs);
    mKernelQkvVec.resize(numReqs);
    mKernelMaskVec.resize(numReqs);
    
    mGwsRearrgQVec.resize(numReqs);
    mLwsRearrgQVec.resize(numReqs);
    mGwsRearrgKVec.resize(numReqs);
    mLwsRearrgKVec.resize(numReqs);
    mGwsRearrgVVec.resize(numReqs);
    mLwsRearrgVVec.resize(numReqs);
    mGwsQkVec.resize(numReqs);
    mLwsQkVec.resize(numReqs);
    mGwsSoftMaxVec.resize(numReqs);
    mLwsSoftMaxVec.resize(numReqs);
    mGwsQkvVec.resize(numReqs);
    mLwsQkvVec.resize(numReqs);
    mGwsMaskVec.resize(numReqs);
    mLwsMaskVec.resize(numReqs);
    
    // Allocate temp buffers (size based on max request)
    int maxSeqLen = 1;
    int maxKvSeqLen = 1;
    if (mMeta != nullptr) {
        for (int reqId : mMeta->calId) {
            auto it = mMeta->mMetas.find(reqId);
            int seqLen = (it != mMeta->mMetas.end() && it->second != nullptr) ? it->second->add : 1;
            int pastKvLen = (it != mMeta->mMetas.end() && it->second != nullptr) ? it->second->previous : 0;
            int kvSeqLen = pastKvLen + seqLen;
            maxSeqLen = std::max(maxSeqLen, seqLen);
            maxKvSeqLen = std::max(maxKvSeqLen, kvSeqLen);
        }
    } else {
        maxSeqLen = mTotalSeqLen;
        maxKvSeqLen = mTotalSeqLen;
    }
    
    mTempQ.reset(Tensor::createDevice<float>({maxSeqLen, mNumHead, mHeadDim}));
    mTempK.reset(Tensor::createDevice<float>({maxKvSeqLen, mKvNumHead, mHeadDim}));
    mTempV.reset(Tensor::createDevice<float>({maxKvSeqLen, mKvNumHead, mHeadDim}));
    mTempQK.reset(Tensor::createDevice<float>({maxSeqLen, mNumHead, maxKvSeqLen}));
    mTempSoftMax.reset(Tensor::createDevice<float>({maxSeqLen, mNumHead, maxKvSeqLen}));
    
    mOpenCLBackend->onAcquireBuffer(mTempQ.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempV.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempQK.get(), Backend::DYNAMIC);
    mOpenCLBackend->onAcquireBuffer(mTempSoftMax.get(), Backend::DYNAMIC);
    
    if (mHasMask) {
        mTempMask.reset(Tensor::createDevice<float>({maxSeqLen, maxKvSeqLen}));
        mOpenCLBackend->onAcquireBuffer(mTempMask.get(), Backend::DYNAMIC);
    }
    
    // Build kernel for each request
    std::set<std::string> buildOptions;
    buildOptions.insert(" -DOPENCL_ENABLE_FP16");
    if (runtime->isSupportedFP16()) {
        buildOptions.insert(" -DUSE_FP16=1");
    }
    if (mGroupSize > 1) {
        buildOptions.insert(" -DGROUP_SIZE=" + std::to_string(mGroupSize));
    }
    
    for (int i = 0; i < numReqs; i++) {
        int reqId = mMeta->calId[i];
        auto it = mMeta->mMetas.find(reqId);
        int seqLen = (it != mMeta->mMetas.end() && it->second != nullptr) ? it->second->add : 1;
        int pastKvLen = (it != mMeta->mMetas.end() && it->second != nullptr) ? it->second->previous : 0;
        int kvSeqLen = pastKvLen + seqLen;
        
        // Rearrange Q kernel
        {
            std::string kernelName = "rearrange_q";
            mKernelRearrangeQVec[i] = runtime->buildKernel("attention_buf", kernelName, buildOptions, mOpenCLBackend->getPrecision());
            
            uint32_t maxGws = seqLen * mNumHead * mHeadDim;
            uint32_t lws = getLocalSize(mHeadDim, mMaxWorkGroupSize);
            mGwsRearrgQVec[i] = {UP_DIV(maxGws, lws), 1, 1};
            mLwsRearrgQVec[i] = {lws, 1, 1};
        }
        
        // Rearrange K kernel
        {
            std::string kernelName = "rearrange_k";
            mKernelRearrangeKVec[i] = runtime->buildKernel("attention_buf", kernelName, buildOptions, mOpenCLBackend->getPrecision());
            
            uint32_t maxGws = seqLen * mKvNumHead * mHeadDim;
            uint32_t lws = getLocalSize(mHeadDim, mMaxWorkGroupSize);
            mGwsRearrgKVec[i] = {UP_DIV(maxGws, lws), 1, 1};
            mLwsRearrgKVec[i] = {lws, 1, 1};
        }
        
        // Rearrange V kernel
        {
            std::string kernelName = "rearrange_v";
            mKernelRearrangeVVec[i] = runtime->buildKernel("attention_buf", kernelName, buildOptions, mOpenCLBackend->getPrecision());
            
            uint32_t maxGws = seqLen * mKvNumHead * mHeadDim;
            uint32_t lws = getLocalSize(mHeadDim, mMaxWorkGroupSize);
            mGwsRearrgVVec[i] = {UP_DIV(maxGws, lws), 1, 1};
            mLwsRearrgVVec[i] = {lws, 1, 1};
        }
        
        // QK kernel
        {
            std::string kernelName = "matmul_qk";
            mKernelQkVec[i] = runtime->buildKernel("attention_buf", kernelName, buildOptions, mOpenCLBackend->getPrecision());
            
            uint32_t maxGws = seqLen * mNumHead * kvSeqLen;
            uint32_t lws = getLocalSize(mHeadDim, mMaxWorkGroupSize);
            mGwsQkVec[i] = {UP_DIV(maxGws, lws), 1, 1};
            mLwsQkVec[i] = {lws, 1, 1};
        }
        
        // Softmax kernel
        {
            std::string kernelName = "softmax";
            mKernelSoftmaxVec[i] = runtime->buildKernel("attention_buf", kernelName, buildOptions, mOpenCLBackend->getPrecision());
            
            uint32_t maxGws = seqLen * mNumHead * kvSeqLen;
            uint32_t lws = getLocalSize(kvSeqLen, mMaxWorkGroupSize);
            mGwsSoftMaxVec[i] = {UP_DIV(maxGws, lws), 1, 1};
            mLwsSoftMaxVec[i] = {lws, 1, 1};
        }
        
        // QKV kernel
        {
            std::string kernelName = "matmul_qkv";
            mKernelQkvVec[i] = runtime->buildKernel("attention_buf", kernelName, buildOptions, mOpenCLBackend->getPrecision());
            
            uint32_t maxGws = seqLen * mNumHead * mHeadDim;
            uint32_t lws = getLocalSize(mHeadDim, mMaxWorkGroupSize);
            mGwsQkvVec[i] = {UP_DIV(maxGws, lws), 1, 1};
            mLwsQkvVec[i] = {lws, 1, 1};
        }
        
        // Mask kernel (if needed)
        if (mHasMask) {
            std::string kernelName = "rearrange_mask";
            mKernelMaskVec[i] = runtime->buildKernel("attention_buf", kernelName, buildOptions, mOpenCLBackend->getPrecision());
            
            uint32_t maxGws = seqLen * kvSeqLen;
            uint32_t lws = getLocalSize(kvSeqLen, mMaxWorkGroupSize);
            mGwsMaskVec[i] = {UP_DIV(maxGws, lws), 1, 1};
            mLwsMaskVec[i] = {lws, 1, 1};
        }
    }
    
    return NO_ERROR;
}

ErrorCode PackedAttentionBufExecution::onExecute(const std::vector<Tensor *> &inputs,
                                                  const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    
    // KV cache management
    if (mMeta != nullptr) {
        mBatchKVCacheManager->reallocKVCache(mMeta, true);
        mBatchKVCacheManager->remove(mMeta);
    }
    
    // Get packed query/key/value tensors
    auto queryTensor = inputs[0];
    auto keyTensor = inputs[1];
    auto valueTensor = inputs[2];
    
    cl::Buffer& queryBuffer = openCLBuffer(queryTensor);
    cl::Buffer& keyBuffer = openCLBuffer(keyTensor);
    cl::Buffer& valueBuffer = openCLBuffer(valueTensor);
    
    cl::Buffer& outputBuffer = openCLBuffer(outputs[0]);
    
    // Get request list
    const std::vector<int>& reqIds = mMeta ? mMeta->calId : std::vector<int>{0};
    
    // Loop over each request and launch kernels
    for (size_t i = 0; i < reqIds.size(); i++) {
        int reqId = reqIds[i];
        
        // Get info from BatchKVMeta directly
        auto it = mMeta->mMetas.find(reqId);
        int seqLen = (it != mMeta->mMetas.end() && it->second != nullptr) ? it->second->add : mTotalSeqLen;
        int pastKvLen = (it != mMeta->mMetas.end() && it->second != nullptr) ? it->second->previous : 0;
        int kvSeqLen = pastKvLen + seqLen;
        int queryOffset = mQueryOffsets[i];
        int maxKeyLen = mBatchKVCacheManager->maxLength(reqId);
        
        // Get per-request KV cache buffer
        const cl::Buffer* kvKeyBuffer = mBatchKVCacheManager->key(reqId);
        const cl::Buffer* kvValueBuffer = mBatchKVCacheManager->value(reqId);
        
        // Get temp buffers
        cl::Buffer& tempQBuffer = openCLBuffer(mTempQ.get());
        cl::Buffer& tempQKBuffer = openCLBuffer(mTempQK.get());
        cl::Buffer& tempSoftMaxBuffer = openCLBuffer(mTempSoftMax.get());
        
        // Set kernel args and launch for each request
        {
            // Rearrange Q
            auto kernel = mKernelRearrangeQVec[i];
            kernel->get().setArg(0, queryBuffer);
            kernel->get().setArg(1, tempQBuffer);
            kernel->get().setArg(2, queryOffset);
            kernel->get().setArg(3, seqLen);
            kernel->get().setArg(4, mNumHead);
            kernel->get().setArg(5, mHeadDim);
            run3DKernelDefault(kernel, mGwsRearrgQVec[i], mLwsRearrgQVec[i], runtime);
        }
        
        {
            // Rearrange K (copy new K to KV cache)
            auto kernel = mKernelRearrangeKVec[i];
            kernel->get().setArg(0, keyBuffer);
            kernel->get().setArg(1, const_cast<cl::Buffer&>(*kvKeyBuffer));
            kernel->get().setArg(2, queryOffset);
            kernel->get().setArg(3, pastKvLen);
            kernel->get().setArg(4, seqLen);
            kernel->get().setArg(5, mKvNumHead);
            kernel->get().setArg(6, mHeadDim);
            kernel->get().setArg(7, maxKeyLen);
            run3DKernelDefault(kernel, mGwsRearrgKVec[i], mLwsRearrgKVec[i], runtime);
        }
        
        {
            // Rearrange V (copy new V to KV cache)
            auto kernel = mKernelRearrangeVVec[i];
            kernel->get().setArg(0, valueBuffer);
            kernel->get().setArg(1, const_cast<cl::Buffer&>(*kvValueBuffer));
            kernel->get().setArg(2, queryOffset);
            kernel->get().setArg(3, pastKvLen);
            kernel->get().setArg(4, seqLen);
            kernel->get().setArg(5, mKvNumHead);
            kernel->get().setArg(6, mHeadDim);
            kernel->get().setArg(7, maxKeyLen);
            run3DKernelDefault(kernel, mGwsRearrgVVec[i], mLwsRearrgVVec[i], runtime);
        }
        
        {
            // QK matmul
            auto kernel = mKernelQkVec[i];
            kernel->get().setArg(0, tempQBuffer);
            kernel->get().setArg(1, const_cast<cl::Buffer&>(*kvKeyBuffer));
            kernel->get().setArg(2, tempQKBuffer);
            kernel->get().setArg(3, seqLen);
            kernel->get().setArg(4, kvSeqLen);
            kernel->get().setArg(5, mNumHead);
            kernel->get().setArg(6, mKvNumHead);
            kernel->get().setArg(7, mHeadDim);
            kernel->get().setArg(8, maxKeyLen);
            kernel->get().setArg(9, mScale);
            run3DKernelDefault(kernel, mGwsQkVec[i], mLwsQkVec[i], runtime);
        }
        
        {
            // Softmax
            auto kernel = mKernelSoftmaxVec[i];
            kernel->get().setArg(0, tempQKBuffer);
            kernel->get().setArg(1, tempSoftMaxBuffer);
            kernel->get().setArg(2, seqLen);
            kernel->get().setArg(3, mNumHead);
            kernel->get().setArg(4, kvSeqLen);
            run3DKernelDefault(kernel, mGwsSoftMaxVec[i], mLwsSoftMaxVec[i], runtime);
        }
        
        {
            // QKV matmul
            auto kernel = mKernelQkvVec[i];
            kernel->get().setArg(0, tempSoftMaxBuffer);
            kernel->get().setArg(1, const_cast<cl::Buffer&>(*kvValueBuffer));
            kernel->get().setArg(2, outputBuffer);
            kernel->get().setArg(3, seqLen);
            kernel->get().setArg(4, kvSeqLen);
            kernel->get().setArg(5, mNumHead);
            kernel->get().setArg(6, mKvNumHead);
            kernel->get().setArg(7, mHeadDim);
            kernel->get().setArg(8, maxKeyLen);
            kernel->get().setArg(9, queryOffset);
            run3DKernelDefault(kernel, mGwsQkvVec[i], mLwsQkvVec[i], runtime);
        }
        
        // Update KV length
        mBatchKVCacheManager->addKvLength(reqId, seqLen);
    }
    
    return NO_ERROR;
}

bool PackedAttentionBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new PackedAttentionBufExecution(mBatchKVCacheManager, op, bn);
    return true;
}

// ============================================================================
// Creator - Note: PackedAttention is created via AttentionBufCreator when packedAttentionMode > 0
// ============================================================================

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
                                                                                                