//
//  PackedAttentionBufExecution.hpp
//  MNN
//
//  Created by MNN on 2025/03/20.
//  Copyright © 2018, Alibaba Group Holding Limited.
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef PackedAttentionBufExecution_hpp
#define PackedAttentionBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
namespace OpenCL {

// Batch KV Cache Manager - manages per-request KV caches
class BatchKVCacheCLManager {
public:
    BatchKVCacheCLManager(Backend *backend, bool kv_cache);
    ~BatchKVCacheCLManager() = default;
    
    // Batch operations
    void allocKVCache(const BatchKVMeta* batchMeta);
    bool reallocKVCache(const BatchKVMeta* batchMeta, bool isExecute = true);
    bool remove(const BatchKVMeta* batchMeta);
    
    void setArgs(int numHead, int kvNumHead, int headDim) {
        mNumHead = numHead;
        mKvNumHead = kvNumHead;
        mHeadDim = headDim;
    }
    
    // Per-request accessors
    int pastKvLength(int reqId) const;
    void addKvLength(int reqId, int seqLen);
    int maxLength(int reqId) const;
    int numHead() const { return mNumHead; }
    int kvNumHead() const { return mKvNumHead; }
    int headDim() const { return mHeadDim; }
    
    const cl::Buffer* key(int reqId) const;
    const cl::Buffer* value(int reqId) const;
    
private:
    bool reallocKVCacheForReq(int reqId, const KVMeta* meta, bool isExecute);
    
    Backend* mBackend;
    OpenCLBackend* mOpenCLBackend;
    bool mKVCache;
    int mByte = 4;
    const int mExpandChunk = 64;
    int mNumHead = 0;
    int mKvNumHead = 0;
    int mHeadDim = 0;
    
    // Per-request KV cache
    std::map<int, std::shared_ptr<cl::Buffer>> mPastKeys;
    std::map<int, std::shared_ptr<cl::Buffer>> mPastValues;
    std::map<int, int> mPastLengths;
    std::map<int, int> mMaxLengths;
};

class PackedAttentionBufExecution : public CommonExecution {
public:
    PackedAttentionBufExecution(const MNN::Op *op, Backend *backend, bool kv_cache);
    PackedAttentionBufExecution(std::shared_ptr<BatchKVCacheCLManager> manager, const MNN::Op *op, Backend *backend);
    
    virtual ~PackedAttentionBufExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    ErrorCode init();
    int getLocalSize(int size, int maxGroupSize);
    
private:
    // Meta and backend
    BatchKVMeta* mMeta = nullptr;
    OpenCLBackend* mOpenCLBackend;
    std::shared_ptr<BatchKVCacheCLManager> mBatchKVCacheManager;
    bool mNeedKvCache = true;
    uint32_t mMaxWorkGroupSize;
    
    // Dimensions
    int mBatch = 1;
    int mTotalSeqLen = 0;
    int mNumHead = 0;
    int mKvNumHead = 0;
    int mHeadDim = 0;
    int mGroupSize = 1;
    float mScale = 1.0f;
    int mBytes = 4;
    bool mHasMask = false;
    bool mIsAddMask = false;
    
    // Computed offsets for packed tensor (other info from mBatchMeta directly)
    std::vector<int> mQueryOffsets;     // Query offset in packed tensor (cumulative)
    
    // Temp buffers (shared, sized for max seqLen)
    std::shared_ptr<Tensor> mTempQ;
    std::shared_ptr<Tensor> mTempK;
    std::shared_ptr<Tensor> mTempV;
    std::shared_ptr<Tensor> mTempQK;
    std::shared_ptr<Tensor> mTempSoftMax;
    std::shared_ptr<Tensor> mTempMask;
    
    // Per-request kernels (stored in vectors, matching AttentionBufExecution naming)
    std::vector<std::shared_ptr<KernelWrap>> mKernelRearrangeQVec;
    std::vector<std::shared_ptr<KernelWrap>> mKernelRearrangeKVec;
    std::vector<std::shared_ptr<KernelWrap>> mKernelRearrangeVVec;
    std::vector<std::shared_ptr<KernelWrap>> mKernelQkVec;
    std::vector<std::shared_ptr<KernelWrap>> mKernelSoftmaxVec;
    std::vector<std::shared_ptr<KernelWrap>> mKernelQkvVec;
    std::vector<std::shared_ptr<KernelWrap>> mKernelMaskVec;
    
    // Per-request work sizes (stored in vectors, matching AttentionBufExecution naming)
    std::vector<std::vector<uint32_t>> mGwsRearrgQVec;
    std::vector<std::vector<uint32_t>> mLwsRearrgQVec;
    std::vector<std::vector<uint32_t>> mGwsRearrgKVec;
    std::vector<std::vector<uint32_t>> mLwsRearrgKVec;
    std::vector<std::vector<uint32_t>> mGwsRearrgVVec;
    std::vector<std::vector<uint32_t>> mLwsRearrgVVec;
    std::vector<std::vector<uint32_t>> mGwsQkVec;
    std::vector<std::vector<uint32_t>> mLwsQkVec;
    std::vector<std::vector<uint32_t>> mGwsSoftMaxVec;
    std::vector<std::vector<uint32_t>> mLwsSoftMaxVec;
    std::vector<std::vector<uint32_t>> mGwsQkvVec;
    std::vector<std::vector<uint32_t>> mLwsQkvVec;
    std::vector<std::vector<uint32_t>> mGwsMaskVec;
    std::vector<std::vector<uint32_t>> mLwsMaskVec;
};

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* PackedAttentionBufExecution_hpp */