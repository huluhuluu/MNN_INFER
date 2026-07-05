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

#include "backend/opencl/execution/buffer/KVCacheCLManager.hpp"
#include "backend/opencl/execution/image/CommonExecution.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
namespace OpenCL {

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
    void registerResetCallback();
    void registerReleaseCallback();

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

    // Per-request packed metadata buffers:
    // meta0: [ioOffsetTokens, seqLen, pastLen, maxLen]
    // meta1: [qBaseElems, qkBaseElems, keyBaseElems, valueBaseElems]
    // meta2: [maskInputOffsetElems, maskBaseElems, kvSeqLen, reserved]
    std::shared_ptr<cl::Buffer> mReqMeta0;
    std::shared_ptr<cl::Buffer> mReqMeta1;
    std::shared_ptr<cl::Buffer> mReqMeta2;
    size_t mReqMetaCapacity = 0;

    // Shared temp buffers sized for the whole packed batch.
    std::shared_ptr<Tensor> mTempQ;
    std::shared_ptr<Tensor> mTempQK;
    std::shared_ptr<Tensor> mTempSoftMax;
    std::shared_ptr<Tensor> mTempMask;

    // Single kernels (not per-request)
    std::shared_ptr<KernelWrap> mKernelRearrangeQ;
    std::shared_ptr<KernelWrap> mKernelRearrangeK;
    std::shared_ptr<KernelWrap> mKernelRearrangeV;
    std::shared_ptr<KernelWrap> mKernelRearrangeMask;
    std::shared_ptr<KernelWrap> mKernelQk;
    std::shared_ptr<KernelWrap> mKernelSoftmax;
    std::shared_ptr<KernelWrap> mKernelQkv;
    std::shared_ptr<KernelWrap> mKernelQkDecode;
    std::shared_ptr<KernelWrap> mKernelSoftmaxDecode;
    std::shared_ptr<KernelWrap> mKernelQkvDecode;
};

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* PackedAttentionBufExecution_hpp */
