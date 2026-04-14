//
//  CPUPackedAttention.hpp
//  MNN
//
//  Created by MNN on 2024/03/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef CPUPACKEDATTENTION_HPP
#define CPUPACKEDATTENTION_HPP

#include <functional>
#include "core/Execution.hpp"
#include "core/OpCommonUtils.hpp"
#include "CPUKVCacheManager.hpp"
#include "MNN/ErrorCode.hpp"

namespace MNN {

class CPUPackedAttention : public Execution {
public:
    CPUPackedAttention(Backend *backend, bool kv_cache);
    virtual ~CPUPackedAttention();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    bool mKVCache        = true;
    int mBytes = 4;
    int mThreadNum = 1;
    int mBlockKV = 512;
    int eP, lP, hP, mPack; // float matmul packing
    int eP8, lP8, hP8;    // GemmInt8 packing
    int mNumHead, mKvNumHead, mHeadDim;
    BatchKVMeta* mBatchMeta = nullptr;

    // common
    std::shared_ptr<Tensor> mPackQ, mPackQKV, mRunningMax, mRunningSum, mTempQKBlock, mTempOut, mExpfDiffMax;
    Backend * mBackend;
    MNN::KVCacheManager::KVCacheConfig kvconfig;
    std::shared_ptr<BatchKVCacheManager> mKVCacheManagers;
    bool mUseFlashAttention = true;

    // quant Query/Key/Value
    bool mQuantKey   = false;
    bool mQuantValue = false;
    int  mBlockNum   = 1;
    MemChunk mSumQ;
    MemChunk mQueryScale, mQueryZeroPoint, mQueryQuantScale, mQueryQuantZero;
    MemChunk mQuantQuery, mAccumBuffer;

    MemChunk mQuantQK, mQKScale, mQKBias, mSumQK, mArray;
    AutoStorage<int8_t> mGemmBias, mGemmRelu;

    std::function<void(const float*, int8_t*, size_t, const float*, ssize_t, ssize_t, const float*, ssize_t)> mQuantFunc;
    decltype(CoreInt8Functions::Int8GemmKernel) mInt8GemmKernel;

    // set up KV cache manager for each batch, and calculate max request length for current batch
    void setKVCache(int& maxReqLen) {
        // remove unused cache manager for current batch
        mKVCacheManagers->remove(mBatchMeta);
        for(int id: mBatchMeta->calId){
            maxReqLen = std::max(maxReqLen, (int)mBatchMeta->mMetas[id]->add);
            if(mKVCacheManagers->getCacheManager(id) == nullptr){
                MNN::KVCacheManager::KVCacheConfig config;
                config.mKVCacheDir = kvconfig.mKVCacheDir;
                config.mPrefixCacheDir = kvconfig.mPrefixCacheDir;
                config.mExpandChunk = kvconfig.mExpandChunk;
                config.mBlockNum = kvconfig.mBlockNum;
                config.mKvAlignNum   = kvconfig.mKvAlignNum;
                config.prefixName =  "req" + std::to_string(id) + "_";
                mKVCacheManagers->addCacheManager(id, new CPUKVCacheManager(mBackend, config));
                static_cast<CPUKVCacheManager*>(mKVCacheManagers->getCacheManager(id))->setAttenQuantKeyValue(mUseFlashAttention, mQuantKey, mQuantValue);
                mKVCacheManagers->getCacheManager(id)->onResize(mKvNumHead, mHeadDim);
            }
        }
    }
};

} // namespace MNN

#endif // CPUPACKEDATTENTION_HPP

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
