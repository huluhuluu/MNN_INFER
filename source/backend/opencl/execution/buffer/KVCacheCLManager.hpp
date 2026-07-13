//
//  KVCacheCLManager.hpp
//  MNN
//
//  Created by MNN on 2026/07/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#ifndef KVCacheCLManager_hpp
#define KVCacheCLManager_hpp

#include "backend/opencl/core/OpenCLBackend.hpp"
#include "core/OpCommonUtils.hpp"

#include <map>
#include <memory>

namespace MNN {
namespace OpenCL {

class KVCacheCLManager {
public:
    KVCacheCLManager(Backend *backend, bool kv_cache);

    ~KVCacheCLManager() = default;
    void allocKVCache(const KVMeta* meta, int seqlen);
    bool reallocKVCache(const KVMeta* meta, int seqlen, bool isExecute = true);
    void setArgs(int numHead, int kvNumHead, int headDim) {
        mNumHead = numHead;
        mKvNumHead = kvNumHead;
        mHeadDim = headDim;
    }
    int pastKvLength() {
        return mPastLength;
    }
    void addKvLength(int seq_len) {
        mPastLength += seq_len;
    }
    int maxLength() {
        return mMaxLength;
    }
    int numHead() {
        return mNumHead;
    }
    const cl::Buffer* key() {
        return mPastKey.get();
    }
    const cl::Buffer* value() {
        return mPastValue.get();
    }
    bool needKVCache() const {
        return mKVCache;
    }

private:
    bool mKVCache;
    const int mExpandChunk = 64;
    std::shared_ptr<cl::Buffer> mPastKey, mPastValue;
    int mPastLength = 0, mMaxLength = 0, mNumHead = 0, mKvNumHead = 0, mHeadDim = 0;
    OpenCLBackend *mOpenCLBackend;
    int mByte = 4;
};

// Batch KV Cache Manager - manages per-request KV caches
class BatchKVCacheCLManager {
public:
    BatchKVCacheCLManager(Backend *backend, bool kv_cache);
    ~BatchKVCacheCLManager();

    // Batch operations
    bool reallocKVCache(const BatchKVMeta* batchMeta, bool isExecute = true);
    bool remove(const BatchKVMeta* batchMeta);
    bool ensureForExecute(const BatchKVMeta* batchMeta);
    bool release(int reqId);
    void onClear();

    void setArgs(int numHead, int kvNumHead, int headDim) {
        mNumHead = numHead;
        mKvNumHead = kvNumHead;
        mHeadDim = headDim;
    }

    // Per-request accessors
    int pastKvLength(int reqId) const;
    void addKvLength(int reqId, int seqLen);
    int maxLength(int reqId) const;
    int keyOffset(int reqId) const;
    int valueOffset(int reqId) const;
    int numHead() const { return mNumHead; }
    int kvNumHead() const { return mKvNumHead; }
    int headDim() const { return mHeadDim; }
    int byteSize() const { return mByte; }

    const cl::Buffer* key() const;
    const cl::Buffer* value() const;
    const cl::Buffer* key(int reqId) const;
    const cl::Buffer* value(int reqId) const;
    bool needKVCache() const {
        return mKVCache;
    }

private:
    struct RequestCacheInfo {
        int maxLength = 0;
        int pastLength = 0;
        int keyOffset = 0;
        int valueOffset = 0;
    };

    int alignedLength(int length) const;
    bool rebuildArena(const BatchKVMeta* batchMeta, bool applyPendingOps);
    bool copyRequestData(const cl::Buffer* oldKeyBuffer,
                         const cl::Buffer* oldValueBuffer,
                         const RequestCacheInfo* oldInfo,
                         cl::Buffer* newKeyBuffer,
                         cl::Buffer* newValueBuffer,
                         const RequestCacheInfo& newInfo,
                         const KVMeta* meta,
                         int currentPast,
                         bool applyPendingOps);

    Backend* mBackend;
    OpenCLBackend* mOpenCLBackend;
    bool mKVCache;
    int mByte = 4;
    const int mExpandChunk = 64;
    int mNumHead = 0;
    int mKvNumHead = 0;
    int mHeadDim = 0;

    std::shared_ptr<cl::Buffer> mPastKeyBuffer;
    std::shared_ptr<cl::Buffer> mPastValueBuffer;
    std::map<int, RequestCacheInfo> mRequestInfos;
};

} // namespace OpenCL
} // namespace MNN

#endif /* KVCacheCLManager_hpp */
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
