//
//  KVCacheCLManager.cpp
//  MNN
//
//  Created by MNN on 2026/07/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "backend/opencl/execution/buffer/KVCacheCLManager.hpp"

#include <algorithm>
#include <cstring>
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

static inline void zeroBuffer(OpenCLRuntime* runtime, cl::Buffer* buffer, size_t sizeBytes) {
    if (runtime == nullptr || buffer == nullptr || sizeBytes == 0) {
        return;
    }
    auto& queue = runtime->commandQueue();
    cl_int err = CL_SUCCESS;
    auto ptr = static_cast<char*>(queue.enqueueMapBuffer(*buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeBytes, nullptr, nullptr, &err));
    if (ptr != nullptr && err == CL_SUCCESS) {
        ::memset(ptr, 0, sizeBytes);
        queue.enqueueUnmapMemObject(*buffer, ptr);
    }
}

} // namespace

KVCacheCLManager::KVCacheCLManager(Backend *backend, bool kv_cahce) : mKVCache(kv_cahce) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
}

void KVCacheCLManager::allocKVCache(const KVMeta* meta, int seqlen) {
    if (!mKVCache) {
        return;
    }
    mPastLength = meta != nullptr ? meta->previous : 0;
    if (mOpenCLBackend->getPrecision() != BackendConfig::Precision_High) {
        mByte = 2;
    }
    reallocKVCache(meta, seqlen, false);
}

bool KVCacheCLManager::reallocKVCache(const KVMeta* meta, int seqlen, bool isExecute) {
    if (!mKVCache || meta == nullptr) {
        return false;
    }
    const int reverseSize = std::max(meta->computeReverseSize(), 0);
    int kvSeqlen = static_cast<int>(meta->previous) + seqlen - static_cast<int>(meta->remove) + reverseSize;
    int start = std::max(0, mPastLength - static_cast<int>(meta->remove));

    // latest length larger than maxLen
    if (kvSeqlen > mMaxLength) {
        int copylen = std::max(0, mPastLength - static_cast<int>(meta->remove) + reverseSize);
        size_t oldSize = mKvNumHead * UP_DIV(mMaxLength, 4) * mHeadDim * 4 * mByte;
        size_t oldMaxlen = ROUND_UP(mMaxLength, 4);
        mMaxLength = kvSeqlen + mExpandChunk;
        size_t newMaxlen = ROUND_UP(mMaxLength, 4);
        size_t bufferSize = UP_DIV(mMaxLength, 4) * mKvNumHead * mHeadDim * 4 * mByte;
        auto runtime = mOpenCLBackend->getOpenCLRuntime();
        auto& queue = runtime->commandQueue();
        auto newKey = new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize);
        auto newValue = new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize);
        bool hasOldBuffer = (mPastKey != nullptr && mPastValue != nullptr && oldSize > 0);
        bool needCopy = copylen > 0 && hasOldBuffer;

        if (!needCopy) {
            cl_int newKeyRes = CL_SUCCESS;
            cl_int newValueRes = CL_SUCCESS;
            char* newKeyPtr = (char*)queue.enqueueMapBuffer(*newKey, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &newKeyRes);
            char* newValuePtr = (char*)queue.enqueueMapBuffer(*newValue, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &newValueRes);
            if (newKeyPtr != nullptr && newKeyRes == CL_SUCCESS) {
                ::memset(newKeyPtr, 0, bufferSize);
                queue.enqueueUnmapMemObject(*newKey, newKeyPtr);
            }
            if (newValuePtr != nullptr && newValueRes == CL_SUCCESS) {
                ::memset(newValuePtr, 0, bufferSize);
                queue.enqueueUnmapMemObject(*newValue, newValuePtr);
            }
        }

        if (needCopy) {
            // copy key
            {
                size_t oldMaxlenSize = oldMaxlen * mByte;
                size_t newMaxlenSize = newMaxlen * mByte;
                cl_int newKeyRes = CL_SUCCESS;
                cl_int keyRes = CL_SUCCESS;
                char *newKeyPtr = (char*)queue.enqueueMapBuffer(*newKey, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &newKeyRes);
                char *keyPtr = (char*)queue.enqueueMapBuffer(*mPastKey.get(), true, CL_MAP_READ, 0, oldSize, nullptr, nullptr, &keyRes);
                if (newKeyPtr != nullptr && keyPtr != nullptr && newKeyRes == CL_SUCCESS && keyRes == CL_SUCCESS) {
                    for (int i = 0; i < mKvNumHead * mHeadDim; ++i) {
                        ::memcpy(newKeyPtr + i * newMaxlenSize, keyPtr + i * oldMaxlenSize, oldMaxlenSize);
                    }
                } else {
                    MNN_ERROR("Map error while copying OpenCL KV key cache\n");
                }
                if (newKeyPtr != nullptr) {
                    queue.enqueueUnmapMemObject(*newKey, newKeyPtr);
                }
                if (keyPtr != nullptr) {
                    queue.enqueueUnmapMemObject(*mPastKey.get(), keyPtr);
                }
            }

            // copy value
            {
                cl_int newValueRes = CL_SUCCESS;
                cl_int valueRes = CL_SUCCESS;
                char *newValuePtr = (char*)queue.enqueueMapBuffer(*newValue, true, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &newValueRes);
                char *valuePtr = (char*)queue.enqueueMapBuffer(*mPastValue.get(), true, CL_MAP_READ, 0, oldSize, nullptr, nullptr, &valueRes);
                if (newValuePtr != nullptr && valuePtr != nullptr && newValueRes == CL_SUCCESS && valueRes == CL_SUCCESS) {
                    for (int i = 0; i < mKvNumHead; ++i) {
                        for (int j = 0; j < copylen; ++j) {
                            ::memcpy(newValuePtr + (i * newMaxlen + j) * mHeadDim * mByte, valuePtr + (i * oldMaxlen + j) * mHeadDim * mByte, mHeadDim * mByte);
                        }
                    }
                } else {
                    MNN_ERROR("Map error while copying OpenCL KV value cache\n");
                }
                if (newValuePtr != nullptr) {
                    queue.enqueueUnmapMemObject(*newValue, newValuePtr);
                }
                if (valuePtr != nullptr) {
                    queue.enqueueUnmapMemObject(*mPastValue.get(), valuePtr);
                }
            }
        }
        mPastKey.reset(newKey);
        mPastValue.reset(newValue);
        // resize phase don't update mPastLength value, execute phase will update it
        if (isExecute) {
            mPastLength = start;
        }
    }

    // Remove
    // resize phase don't remove kvcache, execute phase will do it
    if (isExecute) {
        if (0 == meta->n_reserve) {
            mPastLength = start;
            return true;
        }
        if (mPastKey == nullptr || mPastValue == nullptr) {
            return false;
        }

        size_t pastkvSize = mKvNumHead * UP_DIV(mMaxLength, 4) * mHeadDim * 4 * mByte;
        auto& queue = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
        cl_int keyRes = CL_SUCCESS;
        cl_int valueRes = CL_SUCCESS;
        char *keyPtr = (char*)queue.enqueueMapBuffer(*mPastKey.get(), true, CL_MAP_READ | CL_MAP_WRITE, 0, pastkvSize, nullptr, nullptr, &keyRes);
        char *valuePtr = (char*)queue.enqueueMapBuffer(*mPastValue.get(), true, CL_MAP_READ | CL_MAP_WRITE, 0, pastkvSize, nullptr, nullptr, &valueRes);
        if (keyPtr == nullptr || valuePtr == nullptr || keyRes != CL_SUCCESS || valueRes != CL_SUCCESS) {
            if (keyPtr != nullptr) {
                queue.enqueueUnmapMemObject(*mPastKey.get(), keyPtr);
            }
            if (valuePtr != nullptr) {
                queue.enqueueUnmapMemObject(*mPastValue.get(), valuePtr);
            }
            return false;
        }

        for (int n = 0; n < meta->n_reserve; ++n) {
            auto begin = meta->reserve[2 * n];
            auto length = meta->reserve[2 * n + 1];
            auto copySrcIndex = start + begin;
            auto copyDstIndex = start;
            for (int i = 0; i <  mKvNumHead * mHeadDim; i++) {
                ::memmove(keyPtr + (i * mMaxLength + copyDstIndex) * mByte, keyPtr + (i * mMaxLength + copySrcIndex) * mByte, length * mByte);
            }
            for (int i = 0; i <  mKvNumHead; i++) {
                for (int j = 0; j < length; j++) {
                    ::memmove(valuePtr + (i * mMaxLength + copyDstIndex + j) * mHeadDim * mByte, valuePtr + (i * mMaxLength + copySrcIndex + j) * mHeadDim * mByte, mHeadDim * mByte);
                }
            }
            start += length;
        }
        queue.enqueueUnmapMemObject(*mPastKey.get(), keyPtr);
        queue.enqueueUnmapMemObject(*mPastValue.get(), valuePtr);
        mPastLength = start;
    }
    return true;
}

BatchKVCacheCLManager::BatchKVCacheCLManager(Backend *backend, bool kv_cache)
    : mBackend(backend), mOpenCLBackend(static_cast<OpenCLBackend*>(backend)), mKVCache(kv_cache) {
    mByte = mOpenCLBackend->getPrecision() != BackendConfig::Precision_High ? 2 : 4;
}

BatchKVCacheCLManager::~BatchKVCacheCLManager() {
    onClear();
}

int BatchKVCacheCLManager::alignedLength(int length) const {
    length = std::max(length, 1);
    int chunkAligned = ((length + mExpandChunk - 1) / mExpandChunk) * mExpandChunk;
    return ROUND_UP(chunkAligned, 4);
}

void BatchKVCacheCLManager::allocKVCache(const BatchKVMeta* batchMeta) {
    if (!mKVCache || batchMeta == nullptr) {
        return;
    }
    for (const auto& item : batchMeta->mMetas) {
        const int reqId = item.first;
        const KVMeta* meta = item.second;
        auto& info = mRequestInfos[reqId];
        if (info.pastLength == 0 && meta != nullptr) {
            info.pastLength = static_cast<int>(meta->previous);
        }
        int required = info.pastLength;
        if (meta != nullptr) {
            required = std::max(required, retainedLength(meta, info.pastLength) + static_cast<int>(meta->add));
        }
        info.maxLength = std::max(info.maxLength, alignedLength(required));
    }
}

bool BatchKVCacheCLManager::copyRequestData(const cl::Buffer* oldKeyBuffer,
                                            const cl::Buffer* oldValueBuffer,
                                            const RequestCacheInfo* oldInfo,
                                            cl::Buffer* newKeyBuffer,
                                            cl::Buffer* newValueBuffer,
                                            const RequestCacheInfo& newInfo,
                                            const KVMeta* meta) {
    if (oldKeyBuffer == nullptr || oldValueBuffer == nullptr || oldInfo == nullptr ||
        newKeyBuffer == nullptr || newValueBuffer == nullptr || oldInfo->pastLength <= 0) {
        return true;
    }

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto& queue = runtime->commandQueue();

    const size_t keyOldSize = (size_t)oldInfo->maxLength * mKvNumHead * mHeadDim * mByte;
    const size_t valueOldSize = keyOldSize;
    const size_t keyNewSize = (size_t)newInfo.maxLength * mKvNumHead * mHeadDim * mByte;
    const size_t valueNewSize = keyNewSize;

    cl_int oldKeyErr = CL_SUCCESS;
    cl_int oldValueErr = CL_SUCCESS;
    cl_int newKeyErr = CL_SUCCESS;
    cl_int newValueErr = CL_SUCCESS;

    auto oldKeyPtr = static_cast<char*>(queue.enqueueMapBuffer(*oldKeyBuffer, CL_TRUE, CL_MAP_READ,
                                                               (size_t)oldInfo->keyOffset * mByte, keyOldSize,
                                                               nullptr, nullptr, &oldKeyErr));
    auto oldValuePtr = static_cast<char*>(queue.enqueueMapBuffer(*oldValueBuffer, CL_TRUE, CL_MAP_READ,
                                                                 (size_t)oldInfo->valueOffset * mByte, valueOldSize,
                                                                 nullptr, nullptr, &oldValueErr));
    auto newKeyPtr = static_cast<char*>(queue.enqueueMapBuffer(*newKeyBuffer, CL_TRUE, CL_MAP_WRITE,
                                                               (size_t)newInfo.keyOffset * mByte, keyNewSize,
                                                               nullptr, nullptr, &newKeyErr));
    auto newValuePtr = static_cast<char*>(queue.enqueueMapBuffer(*newValueBuffer, CL_TRUE, CL_MAP_WRITE,
                                                                 (size_t)newInfo.valueOffset * mByte, valueNewSize,
                                                                 nullptr, nullptr, &newValueErr));

    bool success = oldKeyPtr != nullptr && oldValuePtr != nullptr && newKeyPtr != nullptr && newValuePtr != nullptr &&
                   oldKeyErr == CL_SUCCESS && oldValueErr == CL_SUCCESS &&
                   newKeyErr == CL_SUCCESS && newValueErr == CL_SUCCESS;

    if (success) {
        int remove = meta != nullptr ? std::min<int>(meta->remove, oldInfo->pastLength) : 0;
        int prefix = std::max(oldInfo->pastLength - remove, 0);
        int dstIndex = prefix;
        const size_t oldKeyRowStride = (size_t)oldInfo->maxLength * mByte;
        const size_t newKeyRowStride = (size_t)newInfo.maxLength * mByte;
        const size_t oldValueHeadStride = (size_t)oldInfo->maxLength * mHeadDim * mByte;
        const size_t newValueHeadStride = (size_t)newInfo.maxLength * mHeadDim * mByte;

        auto copyRange = [&](int srcIndex, int dstToken, int length) {
            if (length <= 0) {
                return;
            }
            for (int i = 0; i < mKvNumHead * mHeadDim; ++i) {
                ::memcpy(newKeyPtr + i * newKeyRowStride + (size_t)dstToken * mByte,
                         oldKeyPtr + i * oldKeyRowStride + (size_t)srcIndex * mByte,
                         (size_t)length * mByte);
            }
            for (int h = 0; h < mKvNumHead; ++h) {
                ::memcpy(newValuePtr + h * newValueHeadStride + (size_t)dstToken * mHeadDim * mByte,
                         oldValuePtr + h * oldValueHeadStride + (size_t)srcIndex * mHeadDim * mByte,
                         (size_t)length * mHeadDim * mByte);
            }
        };

        copyRange(0, 0, prefix);
        if (meta != nullptr && meta->n_reserve > 0 && meta->reserve != nullptr) {
            const int start = prefix;
            for (int i = 0; i < meta->n_reserve; ++i) {
                const int srcIndex = start + meta->reserve[2 * i];
                const int length = meta->reserve[2 * i + 1];
                copyRange(srcIndex, dstIndex, length);
                dstIndex += length;
            }
        }
    }

    if (oldKeyPtr != nullptr) {
        queue.enqueueUnmapMemObject(*oldKeyBuffer, oldKeyPtr);
    }
    if (oldValuePtr != nullptr) {
        queue.enqueueUnmapMemObject(*oldValueBuffer, oldValuePtr);
    }
    if (newKeyPtr != nullptr) {
        queue.enqueueUnmapMemObject(*newKeyBuffer, newKeyPtr);
    }
    if (newValuePtr != nullptr) {
        queue.enqueueUnmapMemObject(*newValueBuffer, newValuePtr);
    }
    return success;
}

bool BatchKVCacheCLManager::rebuildArena(const BatchKVMeta* batchMeta, bool applyPendingOps) {
    if (!mKVCache || batchMeta == nullptr) {
        return false;
    }

    std::map<int, RequestCacheInfo> newInfos;
    int totalKeyElems = 0;
    int totalValueElems = 0;
    bool layoutChanged = mPastKeyBuffer == nullptr || mPastValueBuffer == nullptr ||
                         mRequestInfos.size() != batchMeta->mMetas.size();

    for (const auto& item : batchMeta->mMetas) {
        const int reqId = item.first;
        const KVMeta* meta = item.second;
        auto oldIt = mRequestInfos.find(reqId);
        RequestCacheInfo info;
        if (oldIt != mRequestInfos.end()) {
            info = oldIt->second;
        } else if (meta != nullptr) {
            info.pastLength = static_cast<int>(meta->previous);
        }

        const int currentPast = std::max(info.pastLength, meta != nullptr ? static_cast<int>(meta->previous) : 0);
        const int nextPast = applyPendingOps ? retainedLength(meta, currentPast) : currentPast;
        const int requiredLength = nextPast + (meta != nullptr ? static_cast<int>(meta->add) : 0);
        info.maxLength = std::max(info.maxLength, alignedLength(requiredLength));
        info.pastLength = nextPast;
        info.keyOffset = totalKeyElems;
        info.valueOffset = totalValueElems;
        totalKeyElems += info.maxLength * mKvNumHead * mHeadDim;
        totalValueElems += info.maxLength * mKvNumHead * mHeadDim;
        newInfos[reqId] = info;

        if (!layoutChanged) {
            if (oldIt == mRequestInfos.end() ||
                oldIt->second.maxLength != info.maxLength ||
                oldIt->second.keyOffset != info.keyOffset ||
                oldIt->second.valueOffset != info.valueOffset ||
                oldIt->second.pastLength != info.pastLength) {
                layoutChanged = true;
            }
        }
    }

    if (!layoutChanged) {
        mRequestInfos = std::move(newInfos);
        return true;
    }

    if (newInfos.empty()) {
        onClear();
        return true;
    }

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const size_t keyBytes = (size_t)totalKeyElems * mByte;
    const size_t valueBytes = (size_t)totalValueElems * mByte;
    std::shared_ptr<cl::Buffer> newKeyBuffer(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, keyBytes, nullptr));
    std::shared_ptr<cl::Buffer> newValueBuffer(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, valueBytes, nullptr));
    zeroBuffer(runtime, newKeyBuffer.get(), keyBytes);
    zeroBuffer(runtime, newValueBuffer.get(), valueBytes);

    for (const auto& item : newInfos) {
        const int reqId = item.first;
        const KVMeta* meta = batchMeta->mMetas.at(reqId);
        auto oldIt = mRequestInfos.find(reqId);
        const RequestCacheInfo* oldInfo = oldIt == mRequestInfos.end() ? nullptr : &oldIt->second;
        if (!copyRequestData(mPastKeyBuffer.get(), mPastValueBuffer.get(), oldInfo,
                             newKeyBuffer.get(), newValueBuffer.get(), item.second, meta)) {
            return false;
        }
    }

    mPastKeyBuffer = newKeyBuffer;
    mPastValueBuffer = newValueBuffer;
    mRequestInfos = std::move(newInfos);
    return true;
}

bool BatchKVCacheCLManager::reallocKVCache(const BatchKVMeta* batchMeta, bool isExecute) {
    return rebuildArena(batchMeta, isExecute);
}

bool BatchKVCacheCLManager::remove(const BatchKVMeta* batchMeta) {
    (void)batchMeta;
    return true;
}

bool BatchKVCacheCLManager::ensureForExecute(const BatchKVMeta* batchMeta) {
    return rebuildArena(batchMeta, true);
}

bool BatchKVCacheCLManager::release(int reqId) {
    return mRequestInfos.erase(reqId) > 0;
}

void BatchKVCacheCLManager::onClear() {
    mPastKeyBuffer.reset();
    mPastValueBuffer.reset();
    mRequestInfos.clear();
}

int BatchKVCacheCLManager::pastKvLength(int reqId) const {
    auto it = mRequestInfos.find(reqId);
    return it == mRequestInfos.end() ? 0 : it->second.pastLength;
}

void BatchKVCacheCLManager::addKvLength(int reqId, int seqLen) {
    auto it = mRequestInfos.find(reqId);
    if (it != mRequestInfos.end()) {
        it->second.pastLength += seqLen;
    }
}

int BatchKVCacheCLManager::maxLength(int reqId) const {
    auto it = mRequestInfos.find(reqId);
    return it == mRequestInfos.end() ? 0 : it->second.maxLength;
}

int BatchKVCacheCLManager::keyOffset(int reqId) const {
    auto it = mRequestInfos.find(reqId);
    return it == mRequestInfos.end() ? 0 : it->second.keyOffset;
}

int BatchKVCacheCLManager::valueOffset(int reqId) const {
    auto it = mRequestInfos.find(reqId);
    return it == mRequestInfos.end() ? 0 : it->second.valueOffset;
}

const cl::Buffer* BatchKVCacheCLManager::key() const {
    return mPastKeyBuffer.get();
}

const cl::Buffer* BatchKVCacheCLManager::value() const {
    return mPastValueBuffer.get();
}

const cl::Buffer* BatchKVCacheCLManager::key(int reqId) const {
    return mRequestInfos.find(reqId) == mRequestInfos.end() ? nullptr : mPastKeyBuffer.get();
}

const cl::Buffer* BatchKVCacheCLManager::value(int reqId) const {
    return mRequestInfos.find(reqId) == mRequestInfos.end() ? nullptr : mPastValueBuffer.get();
}

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
