//
//  BufferPool.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/BufferPool.hpp"
namespace MNN {
namespace OpenCL {

// See the comment in BufferPool.hpp: OpenCL buffers are kgsl device mappings and are
// invisible to VmRSS, so transient allocation has to be counted explicitly.
static size_t gTransientLive = 0;
static size_t gTransientPeak = 0;
void clTrackTransientAlloc(size_t bytes) {
    gTransientLive += bytes;
    if (gTransientLive > gTransientPeak) {
        gTransientPeak = gTransientLive;
    }
}
void clTrackTransientFree(size_t bytes) {
    // Pool accounting follows the buffers that are currently owned by the pool. A
    // release can race neither allocation nor recycle because pool mutation is
    // serialized by the backend, so avoid an unnecessary lock on this hot path.
    if (bytes >= gTransientLive) {
        gTransientLive = 0;
    } else {
        gTransientLive -= bytes;
    }
}
size_t clTransientPeakBytes() { return gTransientPeak; }
size_t clTransientLiveBytes() { return gTransientLive; }
void clResetTransientPeak() { gTransientPeak = gTransientLive; }
cl::Buffer* BufferPool::alloc(size_t size, bool separate) {
    if (!separate) {
        auto iter = mFreeList.lower_bound(size);
        if (iter != mFreeList.end()) {
            auto buffer = iter->second->buffer.get();
            mFreeList.erase(iter);
            return buffer;
        }
    }
    std::shared_ptr<OpenCLBufferNode> node(new OpenCLBufferNode);
    cl_int ret = CL_SUCCESS;
    mTotalSize += size;
    if (mIsTransient) {
        clTrackTransientAlloc(size);
    }
    node->size = size;
    node->buffer.reset(new cl::Buffer(mContext, mFlag, size, NULL, &ret));
    if (nullptr == node->buffer.get() || ret != CL_SUCCESS) {
        MNN_ERROR("Alloc Buffer %lu error, code:%d \n", size, ret);
        return nullptr;
    }
    mAllBuffer.insert(std::make_pair(node->buffer.get(), node));
    return node->buffer.get();
}

void BufferPool::recycle(cl::Buffer* buffer, bool release) {
    auto iter = mAllBuffer.find(buffer);
    if (iter == mAllBuffer.end()) {
        MNN_ERROR("Error for recycle buffer\n");
        return;
    }
    if (release) {
        if (mIsTransient) {
            clTrackTransientFree(iter->second->size);
        }
        mTotalSize -= iter->second->size;
        mAllBuffer.erase(iter);
        return;
    }
    mFreeList.insert(std::make_pair(iter->second->size, iter->second));
}

void BufferPool::clear() {
    if (mIsTransient) {
        clTrackTransientFree(mTotalSize);
    }
    mFreeList.clear();
    mAllBuffer.clear();
    mTotalSize = 0;
}

void BufferPool::releaseFreeList() {
    size_t released = 0;
    for(auto mf : mFreeList){
        auto iter = mAllBuffer.find(mf.second->buffer.get());
        if (iter != mAllBuffer.end()) {
            released += iter->second->size;
            mAllBuffer.erase(iter);
        }
    }
    mFreeList.clear();
    if (released != 0) {
        mTotalSize -= released;
        if (mIsTransient) {
            clTrackTransientFree(released);
        }
    }
}

std::shared_ptr<OpenCLBufferNode> BufferExecutionPool::alloc(size_t size, bool separate) {
    if (!separate) {
        auto iter = mFreeList.lower_bound(size);
        if (iter != mFreeList.end()) {
            auto node = iter->second;
            mFreeList.erase(iter);
            return node;
        } else if(mFreeList.size() != 0){
            cl_int ret = CL_SUCCESS;
            // Synchronize to prevent old buffer references
            mCommand.finish();
            auto maxIter = mFreeList.rbegin();
            auto node = maxIter->second;
            const size_t oldSize = node.get()->size;
            if (size > oldSize) {
                const size_t delta = size - oldSize;
                mTotalSize += delta;
                clTrackTransientAlloc(delta);
            } else if (size < oldSize) {
                const size_t delta = oldSize - size;
                mTotalSize -= delta;
                clTrackTransientFree(delta);
            }
            node.get()->size = size;
            node.get()->buffer.reset(new cl::Buffer(mContext, mFlag, size, NULL, &ret));
            if (nullptr == node.get()->buffer.get() || ret != CL_SUCCESS) {
                MNN_ERROR("Alloc Buffer %lu error, code:%d \n", size, ret);
                return nullptr;
            }
            mFreeList.erase(std::prev(mFreeList.end()));
            return node;
        }
    }
    std::shared_ptr<OpenCLBufferNode> node(new OpenCLBufferNode);
    cl_int ret = CL_SUCCESS;
    mTotalSize += size;
    clTrackTransientAlloc(size);
    node->size = size;
    node->buffer.reset(new cl::Buffer(mContext, mFlag, size, NULL, &ret));
    if (nullptr == node->buffer.get() || ret != CL_SUCCESS) {
        MNN_ERROR("Alloc Buffer %lu error, code:%d \n", size, ret);
        return nullptr;
    }
    mAllBuffer.insert(node);
    return node;
}

void BufferExecutionPool::recycle(std::shared_ptr<OpenCLBufferNode> node, bool release) {
    auto iter = mAllBuffer.find(node);
    if (iter == mAllBuffer.end()) {
        MNN_ERROR("Error for recycle buffer\n");
        return;
    }
    if (release) {
        clTrackTransientFree(node->size);
        mTotalSize -= node->size;
        mAllBuffer.erase(node);
        return;
    }
    mFreeList.insert(std::make_pair(node.get()->size, node));
}

void BufferExecutionPool::clear() {
    clTrackTransientFree(mTotalSize);
    mFreeList.clear();
    mAllBuffer.clear();
    mTotalSize = 0;
}

void BufferExecutionPool::releaseFreeList() {
    size_t released = 0;
    for(auto mf : mFreeList){
        auto iter = mAllBuffer.find(mf.second);
        if (iter != mAllBuffer.end()) {
            released += mf.second->size;
            mAllBuffer.erase(iter);
        }
    }
    mFreeList.clear();
    if (released != 0) {
        mTotalSize -= released;
        clTrackTransientFree(released);
    }
}
} // namespace OpenCL
} // namespace MNN
