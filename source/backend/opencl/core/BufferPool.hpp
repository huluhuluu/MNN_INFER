//
//  BufferPool.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef BufferPool_hpp
#define BufferPool_hpp

#include <set>
#include <map>
#include <memory>
#include <vector>
#include "core/NonCopyable.hpp"
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"

namespace MNN {
namespace OpenCL {
struct OpenCLBufferNode{
    OpenCLBufferNode(){};
    size_t size;
    std::shared_ptr<cl::Buffer> buffer;
};

// Transient (DYNAMIC / DYNAMIC_IN_EXECUTION) OpenCL buffer accounting.
//
// On Adreno every cl::Buffer here is created with CL_MEM_ALLOC_HOST_PTR and ends up as a
// /dev/kgsl-3d0 device mapping. Pages the GPU writes and the host never maps do NOT enter
// the process VmRSS, so sampling /proc/<pid>/status cannot see these allocations at all -
// a 64 MiB attention QK buffer is invisible there. This counter is the only way to measure
// them. Weights live in the STATIC pool and are excluded, so the number reflects just the
// per-forward scratch memory.
void clTrackTransientAlloc(size_t bytes);
size_t clTransientPeakBytes();
size_t clTransientLiveBytes();
void clResetTransientPeak();

class BufferPool : public NonCopyable {
public:
    // isTransient: true for the per-forward DYNAMIC pools, false for the STATIC weight pool
    BufferPool(cl::Context& context, cl_mem_flags flags, bool isTransient = false) : mContext(context) {
        mFlag = flags;
        mIsTransient = isTransient;
    }

    cl::Buffer* alloc(size_t size, bool separate = false);
    void recycle(cl::Buffer* buffer, bool release = false);
    void clear();
    void releaseFreeList();
    size_t totalSize() { return mTotalSize; }

private:
    std::map<cl::Buffer*, std::shared_ptr<OpenCLBufferNode>> mAllBuffer;
    std::multimap<size_t, std::shared_ptr<OpenCLBufferNode>> mFreeList;

    cl::Context& mContext;
    cl_mem_flags mFlag;
    size_t mTotalSize = 0;
    bool mIsTransient = false;
};

class BufferExecutionPool : public NonCopyable {
public:
    BufferExecutionPool(cl::Context& context, cl::CommandQueue& command, cl_mem_flags flags) : mContext(context), mCommand(command) {
        mFlag = flags;
    }

    std::shared_ptr<OpenCLBufferNode> alloc(size_t size, bool separate = false);
    void recycle(std::shared_ptr<OpenCLBufferNode> node, bool release = false);
    void clear();
    void releaseFreeList();
    size_t totalSize() { return mTotalSize; }
private:
    std::set<std::shared_ptr<OpenCLBufferNode>> mAllBuffer;
    std::multimap<size_t, std::shared_ptr<OpenCLBufferNode>> mFreeList;

    cl::Context& mContext;
    cl::CommandQueue& mCommand;
    cl_mem_flags mFlag;
    size_t mTotalSize = 0;
};

} // namespace OpenCL
} // namespace MNN

#endif /* BufferPool_hpp */
