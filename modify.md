# LLM Op Profiling 实现文档

本文档记录了为 MNN-LLM 实现算子级别性能分析功能的核心修改。

---

## 1. 架构设计

### 1.1 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        LLM Layer                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    LLMOpProfiler                        │   │
│  │  - onPrefillStart/End()                                 │   │
│  │  - beforeOpWithInfo() → onMarkOpStart()                 │   │
│  │  - afterOpWithInfo() → onMarkOpEnd(opName, opType)      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Runtime 基类 (Backend.hpp)                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  virtual void onMarkOpStart() const;        // 标记开始  │   │
│  │  virtual void onMarkOpEnd(opName, opType);  // 标记结束  │   │
│  │  virtual map<string, OpProfileInfo> onGetProfileData();  │   │
│  │  virtual void onClearProfileData();                      │   │
│  │  void recordOpProfileTime(opName, opType, timeUs);       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ CPURuntime  │     │ CLRuntime   │     │ QnnRuntime  │
   │             │     │             │     │             │
   │ 默认实现：   │     │ 重写实现：   │     │ 重写实现：   │
   │ 空操作      │     │ markOpStart │     │ TODO        │
   │ (使用       │     │ markOpEnd   │     │             │
   │ recordOp... │     │             │     │             │
   │ 直接记录)   │     │             │     │             │
   └─────────────┘     └─────────────┘     └─────────────┘
```

### 1.2 标记位方案 (Marker-based Approach)

**OpenCL 后端**无法使用 CPU 回调方式同步计时，因为：
1. OpenCL kernel 是异步执行的
2. 需要通过 `cl::Event` 获取实际的 GPU 执行时间

**解决方案**：标记位方案
```
Op 开始 → onMarkOpStart() → 记录 entries 索引
   ↓
pushEvent() → 添加 kernel 条目（opName/opType 为空）
pushEvent() → 添加 kernel 条目
pushEvent() → 添加 kernel 条目
   ↓
Op 结束 → onMarkOpEnd(opName, opType) → 批量设置这些条目的 op 信息
```

**优点**：
- 不修改 `pushEvent` 函数签名
- 避免 thread_local 变量并发覆盖问题
- 架构清晰，各后端使用适合自己的方式

---

## 2. 核心代码修改

### 2.1 Runtime 基类 (Backend.hpp)

```cpp
// ========== Profile Data Structures ==========
struct OpProfileInfo {
    std::string name;       // Op name
    std::string type;       // Op type
    float timeMs = 0.0f;    // Total time in ms
    int callCount = 0;      // Number of calls
};

// ========== Runtime Class ==========
class Runtime {
public:
    // 标记位接口 - 用于 OpenCL 等异步后端
    virtual void onMarkOpStart() const {
        // Default: do nothing, CPU uses recordOpProfileTime directly
    }
    
    virtual void onMarkOpEnd(const std::string& opName, const std::string& opType) const {
        // Default: do nothing, CPU uses recordOpProfileTime directly
    }
    
    // Profile 数据获取
    virtual std::map<std::string, OpProfileInfo> onGetProfileData() const;
    virtual void onClearProfileData();
    
    // CPU 后端直接记录时间
    void recordOpProfileTime(const std::string& opName, const std::string& opType, uint64_t timeUs) const;
    
private:
    mutable std::map<std::string, uint64_t> mProfileData;        // opName -> timeUs
    mutable std::map<std::string, std::string> mProfileOpTypes;  // opName -> opType
};
```

### 2.2 OpenCL 后端 (OpenCLRuntime.hpp)

```cpp
// Profile 条目结构
struct OpProfileEntry {
    std::string opName;
    std::string opType;
    std::string kernelName;
    cl::Event event;
};

// Thread-local 标记索引
extern thread_local size_t gPendingOpStartIndex;

class OpenCLRuntime {
public:
    // pushEvent 不修改签名，只添加条目
    void pushEvent(std::pair<std::string, cl::Event> data) {
        mEvents.push_back(data);
#ifdef ENABLE_OPENCL_TIME_PROFILER
        OpProfileEntry entry;
        entry.kernelName = data.first;
        entry.event = data.second;
        // opName/opType 留空，由 markOpEnd 设置
        mOpProfileEntries.push_back(entry);
#endif
    }
    
    // 标记位接口
    void markOpStart() {
        gPendingOpStartIndex = mOpProfileEntries.size();
    }
    
    void markOpEnd(const std::string& opName, const std::string& opType) {
        for (size_t i = gPendingOpStartIndex; i < mOpProfileEntries.size(); ++i) {
            mOpProfileEntries[i].opName = opName;
            mOpProfileEntries[i].opType = opType;
        }
    }
    
private:
    std::vector<OpProfileEntry> mOpProfileEntries;
};
```

### 2.3 CLRuntime 实现 (OpenCLBackend.cpp)

```cpp
// 标记位接口
void CLRuntime::onMarkOpStart() const {
    mOpenCLRuntime->markOpStart();
}

void CLRuntime::onMarkOpEnd(const std::string& opName, const std::string& opType) const {
    mOpenCLRuntime->markOpEnd(opName, opType);
}

// Profile 数据获取 - 按 opType 聚合
std::map<std::string, OpProfileInfo> CLRuntime::onGetProfileData() const {
    std::map<std::string, OpProfileInfo> result;
#ifdef ENABLE_OPENCL_TIME_PROFILER
    const auto& entries = mOpenCLRuntime->getOpProfileEntries();
    
    // Step 1: 按 opName 分组，计算每个 op 实例时间
    // Time = min(SUBMIT) - max(END) for all kernels in this op
    std::map<std::string, std::vector<const OpProfileEntry*>> opEntriesMap;
    for (const auto& entry : entries) {
        opEntriesMap[entry.opName].push_back(&entry);
    }
    
    // Step 2: 计算每个 op 时间，按 opType 聚合
    for (const auto& pair : opEntriesMap) {
        const auto& opName = pair.first;
        const auto& opEntries = pair.second;
        
        uint64_t minSubmit = UINT64_MAX;
        uint64_t maxEnd = 0;
        std::string opType;
        
        for (const auto* entry : opEntries) {
            auto submit = entry->event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
            auto end = entry->event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            minSubmit = std::min(minSubmit, submit);
            maxEnd = std::max(maxEnd, end);
            if (opType.empty()) opType = entry->opType;
        }
        
        if (maxEnd > minSubmit) {
            float opTimeMs = (maxEnd - minSubmit) / 1000000.0f;  // ns -> ms
            auto& info = result[opType.empty() ? opName : opType];
            info.name = opType.empty() ? opName : opType;
            info.type = opType;
            info.timeMs += opTimeMs;
            info.callCount++;
        }
    }
#endif
    return result;
}
```

### 2.4 LLM Profiler 调用 (llm_profiler.cpp)

```cpp
bool LLMOpProfiler::beforeOpWithInfo(const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
    if (!mEnabled) return true;
    
    mCurrentOpName = info->name();
    mCurrentOpType = info->type();
    mOpTimer.reset();
    
#ifdef MNN_OPENCL
    // 标记 Op 开始 - 记录当前 OpenCL entries 索引
    auto executor = MNN::Express::ExecutorScope::Current();
    if (executor) {
        auto runtimeInfo = executor->getRuntime();
        auto it = runtimeInfo.first.find(MNN_FORWARD_OPENCL);
        if (it != runtimeInfo.first.end() && it->second) {
            it->second->onMarkOpStart();
        }
    }
#endif
    
    return true;
}

void LLMOpProfiler::afterOpWithInfo(const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
    if (!mEnabled) return;
    
#ifdef MNN_OPENCL
    // 标记 Op 结束 - 批量设置 opName/opType
    auto executor = MNN::Express::ExecutorScope::Current();
    if (executor) {
        auto runtimeInfo = executor->getRuntime();
        auto it = runtimeInfo.first.find(MNN_FORWARD_OPENCL);
        if (it != runtimeInfo.first.end() && it->second) {
            it->second->onMarkOpEnd(info->name(), info->type());
        }
    }
#endif
    
    float opTime = mOpTimer.durationInUs() / 1000.0f;  // us -> ms
    
    // 记录到 Runtime 的 mProfileData
    auto executor = MNN::Express::ExecutorScope::Current();
    if (executor) {
        auto runtimeInfo = executor->getRuntime();
        // 获取当前 Runtime
        std::shared_ptr<Runtime> currentRuntime;
        for (const auto& pair : runtimeInfo.first) {
            if (pair.first != MNN_FORWARD_CPU) {
                currentRuntime = pair.second;
                break;
            }
        }
        if (!currentRuntime && !runtimeInfo.first.empty()) {
            currentRuntime = runtimeInfo.first.begin()->second;
        }
        
        if (currentRuntime) {
            uint64_t timeUs = static_cast<uint64_t>(opTime * 1000.0f);
            currentRuntime->recordOpProfileTime(info->name(), info->type(), timeUs);
        }
    }
}
```

---

## 3. 编译选项

```bash
# 启用 OpenCL profiling
cmake -DMNN_OPENCL=ON -DMNN_GPU_TIME_PROFILE=ON -DMNN_BUILD_LLM=ON ..
make -j$(nproc)
```

**宏定义**：
- `ENABLE_OPENCL_TIME_PROFILER` - 由 `MNN_GPU_TIME_PROFILE` CMake 选项控制
- `MNN_OPENCL` - 启用 OpenCL 后端

---

## 4. 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `source/core/Backend.hpp` | 添加 `onMarkOpStart()`, `onMarkOpEnd()`, `OpProfileInfo` 结构 |
| `source/backend/opencl/core/runtime/OpenCLRuntime.hpp` | 添加 `OpProfileEntry`, `markOpStart()`, `markOpEnd()`, 修改 `pushEvent()` |
| `source/backend/opencl/core/runtime/OpenCLRuntime.cpp` | 定义 `thread_local gPendingOpStartIndex` |
| `source/backend/opencl/core/OpenCLBackend.hpp` | CLRuntime 添加标记位接口声明 |
| `source/backend/opencl/core/OpenCLBackend.cpp` | CLRuntime 实现标记位接口和 `onGetProfileData()` |
| `transformers/llm/engine/src/llm_profiler.cpp` | 调用 `onMarkOpStart()`, `onMarkOpEnd()` |
