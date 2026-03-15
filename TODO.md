# LLM Generate 算子耗时测试 TODO

## 需求概述

测试 `llm::generate` 整个模型的算子耗时，需要满足以下要求：
1. 特殊的 op name 不与其他 op 共同计时
2. 上层设置 prefill 和 decode 的分开计时策略
3. 支持 OpenCL/QNN 等 GPU/NPU 后端的准确时间测试

---

## 一、现有代码分析

### 1.1 已有的时间统计机制

| 位置 | 功能 | 说明 |
|------|------|------|
| `LlmContext->prefill_us / decode_us` | Prefill/Decode 阶段总耗时 | 已实现，在 `llm.hpp:84-85` |
| `AUTOTIME` 宏 | 函数级计时 | 需定义 `MNN_OPEN_TIME_TRACE` |
| `tools/cpp/Profiler.cpp` | 算子级计时统计 | 仅适用于 CPU 后端 |
| `MNN_GPU_TIME_PROFILE` | GPU 时间 Profile | 编译选项，需要特殊处理 |

### 1.2 已有的 Op 级计时实现参考

**文件**: `transformers/llm/engine/demo/llm_benchOp.cpp`

```cpp
// 关键变量
std::map<std::string, std::pair<float, int>> opTypeTimesPrefill; // type -> <total time, count>
std::map<std::string, std::pair<float, int>> opTypeTimesDecode;  // type -> <total time, count>

// 使用 Executor::setCallBack 设置回调
exe->setCallBack(std::move(beforeOp), std::move(afterOp));
```

**判断 Prefill/Decode 阶段的方法**:
```cpp
// 通过 op 首次出现判断是否为 prefill 阶段
if (opTypes.find(name) == opTypes.end()) {
    g_isCurrentOpPrefill = true;  // 首次出现 = prefill
} else {
    g_isCurrentOpPrefill = false; // 后续出现 = decode
}
```

### 1.3 OpenCL/QNN 后端时间测试问题

**问题**: 对于非 CPU 后端，普通 callback 只能测试提交时间，无法测试真实执行时间

**解决方案**: 
- **OpenCL**: 编译时添加 `-DMNN_GPU_TIME_PROFILE=ON`，启用 `ENABLE_OPENCL_TIME_PROFILER`
- **Vulkan**: 编译时添加 `-DMNN_GPU_TIME_PROFILE=ON`，启用 `ENABLE_VULKAN_TIME_PROFILE`
- **QNN**: 需要在 graph execute 后同步

---

## 二、实现计划

### TODO 2.1: 创建 LLM Op Profiler 类

**优先级**: 高

**任务描述**:
创建独立的 LLM 算子性能分析器类，封装计时逻辑

**实现要点**:
```cpp
// 文件位置: transformers/llm/engine/include/llm/llm_profiler.hpp

class LLMOpProfiler {
public:
    // 特殊 op 名称配置
    void setSpecialOps(const std::vector<std::string>& specialOps);
    
    // 阶段控制
    void setPhase(bool isPrefill);
    
    // 回调函数
    bool beforeOp(const std::vector<MNN::Tensor*>& tensors, const OperatorInfo* info);
    bool afterOp(const std::vector<MNN::Tensor*>& tensors, const OperatorInfo* info);
    
    // 结果输出
    void printStats();
    void reset();
    
private:
    std::map<std::string, OpRecord> mPrefillOps;  // prefill 阶段 op 统计
    std::map<std::string, OpRecord> mDecodeOps;   // decode 阶段 op 统计
    std::set<std::string> mSpecialOps;            // 特殊 op (单独计时)
    bool mIsPrefillPhase = true;
};
```

**文件修改**:
- [ ] 新建 `transformers/llm/engine/include/llm/llm_profiler.hpp`
- [ ] 新建 `transformers/llm/engine/src/llm_profiler.cpp`

---

### TODO 2.2: 改进 Prefill/Decode 阶段判断逻辑

**优先级**: 高

**当前问题**:
`llm_benchOp.cpp` 中通过 op 首次出现判断阶段，不够准确

**改进方案**:
1. 在 LLM 层面增加阶段标志
2. 通过 `LlmContext` 传递当前阶段信息

**实现位置**: `transformers/llm/engine/src/llm.cpp`

```cpp
// 在 generate 相关函数中设置阶段标志
// prefill 阶段
mContext->current_phase = LlmPhase::PREFILL;
// ... prefill 执行 ...

// decode 阶段
mContext->current_phase = LlmPhase::DECODE;
// ... decode 执行 ...
```

**文件修改**:
- [ ] 修改 `transformers/llm/engine/include/llm/llm.hpp` 增加 `LlmPhase` 枚举
- [ ] 修改 `transformers/llm/engine/src/llm.cpp` 设置阶段标志

---

### TODO 2.3: 特殊 Op 单独计时

**优先级**: 中

**需求**:
某些 op（如 `lm_head`）需要单独统计，不与其他 op 合并

**实现方案**:
```cpp
struct OpRecord {
    std::string name;
    std::string type;
    float totalTime = 0.0f;      // 总耗时 (ms)
    int callCount = 0;           // 调用次数
    float flops = 0.0f;          // 计算量
    bool isSpecial = false;      // 是否为特殊 op
    std::vector<float> timeHistory; // 每次调用耗时（可选）
};

// 特殊 op 配置
profiler.setSpecialOps({
    "lm_head",           // 输出层
    "embedding",         // embedding 层
    "attention_*",       // 注意力层（支持通配符）
});
```

**文件修改**:
- [ ] 在 `LLMOpProfiler` 类中实现特殊 op 逻辑

---

### TODO 2.4: OpenCL 后端时间同步

**优先级**: 高

**问题**:
OpenCL 是异步执行，callback 中的时间不是真实执行时间

**解决方案**:

**方案 A: 编译时启用 GPU Time Profile**
```bash
cmake .. -DMNN_GPU_TIME_PROFILE=ON
```

**方案 B: 在 callback 中同步 OpenCL 队列**
```cpp
// 参考 llm_benchOp.cpp 中的实现
MNN::TensorCallBackWithInfo afterOp = [...]() {
    // 同步 OpenCL
    auto executor = MNN::Express::ExecutorScope::Current();
    RuntimeInfo runtime = executor->getRuntime();
    for(auto iter: runtime.first) {
        if(iter.first == MNN_FORWARD_OPENCL) {
            // 需要 OpenCL Runtime 提供同步接口
            // auto clRuntime = static_cast<MNN::OpenCL::CLRuntime*>(iter.second.get());
            // clRuntime->finish();
        }
    }
    // 记录时间
    ...
};
```

**方案 C: 使用 OpenCL Event 时间戳** (更精确)
```cpp
// 在 OpenCL execution 中获取 event 时间
cl_event event;
clEnqueueNDRangeKernel(..., &event);
clWaitForEvents(1, &event);
cl_ulong start, end;
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, ...);
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, ...);
```

**文件修改**:
- [ ] 研究 OpenCL Backend 同步接口
- [ ] 在 `source/backend/opencl/core/runtime/OpenCLRuntime.cpp` 暴露同步方法
- [ ] 修改 profiler callback 添加同步逻辑

---

### TODO 2.5: QNN 后端时间同步

**优先级**: 中

**问题**:
QNN 执行图在 `_exitExecute` 时才真正执行

**解决方案**:
需要与 QNN 后端协调，在每次执行后同步

**文件修改**:
- [ ] 研究 `source/backend/qnn/` 的执行模型
- [ ] 实现同步机制

---

### TODO 2.6: 创建测试工具 llm_op_profiler

**优先级**: 高

**任务描述**:
基于 `llm_benchOp.cpp` 创建专门的算子级性能测试工具

**命令行参数**:
```bash
./llm_op_profiler \
    -m <model_path> \
    -p <prompt_len> \
    -n <decode_len> \
    --special-ops <op1,op2,...> \
    --backend <cpu|opencl|qnn> \
    --output <json|markdown>
```

**输出示例**:
```
=== Prefill Phase Ops Statistics ===
Op Name              Time(avg ± std)          Account
----------------------------------------------------------
Convolution          12.345 ± 0.234 ms        45.67%
MatMul               8.123 ± 0.123 ms         30.12%
lm_head (Special)    5.234 ± 0.089 ms         19.38%
...

=== Decode Phase Ops Statistics (per token) ===
Op Name              Time(avg ± std)          Account
----------------------------------------------------------
Convolution          0.234 ± 0.012 ms         42.34%
MatMul               0.156 ± 0.008 ms         28.23%
lm_head (Special)    0.089 ± 0.005 ms         16.12%
...
```

**文件修改**:
- [ ] 新建 `transformers/llm/engine/tools/llm_op_profiler.cpp`

---

### TODO 2.7: 集成到 LLM 类中

**优先级**: 中

**任务描述**:
将 profiler 集成到 LLM 类，支持运行时开启/关闭

**API 设计**:
```cpp
// 开启 profiler
llm->enableProfiler(true);

// 运行 generate
llm->generate(...);

// 获取 profiler 结果
auto profiler = llm->getProfiler();
profiler->printStats();

// 导出结果
profiler->exportJSON("result.json");
```

**文件修改**:
- [ ] 修改 `transformers/llm/engine/include/llm/llm.hpp` 添加 profiler 接口
- [ ] 修改 `transformers/llm/engine/src/llm.cpp` 实现 profiler 集成

---

## 三、编译选项

### 3.1 启用 GPU 时间 Profile

```bash
# OpenCL
cmake .. -DMNN_OPENCL=ON -DMNN_GPU_TIME_PROFILE=ON

# Vulkan  
cmake .. -DMNN_VULKAN=ON -DMNN_GPU_TIME_PROFILE=ON
```

### 3.2 启用算子形状追踪 (调试用)

```bash
cmake .. -DLLM_OP_SHAPE_TRACE=ON
```

---

## 四、测试计划

### 4.1 单元测试
- [ ] 测试 Prefill/Decode 阶段判断准确性
- [ ] 测试特殊 op 单独计时功能
- [ ] 测试时间统计精度

### 4.2 集成测试
- [ ] 测试不同后端 (CPU/OpenCL/QNN) 的时间准确性
- [ ] 测试不同模型 (Qwen/Llama/等)
- [ ] 测试不同输入长度

### 4.3 性能对比
- [ ] 与 `timeProfile.out` 工具对比 CPU 后端结果
- [ ] 与理论计算量对比验证准确性

---

## 五、关键文件清单

| 文件 | 作用 | 状态 |
|------|------|------|
| `transformers/llm/engine/include/llm/llm.hpp` | LLM 主接口 | 需修改 |
| `transformers/llm/engine/src/llm.cpp` | LLM 实现 | 需修改 |
| `transformers/llm/engine/demo/llm_benchOp.cpp` | 现有 op bench | 参考 |
| `tools/cpp/Profiler.cpp` | 通用 profiler | 参考 |
| `source/backend/opencl/core/runtime/OpenCLRuntime.cpp` | OpenCL 运行时 | 需研究 |
| `source/backend/qnn/` | QNN 后端 | 需研究 |

---

## 六、参考文档

1. MNN 文档: `docs/tools/test.md` - timeProfile.out 使用说明
2. OpenCL Time Profiler: `source/backend/opencl/CMakeLists.txt:13`
3. Vulkan Time Profile: `source/backend/vulkan/CMakeLists.txt:8`
4. 现有实现: `transformers/llm/engine/demo/llm_benchOp.cpp`

---

## 七、执行顺序建议

1. **第一步**: 研究 `llm_benchOp.cpp` 现有实现，理解 callback 机制
2. **第二步**: 实现 TODO 2.1 - 创建 LLMOpProfiler 类
3. **第三步**: 实现 TODO 2.2 - 改进阶段判断逻辑
4. **第四步**: 实现 TODO 2.4 - OpenCL 同步（优先解决 GPU 后端问题）
5. **第五步**: 实现 TODO 2.6 - 创建测试工具
6. **第六步**: 测试验证

---

*创建时间: 2026-03-03*
*更新时间: 2026-03-06*

---

## 八、后端 Profile 数据接口实现

### 8.1 已实现的后端

| 后端 | Runtime 类 | 状态 |
|------|-----------|------|
| CPU | `CPURuntime` | ✅ 已实现 `onGetProfileData()`, `onClearProfileData()` |
| OpenCL | `CLRuntime` | ✅ 已实现 `onGetProfileData()`, `onClearProfileData()` |
| QNN | `QnnRuntime` | ✅ 已实现 `onGetProfileData()`, `onClearProfileData()` |

### 8.2 待实现的后端 (TODO)

以下后端已在 `Runtime` 类中标记 TODO，需要实现 `onGetProfileData()` 和 `onClearProfileData()` 方法：

| 后端 | Runtime 类 | 文件位置 | 优先级 |
|------|-----------|----------|--------|
| TensorRT | `TRTRuntime` | `source/backend/tensorrt/backend/TRTBackend.hpp` | 中 |
| OpenGL | `GLRuntime` | `source/backend/opengl/GLBackend.hpp` | 低 |
| HIAI | `NPURuntime` | `source/backend/hiai/backend/NPUBackend.hpp` | 中 |
| Metal | `MetalRuntime` | `source/backend/metal/MetalBackend.hpp` | 中 |
| CUDA | `CUDARuntimeWrapper` | `source/backend/cuda/core/CUDABackend.hpp` | 高 |
| CoreML | `CoreMLRuntime` | `source/backend/coreml/backend/CoreMLBackend.hpp` | 中 |
| Vulkan | `VulkanRuntime` | `source/backend/vulkan/runtime/VulkanRuntime.hpp` | 中 |
| NNAPI | `NNAPIRuntime` | `source/backend/nnapi/backend/NNAPIBackend.hpp` | 中 |

### 8.3 接口定义

在 `source/core/Backend.hpp` 的 `Runtime` 基类中：

```cpp
/**
 * @brief Get profile data for operator timing statistics
 * @return map of op name to time in milliseconds
 */
virtual std::map<std::string, float> onGetProfileData() const {
    // Default: return empty map, implemented by backends that support profiling
    return {};
}

/**
 * @brief Clear profile data after collection
 */
virtual void onClearProfileData() {
    // Default: do nothing, implemented by backends that support profiling
}
```

### 8.4 实现示例 (以 CPU 为例)

```cpp
// CPUBackend.hpp - CPURuntime 类中添加
std::map<std::string, float> onGetProfileData() const override;
void onClearProfileData() override;
void recordOpTime(const std::string& opName, float timeMs) const;

// 数据存储
mutable std::map<std::string, float> mProfileData;

// CPUBackend.cpp - 实现
std::map<std::string, float> CPURuntime::onGetProfileData() const {
    return mProfileData;
}

void CPURuntime::onClearProfileData() {
    mProfileData.clear();
}

void CPURuntime::recordOpTime(const std::string& opName, float timeMs) const {
    mProfileData[opName] += timeMs;
}
```

- 其它的解码实现 现在支持argenerate
- ?多次解码平均值
- 添加编译的profile宏