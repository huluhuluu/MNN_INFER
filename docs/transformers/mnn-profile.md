# MNN-LLM Profiling 修改说明

本文档记录 MNN-LLM 算子级 profiling、推测解码评估工具和 Android 后端 profile 脚本的核心修改。

## 1. 功能

### 1.1 基础算子 Profiling

```cpp
#include "llm/llm.hpp"

auto llm = Llm::createLLM("config.json");
llm->load();

// 启用 profiler
llm->enableProfiler(true);

// 执行推理
llm->response("你好，请介绍一下你自己");

// 打印统计结果
llm->printProfilerStats();
```

适用场景：

- 分析 Prefill 和 Decode 两个阶段的耗时
- 观察不同 OpType 的累计时间占比
- 对比 CPU、OpenCL、QNN 后端的执行分布
- 观察 fallback 到 CPU 的算子

### 1.2 单独统计特殊算子

```cpp
// 特殊算子不会被合并显示，会单独输出
llm->setProfilerSpecialOps({"Attention*", "*matmul_converted"});
```

适合把 `Attention`、`MatMul`、`lm_head` 等关键算子单独统计。

### 1.3 统计宏

CPU 后端不需要额外编译开关，`llm->enableProfiler(true)` 后通过同步 callback 记录 Op wall-clock 时间。

OpenCL / QNN 这类异步或整图执行后端需要打开后端时间统计。推荐使用统一开关：

```bash
cmake .. -DMNN_OP_TIME_PROFILE=ON
```

该开关会：

- 定义 C++ 宏 `MNN_OP_TIME_PROFILE`
- 自动打开 `MNN_GPU_TIME_PROFILE`
- OpenCL 编译 `ENABLE_OPENCL_TIME_PROFILER`
- QNN 编译 `ENABLE_QNN_TIME_PROFILER`

注意：

- Android runtime profiler 构建需要打开 `MNN_OP_TIME_PROFILE=ON`。
- QNN host 导出工具是另一条构建路径，不应打开 runtime profile；`profile.sh` 中显式设置 `MNN_OP_TIME_PROFILE=OFF` 和 `MNN_GPU_TIME_PROFILE=OFF`，避免旧 CMake cache 污染 convert mode。
- OpenCL 在 `MNN_OP_TIME_PROFILE` 下会屏蔽逐 kernel 的 `kernel time = ...` 明细输出，保留汇总和 LLM profiler 输出。

### 1.4 Eagle / Speculative Decoding 模式

Eagle 推测解码会分别统计：

- 目标模型 `mProfiler`
- 草稿模型 `mDraftProfiler`

使用方式：

```cpp
llm->enableProfiler(true);
llm->response("你好");  // 使用 Eagle 推测解码

// 目标模型统计
llm->printProfilerStats();

// 草稿模型统计
llm->getDraftProfiler()->printStats();
```

Eagle 生成过程中会通过 `setActiveProfiler()` 在 target / draft profiler 之间切换，并在切换前收集 OpenCL / QNN runtime 中已经累计的后端 profile 数据。

### 1.5 profiler 工具

`transformers/llm/engine/demo/profiler.cpp`，对应可执行目标 `profiler`，运行方法：

```bash
./profiler config.json \
    --prompt-lens=128 \
    --decode-lens=1,6 \
    --warmup=2 \
    --repeat=3
```

该工具用于测试固定 prompt / decode token 长度的时间统计，适合快速验证 profiler 功能和分析特定 token 长度的性能表现。

输出内容包括：

- Prefill token 级耗时
- Decode token 级耗时
- Op 级累计耗时
- Backend breakdown
- OpType 统计

### 1.6 eagle_benchmark 工具
`transformers/llm/engine/demo/eagle_benchmark.cpp`，对应可执行目标 `eagle_benchmark`，运行方法：

```bash
./eagle_benchmark config.json

./eagle_benchmark config.json \
    --warmup=2 \
    --repeat=3 \
    --max-new-tokens=5 \
    --prompt-len=64,128,256 \
    --backend=opencl \
    --precision=low \
    --thread=4
```

该工具用于测试不同输入长度/草稿树长度下，目标模型/草稿模型的prefill/decode时间。

### 1.7 spec_eval 工具

`transformers/llm/engine/demo/spec_eval.cpp`，对应可执行目标 `spec_eval`，运行方法：

```bash
./spec_eval config.json data.txt \
    --backend=opencl \
    --precision=low \
    --max-tokens=512 \
    --limit=100 \
    --output=result.json
```

支持 prompt template：

```bash
./spec_eval config.json gsm8k.jsonl \
    --template-file=prompt_templates.json \
    --template-name=gsm8k \
    --no-thinking \
    --output=gsm8k_result.json
```
该工具用于评测推测解码模型的性能和平均接受长度、理论加速比等.

### 1.8 profile.sh 全流程测试脚本

脚本：`project/android/profile.sh`可用于测试`Android`多后端、多输入/解码长度的性能表现，包含从构建、推送、执行到结果收集的全流程自动化。

默认测试参数在文件开头，根据实际情况替换：

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CPU_BUILD_DIR="${CPU_BUILD_DIR:-$SCRIPT_DIR/build_64_cpu}"
OPENCL_BUILD_DIR="${OPENCL_BUILD_DIR:-$SCRIPT_DIR/build_64_opencl}"
QNN_BUILD_DIR="${QNN_BUILD_DIR:-$SCRIPT_DIR/build_64_qnn}"
HOST_BUILD_DIR="${HOST_BUILD_DIR:-$ROOT_DIR/build_qnn_export}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/local/tmp/mnn-profiler}"
ADB_SERIAL="${ADB_SERIAL:-192.168.124.101:47954}"
MODEL_SRC_DIR="${MODEL_SRC_DIR:-/data/HF_MODELS/Qwen3-1.7B-MNN}"
QNN_EXPORT_SCRIPT="${QNN_EXPORT_SCRIPT:-$ROOT_DIR/transformers/llm/export/npu/generate_llm_qnn.py}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-}"
QNN_SOC_ID="${QNN_SOC_ID:-69}"
QNN_DSP_ARCH="${QNN_DSP_ARCH:-v75}"
QNN_HEXAGON_ARCH="${QNN_HEXAGON_ARCH:-${QNN_DSP_ARCH#v}}"
BENCH_BIN="${BENCH_BIN:-}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/llm_profile_logs}"
WORK_ROOT="${WORK_ROOT:-$SCRIPT_DIR/llm_profile_work}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

PROMPT_LENS=(128)
DECODE_LENS=(6)
BACKENDS=(qnn) # cpu opencl qnn
WARMUP=2
REPEAT=3

CPU_LOG="$LOG_DIR/cpu.log"
OPENCL_LOG="$LOG_DIR/opencl.log"
QNN_LOG="$LOG_DIR/qnn.log"
```

脚本功能：

- 自动构建 CPU / OpenCL / QNN Android runtime
- 自动推送 runtime so、`profiler` 和模型目录
- QNN 自动构建 host 导出工具并生成 `config_qnn.json`
- 每次运行前创建远端 `tmp` 目录，避免 cache 写入失败
- CPU / OpenCL / QNN 运行结束后清理本地和远端临时模型目录

Android 交叉编译统一调用 `project/android/build_64.sh`，再单独构建 `profiler` target：

```bash
cd "$build_dir"
"$SCRIPT_DIR/build_64.sh" "${flags[@]}"
cmake --build "$build_dir" --target profiler -j"$BUILD_JOBS"
```

QNN host 导出工具使用 server/native build：

```bash
cmake -S "$ROOT_DIR" -B "$HOST_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS= \
    -DCMAKE_CXX_FLAGS= \
    -DMNN_BUILD_LLM=ON \
    -DMNN_LOW_MEMORY=ON \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=true \
    -DMNN_OP_TIME_PROFILE=OFF \
    -DMNN_GPU_TIME_PROFILE=OFF \
    -DMNN_QNN=ON \
    -DMNN_QNN_CONVERT_MODE=ON \
    -DMNN_WITH_PLUGIN=ON \
    -DQNN_SDK_ROOT="$QNN_SDK_ROOT"

cmake --build "$HOST_BUILD_DIR" --target generateLlmIO -j"$BUILD_JOBS"
cmake --build "$HOST_BUILD_DIR" --target compilefornpu -j"$BUILD_JOBS"
```

QNN 设备端执行时需要在 `$REMOTE_ROOT` 下运行，并传入模型子目录中的 config：

```bash
cd /data/local/tmp/mnn-profiler &&
export LD_LIBRARY_PATH=/data/local/tmp/mnn-profiler:${LD_LIBRARY_PATH:-} &&
export ADSP_LIBRARY_PATH=/data/local/tmp/mnn-profiler:${ADSP_LIBRARY_PATH:-} &&
/data/local/tmp/mnn-profiler/profiler qnn_p128_d6/config_qnn.json \
    --prompt-lens=128 \
    --decode-lens=6 \
    --warmup=2 \
    --repeat=3
```

---

## 2. 关键实现说明

### 2.1 接口调用逻辑图

```
┌─────────────────────────────────────────────────────────────────┐
│                        LLM Layer                                │
│                                                                 │
│  Llm::enableProfiler(true)                                      │
│        │                                                        │
│        ▼                                                        │
│  Executor::setCallBack(beforeOp, afterOp)                       │
│        │                                                        │
│        ▼                                                        │
│  LLMOpProfiler::beforeOp() / afterOp()                          │
│        │                                                        │
│        ├─ CPU / fallback                                        │
│        │    └─ Runtime::recordOpProfileTime()                   │
│        │                                                        │
│        └─ OpenCL / QNN                                          │
│             ├─ beforeOp() -> Runtime::profileBegin()            │
│             └─ afterOp()  -> Runtime::profileEnd()              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Runtime 基类 (Backend.hpp)                 │
│                                                                 │
│  profileBegin(tensors, info)                                    │
│  profileEnd(tensors, info)                                      │
│  onGetProfileData()                                             │
│  onClearProfileData()                                           │
│  recordOpProfileTime(opName, opType, timeUs, backendName)       │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ CPU Runtime     │   │ OpenCL Runtime  │   │ QNN Runtime     │
│                 │   │                 │   │                 │
│ callback 同步计时│   │ profileBegin()  │   │ profileBegin()  │
│ 直接 record      │   │ profileEnd()    │   │ profileEnd()    │
│                 │   │ 记录当前 Op      │   │ graphExecute()  │
│                 │   │                 │   │ 后读取 QNN      │
│                 │   │                 │   │ Profile API     │
│                 │   │ pushEvent()      │   │                 │
│                 │   │ 记录 cl::Event   │   │ recordProfileData │
│                 │   │                 │   │                 │
│                 │   │ onGetProfileData │   │ recordOpProfile │
│                 │   │ 读取 START/END   │   │ backendName=QNN │
└─────────────────┘   └─────────────────┘   └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Llm::collectBackendProfileData()                │
│                                                                 │
│  runtime->onGetProfileData()                                    │
│        │                                                        │
│        ▼                                                        │
│  LLMOpProfiler::collectBackendProfile(data)                     │
│        │                                                        │
│        ▼                                                        │
│  LLMOpProfiler::printStats() / printOpInfo()                    │
└─────────────────────────────────────────────────────────────────┘
```

阶段调用关系：

```
Prefill:
  onPrefillStart()
    -> callback beforeOp()/afterOp()
    -> collectBackendProfileData()
    -> onPrefillEnd(promptTokenCount)

Decode:
  onDecodePhaseStart()
    -> onDecodeTokenBegin()
    -> forward / forwardRaw / graphExecute
    -> onDecodeTokenEnd(count)
    -> collectBackendProfileData()
  onDecodePhaseEnd()
```

Eagle 推测解码下会在 target / draft 执行前调用 `setActiveProfiler()` 切换当前 profiler。切换时会先调用 `collectBackendProfileData()`，避免 OpenCL / QNN runtime 中已经累计的后端 profile 数据被归到错误的 profiler。

### 2.2 LLMOpProfiler

核心目标是把 LLM 推理拆成两个维度统计：

- 阶段维度：Prefill / Decode
- 算子维度：按 OpName、OpType、Backend 聚合

核心接口：

```cpp
void onPrefillStart();
void onPrefillEnd(int promptTokenCount);
void onDecodePhaseStart();
void onDecodePhaseEnd();
void onDecodeTokenBegin();
void onDecodeTokenEnd(int count = 1);
bool beforeOp(const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info);
void afterOp(const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info);
void collectBackendProfile(const BackendProfileData& data);
void printStats() const;
void printOpInfo() const;
```

### 2.3 Runtime Profile 接口

`source/core/Backend.hpp` 添加 Runtime 级 profile 数据接口：

```cpp
virtual void profileBegin(const std::vector<MNN::Tensor*>& tensors,
                          const MNN::OperatorInfo* info) const;
virtual void profileEnd(const std::vector<MNN::Tensor*>& tensors,
                        const MNN::OperatorInfo* info) const;
virtual std::map<std::string, OpProfileInfo> onGetProfileData() const;
virtual void onClearProfileData();
void recordOpProfileTime(const std::string& opName,
                         const std::string& opType,
                         uint64_t timeUs,
                         const std::string& backendName = "");
```

CPU 同步执行时直接记录 wall-clock 时间。OpenCL / QNN 这类异步或整图后端通过 Runtime 收集设备侧事件，再统一进入 `LLMOpProfiler::collectBackendProfile()`。

### 2.4 CPU 后端计时方式

CPU 路径使用 callback 直接计时：

- `beforeOp()` 记录开始时间
- `afterOp()` 记录结束时间
- 根据 `Tensor` 实际绑定的 backend 处理 fallback 情况
- 最终调用 Runtime 的 `recordOpProfileTime()` 聚合

这个方案对 CPU 可靠，逻辑简单，适合精确统计同步执行的 Op。

### 2.5 OpenCL 异步 Kernel 统计

OpenCL 不能简单复用 CPU 的同步回调计时，因为 kernel 是异步提交的。当前采用 event 归属方案：

- `beforeOp()` 调 `Runtime::profileBegin()`，OpenCL runtime 创建当前 Op 的 profile entry
- kernel enqueue 后通过 `pushEvent()` 把 OpenCL event 记录到当前 Op entry
- `onGetProfileData()` 中 `commandQueue().finish()`，再读取每个 event 的 `START/END`
- 单个 Op 内多个 kernel 使用 `max(end) - min(start)` 作为设备执行窗口
- 最终调用 `Runtime::recordOpProfileTime()` 进入统一聚合表

该统计依赖 `MNN_OP_TIME_PROFILE=ON` 或 `MNN_GPU_TIME_PROFILE=ON`。

### 2.6 QNN NPU Node 统计

QNN 后端的 MNN op 在 resize 阶段被编码成 QNN graph node，真正执行发生在 QNN graph execute 中。因此 QNN 不使用 MNN callback 的 wall-clock 作为 op 时间，而是读取 QNN Profile API：

- graph 执行时传入 profile handle
- graph execute 后调用 `recordProfileData()`
- 遍历 QNN profile event
- 记录 `QNN_PROFILE_EVENTTYPE_NODE`
- 优先使用 `QNN_PROFILE_EVENTUNIT_MICROSEC`
- 如果设备只返回 cycles，则在没有 microsecond node 的情况下，按 execute wall time 和 cycles 占比估算
- 使用 QNN node `identifier` 作为 op name，并从名称前缀推导 op type
- 最终调用 `Runtime::recordOpProfileTime(..., "QNN")` 进入统一聚合表

### 2.7 QNN 导出脚本

`transformers/llm/export/npu/generate_llm_qnn.py` 当前流程：

1. `generateLlmIO` 生成不同 chunk size 的输入输出样例
2. `compilefornpu` 分离 QNN 子图并生成 `npu_postreat.json`
3. `npu_convert.py` 调用 QNN SDK 工具生成 context binary
4. 移动生成的 `qnn/` 目录到模型目录，并写入 `config_qnn.json`

脚本会检查关键步骤返回码和最终 `qnn/llm.mnn` 是否存在，避免导出失败后继续进入设备端运行。

---

## 3. 当前状态

| 后端 | 状态 | 说明 |
|------|------|------|
| CPU | 支持 | 使用 callback 直接统计 |
| OpenCL | 支持 | 使用 OpenCL event 统计 kernel 设备执行时间，依赖 `MNN_OP_TIME_PROFILE` 或 `MNN_GPU_TIME_PROFILE` |
| QNN | 支持 | 使用 QNN detailed profile 的 node event，依赖 `MNN_OP_TIME_PROFILE` 或 `MNN_GPU_TIME_PROFILE` |
| Metal / CUDA / Vulkan | 暂未接入 LLM profiler | 需要后续扩展 |

### 3.1 修改文件汇总

| 文件路径 | 修改类型 | 说明 |
|----------|----------|------|
| `CMakeLists.txt` | 修改 | 添加 `MNN_OP_TIME_PROFILE` 上层开关，并向 C++ 传递同名宏 |
| `source/core/Backend.hpp` | 修改 | 添加 `OpProfileInfo` 和 Runtime profile 接口 |
| `source/backend/opencl/core/OpenCLBackend.cpp` | 修改 | 汇总 OpenCL event 时间 |
| `source/backend/opencl/core/runtime/OpenCLRuntime.hpp` | 修改 | 记录 OpenCL op profile entry 和 kernel event |
| `source/backend/opencl/core/runtime/OpenCLRuntime.cpp` | 修改 | `MNN_OP_TIME_PROFILE` 下屏蔽逐 kernel 明细打印 |
| `source/backend/qnn/CMakeLists.txt` | 修改 | `MNN_GPU_TIME_PROFILE` 下启用 QNN time profiler 宏 |
| `source/backend/qnn/backend/QNNBackend.cpp` | 修改 | 读取 QNN Profile node 时间并写入 Runtime profile 表 |
| `source/backend/qnn/backend/QNNBackend.hpp` | 修改 | 添加 QNN profile 数据收集入口 |
| `transformers/llm/engine/include/llm/llm.hpp` | 修改 | 添加 profiler、active profiler、SpecContext 相关接口 |
| `transformers/llm/engine/include/llm/llm_profiler.hpp` | 修改 | 增强 profiler 数据结构与接口 |
| `transformers/llm/engine/src/llm.cpp` | 修改 | 集成 profiler、切换 active profiler、收集 backend profile、暴露 SpecContext |
| `transformers/llm/engine/src/llm_profiler.cpp` | 修改 | 实现阶段统计、后端 profile 聚合和输出 |
| `transformers/llm/engine/src/speculative_decoding/generate.hpp` | 修改 | 添加 `SpecContext` 统计接口 |
| `transformers/llm/engine/src/speculative_decoding/eagle.cpp` | 修改 | 记录 SpecContext，切换 draft / target profiler |
| `transformers/llm/engine/demo/profiler.cpp` | 新增 | LLM profiler 专用测试工具 |
| `transformers/llm/engine/demo/eagle_benchmark.cpp` | 新增 | Eagle 性能基准测试工具 |
| `transformers/llm/engine/demo/spec_eval.cpp` | 新增 | Speculative Decoding 评估工具，替代旧 `eagle_eval` |
| `transformers/llm/engine/CMakeLists.txt` | 修改 | 添加 `profiler` / `eagle_benchmark` / `spec_eval` 可执行目标 |
| `transformers/llm/export/npu/generate_llm_qnn.py` | 修改 | 增加 QNN 导出失败检查 |
| `project/android/profile.sh` | 新增 | Android CPU / OpenCL / QNN profiler 全流程测试脚本 |

### 3.2 已知限制与注意事项

1. OpenCL / QNN 后端 Op 时间统计依赖 `MNN_OP_TIME_PROFILE` 或 `MNN_GPU_TIME_PROFILE`，未开启时不会输出对应后端的设备级 Op 时间。
2. Token 级耗时和 Op 级累计耗时可能不完全一致，因为前者包含采样、调度、同步等额外开销。
3. QNN host 导出工具和 Android runtime 是两条独立构建路径，host convert mode 不应继承 runtime profile 开关。
4. QNN 设备端运行需要在 `$REMOTE_ROOT` 下执行，并传入 `qnn_p*_d*/config_qnn.json`，不要在 QNN 模型目录内直接执行。
5. `profile.sh` 默认只跑 `BACKENDS=(qnn)`，需要测试 CPU 或 OpenCL 时修改 `BACKENDS` 或通过环境覆盖。

---
