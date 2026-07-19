# MNN 双流水线、QNN 动态预取与 Resident Cache 代码 Review

本文只解释当前仓库的真实代码路径和版本差异。主入口是 `Llm::generate(vector<vector<int>>)`，重点跟踪配置、请求分组、QNN 图预测、预取 worker、Host/QNN stage、两个独立 Module，以及跨 decode wave 的 resident graph 生命周期。

## 0. 模块接口调用总图

下面先给出接口边界，再按同一条线展开。

~~~text
用户 / llm_demo
    │
    ├─ Llm::createLLM(config)
    │      └─ LlmConfig::dual_pipeline_*()
    │
    ├─ Llm::load()
    │      ├─ 主 Module + RuntimeManager
    │      ├─ refreshDualPipelineGraphSnapshot()
    │      │      ├─ Session::getSession() -> buildGraphSnapshot()
    │      │      ├─ FlatBuffer model -> buildQnnGraphSnapshotFromModel()
    │      │      └─ mergeQnnGraphSnapshotsInExecutionOrder()
    │      └─ prepareDualPipelineExecutionState()
    │             ├─ Executor::newExecutor() x 2
    │             ├─ RuntimeManager + BatchKVMeta x 2
    │             └─ Executor::setCallBack(before, after)
    │
    └─ Llm::generate(batch)
           │
           ├─ BatchScheduler::scheduleWave()
           │      ├─ request -> stable pipelineId
           │      └─ same segment -> pipeline 0/1 disjoint chunks
           │
           ├─ dualPipelinePaddedCulLen(chunk)
           │      └─ selectQnnCompatibleBucketSize()
           │             ├─ pipeline group size=2 -> bucket 8
           │             └─ pipeline group size=1 -> bucket 1
           │
           ├─ prepareDualPipelineBatchMeta()
           │      └─ upper BatchKVMeta -> pipeline BatchKVMeta
           │
           ├─ DualPipelineScheduler::beginGraphPrefetchWave()
           │      └─ graph worker queue: GRAPH_LOAD
           │             ├─ distance(cursor) priority
           │             ├─ resident/loading dedup
           │             └─ callback -> QNN::preloadRawGraph()
           │                    └─ RawExecutorWrapper::compileModel()
           │
           ├─ getDualPipelineModule()
           │      ├─ first bucket -> Module::load()
           │      └─ later bucket -> Module::clone()
           │             └─ pipeline-local KV cache manager preserved
           │
           ├─ DualPipelineScheduler::beginStageWave()
           │      ├─ Host ready queue
           │      └─ QNN ready queue
           │
           ├─ pipeline worker 0/1: Module::onForward()
           │      └─ Executor callback
           │             ├─ Host op -> enterStage(STAGE_HOST)
           │             └─ QNN Plugin -> enterGraphStage()
           │                    ├─ wait current graph resident
           │                    ├─ execution-priority gate
           │                    └─ enterStage(STAGE_QNN)
           │
           ├─ after callback
           │      ├─ leaveStage()
           │      └─ leaveGraphStage()
           │             ├─ clear mQnnExecutionActive
           │             ├─ TASK_GRAPH_COMPLETE
           │             └─ activeUseCount-- / LRU touch
           │
           ├─ finishStageWave() + finishGraphPrefetchWave()
           ├─ syncDualPipelineBatchMeta()
           ├─ sampling / scheduler update
           └─ releaseDualPipelineRequestExecution(requestId)
                  ├─ BatchKVMeta::releaseKV()
                  └─ releaseRequestGraphs()  // 只清 owner，不立即销毁 resident

容量不足 / scheduler stop
    └─ onGraphRelease()
           └─ QNN::releaseRawGraph() / releaseAllRawGraphs()
~~~

### 核心并发关系

~~~text
pipeline 0: Host attention ─────── QNN graph A ───── Host ─── QNN graph B
pipeline 1:       QNN graph X ─── Host attention ─── QNN graph Y
                         │
                         └─ 允许 Host + QNN 重叠

QNN compile 与 QNN execute:
    gQnnContextOperationMutex 串行
    因此当前版本优先做跨 wave resident，避免重复 compile
~~~

## 1. Review 版本边界

| 版本 | 基线 | 主要内容 |
|---|---|---|
| `v0.1` | `b2f3a713` | 请求分组、图快照、基础 graph scheduler、LLM 双 pipeline 骨架 |
| `v0.2` | `dd225512` | QNN raw graph prefetch API、QNN pool、动态窗口和 Host/QNN 回调接线 |
| `v0.3` | `66c72045` | 两套 Executor/Runtime/Module/KV、same-segment wave、stage ready queue、bucket/order 修复 |
| 当前 worktree | `HEAD + worktree` | resident LRU、请求 owner 清理、执行优先预取、Android 测试/构建辅助 |

`b2f3a713..HEAD` 共涉及 21 个历史实现/测试文件；当前未提交层新增 8 个文件的修改，详见第 12 节。

## 2. 配置进入运行时

配置 accessor 位于 `transformers/llm/engine/src/llmconfig.hpp:341`：

~~~cpp
// transformers/llm/engine/src/llmconfig.hpp:341
bool dual_pipeline_mode() const {
    return config_.value("dual_pipeline_mode", config_.value("dual_pipeline", false));
}
int dual_pipeline_split_count() const {
    return config_.value("dual_pipeline_split_count", 2);
}
int dual_pipeline_max_resident_graphs() const {
    return config_.value("dual_pipeline_max_resident_graphs", 5);
}
int dual_pipeline_prefetch_window() const {
    return config_.value("dual_pipeline_prefetch_window", 2);
}
~~~

`Llm::configureDualPipelineMode()` 在 `transformers/llm/engine/src/llm.cpp:205`：

1. 调用 `BatchScheduler::setDualPipelineMode()`。
2. 创建并配置 `DualPipelineScheduler`。
3. 把 scheduler 的 load/release callback 绑定到 `QNN::preloadRawGraph()` 和 `QNN::releaseRawGraph()`。
4. 启动 graph worker。

关闭 dual mode 时，scheduler、图状态和执行资源会被重置；非 dual 路径仍走原有单 Module 流程。

## 3. 请求如何形成双 pipeline wave

### 3.1 `BatchScheduler` 的稳定绑定

`Chunk` 增加 `pipelineId` 和 `segmentIndex`，见 `transformers/llm/engine/include/llm/BatchScheduler.hpp:59`。请求第一次进入 scheduler 时建立 request-to-pipeline 绑定，后续 prefill/decode 保持同一逻辑 pipeline。

`scheduleWave()` 位于 `transformers/llm/engine/src/BatchScheduler.cpp:213`：

~~~cpp
// transformers/llm/engine/src/BatchScheduler.cpp:213
std::vector<std::shared_ptr<BatchScheduler::Chunk>> BatchScheduler::scheduleWave(int blockSize, int bs) {
    std::vector<std::shared_ptr<Chunk>> wave;
    auto first = schedule(blockSize, bs);
    if (!first) {
        return wave;
    }
    wave.push_back(first);
    if (!mDualPipelineMode) {
        return wave;
    }
    const int segmentIndex = first->segmentIndex;
    while (!mPendingChunks.empty()) {
        const auto& next = mPendingChunks.front();
        if (!next || next->segmentIndex != segmentIndex) {
            break;
        }
        wave.push_back(_popPendingChunk());
    }
    return wave;
}
~~~

它保证 coordinator 一次拿到同一个 segment 的两个不相交 chunk，而不是调用两次 `schedule()` 后猜测是否属于同一 wave。当前三请求用例是 pipeline 0: 请求 1/2，pipeline 1: 请求 3。

### 3.2 请求大小到 QNN bucket

`Llm::dualPipelinePaddedCulLen()` 位于 `llm.cpp:437`，根据 chunk 的 packed 长度调用 `selectQnnCompatibleBucketSize()`。当前测试模型 buckets 为 `1/8/128`：

~~~text
pipeline 0: group size 2 -> QNN shape bucket 8
pipeline 1: group size 1 -> QNN shape bucket 1
~~~

如果不存在能容纳当前 packed length 的 bucket，`generate()` 在创建 Module 和启动 worker 前设置 `INTERNAL_ERROR`，避免执行到一半才发现图形状不匹配。

## 4. 双 pipeline 的执行资源与 KV 状态

### 4.1 两套 Executor/Runtime/Module

`prepareDualPipelineExecutionState()` 位于 `llm.cpp:277`，对每条 pipeline 创建独立的 `Executor`、`RuntimeManager`、`BatchKVMeta`，并给 Executor 安装 before/after callback。

`getDualPipelineModule()` 位于 `llm.cpp:349`：

- 每条 pipeline 的第一个 bucket 通过 `Module::load()` 创建。
- 同一 pipeline 后续 bucket 从已有 Module `clone()`。
- clone 后保留该 pipeline 的 packed-attention KV cache manager，而不是并发使用共享 Module/Session。
- QNN clone 路径继承 NPU model directory，使 plugin 查找和预取 cache key 一致。

### 4.2 上层与临时 `BatchKVMeta`

`prepareDualPipelineBatchMeta()` / `syncDualPipelineBatchMeta()` 位于 `llm.cpp:78` 和 `llm.cpp:94`。上层 `mBatchMeta` 作为 wave 的统一视图；每个 pipeline 在 forward 前复制自己的 request entry，forward 完成后再同步回上层。请求完成时 `releaseDualPipelineRequestExecution()` 同时释放 pipeline KV 和 graph owner。

dual mode 会在 `llm.cpp:501` 强制 packed attention，因此普通单请求 API 仍需走与 batch metadata 一致的调用约束；第 14 节列出这一边界。

## 5. QNN 图顺序、资源标识和 bucket metadata

### 5.1 两个 snapshot 的职责

`refreshDualPipelineGraphSnapshot()` 位于 `llm.cpp:399`：

~~~text
Session snapshot: 当前实际 Session command 顺序，可能被 resize/prune 截断
Model snapshot:   FlatBuffer 中完整 QNN Plugin binary metadata
                         │
                         ▼
mergeQnnGraphSnapshotsInExecutionOrder()
    ├─ Session 可观察顺序优先
    ├─ model-only QNN op 补到序列尾部
    └─ buildQnnGraphRequests()
~~~

每个 QNN op 再映射到唯一 `opName -> graphIndex`。重复 op name 或 graph request 数量不一致会使 snapshot 无效，避免 callback 使用错误图。

### 5.2 physical graph id 与 shape graph

历史版本曾按 target graph name 生成 id，shape variant 容易覆盖同一 binary。当前 request 将 path、offset、size、allGraphName 纳入 physical resource identity；shape index/target graph name 仍用于 `RawExecutorWrapper::invokModel(shapeIndex)` 选择具体 QNN graph。

## 6. 动态预取窗口和 loader priority

### 6.1 初始窗口

`DualPipelineScheduler::beginGraphPrefetchWave()` 位于 `transformers/llm/engine/src/DualPipelineScheduler.cpp:179`。每条 pipeline 只 enqueue 当前 cursor 后 `W` 个 graph，不再等待整条 forward 的所有 graph ready。

每条 pipeline 的状态位于 `DualPipelineScheduler.hpp:176`：

| 状态 | 含义 |
|---|---|
| `currentGraphIndex` | 最近进入的 QNN graph |
| `requestedUntil` | 已请求窗口末端 |
| `completedGraphIndex` | 最近完成 graph |
| `executingGraphIndex` | 当前持有 QNN lane 的 graph |
| `registeredGraphIndices` | 已向 physical graph 注册 active use 的索引 |
| `mPendingQnnGraphs` | 已到达 QNN command、正在等 graph/lane 的 pipeline |

### 6.2 动态距离排序

`_graphLoadDistanceLocked()` 和 `_popNextTaskLocked()` 位于 `DualPipelineScheduler.cpp:547`、`:558`：

~~~text
pipeline 0 current=B -> C(distance=1), D(distance=2)
pipeline 1 current=E -> F(distance=1), G(distance=2)

优先级：C/F -> D/G
同距离：pipelineId -> enqueueSequence
~~~

worker 在 `DualPipelineScheduler.cpp:638`。`_hasRunnableTaskLocked()` 观察 `mQnnExecutionActive` 和 `mPendingQnnGraphs`：

- 当前图未 ready：允许 loader 继续加载当前图。
- 当前图 ready、pipeline 等待 QNN lane：暂停未来图 compile，优先当前执行。
- QNN execute 完成：清除 gate，loader 可在下一个 Host 区间继续预取。

这不是绕过 QNN 全局锁，而是把 compile 尽量放到 Host attention 区间，并避免 ready graph 被未来 compile 抢占。

### 6.3 跳图和取消

`enterGraphStage()` 位于 `DualPipelineScheduler.cpp:218`。当实际 callback index 向前跳时：

1. 删除尚未开始的区间 load task。
2. 为已注册但未执行的预测图 enqueue `TASK_GRAPH_COMPLETE`，使 active use 归零。
3. 从实际 index 重新扩展 lookahead。
4. 只等待实际当前 graph resident。

未执行 tail 不再使 wave 失败；它们是 cleanup，而不是执行成功条件。

## 7. Host/QNN stage 接口

### 7.1 callback 到 scheduler

`prepareDualPipelineExecutionState()` 的 before callback：

- QNN Plugin op name 命中 `mDualPipelineQnnOpIndices` -> `enterGraphStage(pipelineId, graphIndex)`。
- 其他有 `OperatorInfo` 的 command -> `enterStage(pipelineId, STAGE_HOST)`。

after callback 对称调用 `leaveGraphStage()` 或 `leaveStage()`。`cmd.info == nullptr` 的 command 仍绕过这套分类。

### 7.2 两个 ready queue

`beginStageWave()`、`enterStage()`、`leaveStage()` 位于 `DualPipelineScheduler.cpp:393`、`:421`、`:443`：

- Host lane 同时只授予一个 Host stage。
- QNN lane 当前也只授予一个 QNN stage，保证 QNN binary executor 的调用安全。
- Host 和 QNN 可同时 active；`hostQnnOverlapGrants` 记录过 overlap grant。
- pipeline 自身不能在前一个 command 未 leave 前再次进入 stage。

因此实现的是 command-boundary Host/QNN overlap，不是把 attention 和 FFN 拆成可独立传递 tensor 的两个完整 stage graph。

## 8. QNN prefetch API 与 raw pool

### 8.1 public API

`include/MNN/QNNPrefetch.hpp:17` 定义配置，公开接口位于 `:26`：

~~~cpp
// include/MNN/QNNPrefetch.hpp:26
bool preloadRawGraph(const RawGraphPrefetchConfig& config);
void releaseRawGraph(const std::string& graphId, bool forceRelease, bool unpinAfterRelease);
void releaseAllRawGraphs();
~~~

LLM scheduler 只依赖这个边界，不直接调用 QNN private loader。

### 8.2 pool lookup/load/release

`source/backend/qnn/backend/QNNBackend.cpp:942` 用 normalized path、offset、size、allGraphName 生成 cache key。`preloadRawGraphInternal()` 在 `:976`：

1. 先按 graph id / physical cache key 去重。
2. miss 时创建 `RawExecutorWrapper`，调用 `compileModel()`。
3. 将 executor 存到 `gRawGraphByKey`，并建立 graph id alias。

`PluginExecuteRaw::compute()` 在 `QNNBackend.cpp:1127` 每次按 physical key 查 pool，再调用 `invokModel(shapeIndex)`。pool miss 仍保留 fallback compile，这是当前 dual prefetch 失效时的兼容路径。

`compileModel()` 和 `invokModel()` 分别在 `QNNBackend.cpp:789`、`:861` 持有 `gQnnContextOperationMutex`，所以 compile/execute 不能真正并行。

## 9. 跨 wave resident graph 生命周期

### 9.1 正常完成不释放

`leaveGraphStage()` 在 `DualPipelineScheduler.cpp:288` 只 enqueue `TASK_GRAPH_COMPLETE`。worker 的 `_processGraphComplete()` 在 `:743`：

~~~text
activeUseCount > 0 -> --activeUseCount
lastUseSequence = next sequence
resident graph 保留
~~~

下一 decode wave 请求同一 graph 时，`_processGraphRequest()` 将其视为 resident hit，不再次调用 `onGraphLoad`。

### 9.2 LRU eviction

`_planEvictionsLocked()` 在 `DualPipelineScheduler.cpp:755`：

- resident 且非 pinned 且 `activeUseCount == 0` 才能作为 candidate。
- 选择最小 `lastUseSequence`。
- 标记非 resident，清 ownership，回调 `onGraphRelease(forceRelease=true)`。
- active graph 不能被执行中回收，因此配置上限是软水位。

### 9.3 request owner 与 scheduler stop

`releaseRequestGraphs(requestId)` 在 `DualPipelineScheduler.cpp:359` 只从各 graph 的 owner set 删除 request id。`Llm::releaseDualPipelineRequestExecution()` 在 `llm.cpp:449` 同步调用它和 `BatchKVMeta::releaseKV()`。

`DualPipelineScheduler::stop()` 在 `DualPipelineScheduler.cpp:136` 等 worker 退出后遍历 resident graph，统一发出 force release；`Llm::~Llm()` 再按 Module -> runtime -> QNN pool 的顺序清理。

## 10. 端到端 `generate(batch)` 主线

双 pipeline 分支位于 `llm.cpp:1264`，每个 wave 顺序如下：

1. `scheduleWave()` 取同 segment chunks。
2. 对每个 chunk 做 QNN bucket 预校验。
3. 构造 embedding、mask、position ids。
4. 准备 per-pipeline `BatchKVMeta`。
5. `beginGraphPrefetchWave()` 提交图窗口。
6. `getDualPipelineModule()` 获取对应 bucket Module。
7. `beginStageWave()` 建立 Host/QNN stage wave。
8. 为每个 chunk 启动一个 worker，进入独立 ExecutorScope，调用 `Module::onForward()`。
9. join workers，`finishStageWave()`、`finishGraphPrefetchWave()`。
10. 同步 pipeline metadata，采样、更新 request、释放结束请求。

如果 Module forward 返回空 logits、stage wave 失败、graph load 失败或 bucket 不支持，设置 `LlmStatus::INTERNAL_ERROR`，不继续采样无效输出。

## 11. 当前 worktree 的冗余清理

本轮跨 wave 改动同时删除了原来已经失效的 per-wave release 状态：

| 删除项 | 原因 |
|---|---|
| `TASK_GRAPH_RELEASE` | 正常 graph complete 不再触发释放；真实释放只有 LRU 和 stop |
| `GraphRecord::pendingRelease` | 不再需要等待 active use 归零后销毁 |
| `pendingReleaseRequest` | 没有延迟 release callback 入口 |
| 每个 owner 的 graph release task | owner 清理改为显式 `releaseRequestGraphs()` |

历史版本已经删除/替换的冗余接口包括 `waitForGraphsReady()`、`markExecutionComplete()`、旧的全图 barrier 测试、重复 QNN resize tensor mapping，以及被全局 QNN operation mutex 覆盖的 `mInvokeMutex`。

## 12. 逐文件修改版本表

### 12.1 `v0.1` 到 `v0.3` 的实现文件

| 文件 | `v0.1` 初始 | `v0.2` QNN/预取 | `v0.3` 真双执行 | 当前关注 |
|---|---|---|---|---|
| `transformers/llm/engine/src/llm.cpp` | 双模式入口和基础 graph 调度 | QNN graph callback、bucket、metadata | 两套 runtime/module/KV、并行 wave | resident owner 清理、执行主线 |
| `transformers/llm/engine/src/BatchScheduler.cpp` | request chunk | pipeline/segment metadata | `scheduleWave()` 同 segment 聚合 | 稳定 pipeline affinity |
| `transformers/llm/engine/src/DualPipelineGraph.cpp` | Session snapshot | QNN binary metadata/request | order merge、bucket/resource id | Session 优先、model-only tail |
| `transformers/llm/engine/src/DualPipelineScheduler.cpp` | graph worker/window | QNN load/release、dynamic priority | Host/QNN ready queues、readiness/skip | resident LRU、pending QNN gate |
| `source/backend/qnn/backend/QNNBackend.cpp` | 原有 raw QNN executor | preload/release pool、lazy lookup | clone/path/cache 修复 | global mutex、fallback compile |
| `express/Executor.cpp` | 原 callback 路径 | callback 存储 | 每个 Executor 的 pipeline callback | callback 失败清理边界 |
| `express/module/StaticModule.cpp` | 原 Module 执行 | QNN model directory 支持 | clone 后保持 pipeline 资源 | clone Session/KV 一致性 |
| `transformers/llm/engine/src/kvmeta.hpp` | 单 KVMeta | BatchKVMeta | pipeline KV release/sync | upper 与 temporary meta |
| `transformers/llm/engine/src/llmconfig.hpp` | 原配置 | dual/prefetch 配置 | 默认值兼容 | memory policy 默认值 |

### 12.2 头文件和构建接口

| 文件 | 修改版本 | 接口 |
|---|---|---|
| `transformers/llm/engine/include/llm/BatchScheduler.hpp` | `v0.1`/`v0.3` | `Chunk.pipelineId`、`segmentIndex`、`scheduleWave()` |
| `transformers/llm/engine/include/llm/DualPipelineGraph.hpp` | `v0.1`~`v0.3` | snapshot、bucket、QNN request 构造 |
| `transformers/llm/engine/include/llm/DualPipelineScheduler.hpp` | `v0.1`~当前 | graph/stage API、LRU state、`releaseRequestGraphs()` |
| `transformers/llm/engine/include/llm/llm.hpp` | `v0.1`~`v0.3` | dual runtime/module/KV 成员和 helper |
| `include/MNN/QNNPrefetch.hpp` | `v0.2` | public QNN preload/release 边界 |
| `CMakeLists.txt` | `v0.2` | LLM/QNN 构建开关接线 |
| `test/CMakeLists.txt` | `v0.2`/`v0.3` | LLM scheduler/graph/stage 测试源接线 |

### 12.3 测试与设备辅助文件

| 文件 | 修改内容 | 用途 |
|---|---|---|
| `test/llm/BatchSchedulerTest.cpp` | 2/3/4 request partition、wave、pipeline persistence | 验证请求分组不重叠 |
| `test/llm/DualPipelineGraphTest.cpp` | bucket、physical id、order merge、metadata | 验证 QNN 图预测 |
| `test/llm/DualPipelineSchedulerTest.cpp` | priority、gap、reuse、LRU、owner、execution gate | 验证 graph 生命周期 |
| `test/llm/DualPipelineStageSchedulerTest.cpp` | Host/QNN overlap、QNN 串行、取消 | 验证 stage queue |
| `transformers/llm/engine/demo/llm_demo.cpp` | `--dual-batch-test` 三请求入口 | Android smoke test |
| `project/android/build_64.sh` | 显式打开 LLM/OpenCL/QNN/plugin/online finalize | 可复现 Android 构建 |
| `project/android/mv2adb.sh` | shebang、目标目录、`libQnnSystem.so` | 完整部署 QNN runtime |
| `dual_pipeline_code_review.md` | 本文 | review 主线和版本表 |

## 13. 运行时清理顺序

~~~text
request finished
    -> mScheduler->releaseKVCache()
    -> BatchKVMeta::releaseKV(requestId)
    -> DualPipelineScheduler::releaseRequestGraphs(requestId)
    -> resident graph 继续可复用

Llm::~Llm()
    -> DualPipelineScheduler::stop()
       -> worker join
       -> release all resident graph
    -> reset dual Executor callbacks / dual Runtime / dual Module pools
    -> clear main Module pool / main Module
    -> QNN::releaseAllRawGraphs()
    -> reset RuntimeManager
~~~

这里的关键分离是：request 生命周期不再等于 QNN graph 生命周期；graph 生命周期由 resident capacity 和 scheduler teardown 管理。

## 14. 验证结果与 review 边界

### 14.1 Host

当前 `/tmp/mnn-pp-review-llm/run_test.out llm`：

~~~text
passed: 30
failed: 0
blocked: 0
skipped: 0
~~~

专项 `llm/dual_pipeline_scheduler_execution_before_lookahead` 连续 5 次通过；`git diff --check` 通过。

### 14.2 Android 三请求

设备：`192.168.124.101:47954`；模型：`/workspace/code/mnn-profiler/project/android/llm_profile_work/qnn_fixed_s1_8_128`。

`--dual-batch-test` 使用 resident limit 30：

~~~text
2-token real: 5.03s   (原记录约 12.52s)
16-token real: 7.36s (原记录约 47.06s)
ReqId 0/1/2: 有效输出
exit: 0
~~~

当前日志没有 `Acquire buffer size = 0`、QNN validate/execute、prefetch、stage、no-logits、SIGSEGV 或 FORTIFY 错误。

### 14.3 不能过度推断的边界

1. QNN compile/execute 仍被 `gQnnContextOperationMutex` 串行；当前优化是 resident reuse + execution-priority，不是 compile/execute 真并行。
2. `maxResidentGraphs` 是软水位，active graph 不会被强制淘汰，内存可能临时超过配置值。
3. QNN raw pool 是进程级全局池，多实例 `Llm` 没有 client refcount 隔离。
4. `PluginExecuteRaw` pool miss 仍允许 fallback compile；这会绕过 scheduler 的预取统计。
5. `cmd.info == nullptr` 的 command 不进入 Host/QNN stage callback。
6. OpenCL callback 当前证明的是 command submission 边界，不等于硬件时间线已经证明 CPU/OpenCL 与 QNN 完整重叠。
7. `Llm::load()` 对 snapshot mapping 失败仍保留兼容行为，可能让 Plugin fallback compile；这是后续可收紧的错误策略。
8. 当前没有独立的 OpenCL `onResize`/kernel compile prefetch queue；OpenCL 的提前准备来自 bucket Module/Session 创建与 MNN 原有 resize/cache 路径，`DualPipelineScheduler` 只在 command callback 层调度 Host lane。

## 15. 建议的 Review 阅读顺序

1. `transformers/llm/engine/src/llm.cpp:205`：dual mode 配置与 callback 注册。
2. `transformers/llm/engine/src/BatchScheduler.cpp:213`：same-segment wave。
3. `transformers/llm/engine/src/DualPipelineGraph.cpp:507`：Session/model order merge。
4. `transformers/llm/engine/src/llm.cpp:1264`：batch wave coordinator。
5. `transformers/llm/engine/src/DualPipelineScheduler.cpp:179`：初始预取窗口。
6. `transformers/llm/engine/src/DualPipelineScheduler.cpp:218`：current graph readiness 与 execution gate。
7. `transformers/llm/engine/src/DualPipelineScheduler.cpp:421`：Host/QNN ready queue。
8. `source/backend/qnn/backend/QNNBackend.cpp:976`：QNN raw graph pool preload。
9. `source/backend/qnn/backend/QNNBackend.cpp:1127`：Plugin lookup 和 QNN execute。
10. `transformers/llm/engine/src/DualPipelineScheduler.cpp:755`：resident LRU。
11. `test/llm/DualPipelineSchedulerTest.cpp:257`：跨 wave reuse / LRU / gate 测试。

## 小结

- 双 pipeline 的实际并发单位是两个隔离的 `Module::onForward()`，每条 pipeline 有自己的 Executor、RuntimeManager、Module pool 和 BatchKVMeta。
- 请求 wave 由 `BatchScheduler::scheduleWave()` 形成；QNN 图顺序由 Session snapshot 主导，FlatBuffer metadata 补全。
- graph worker 维护执行游标驱动的有界 lookahead；当前 QNN graph ready 后优先执行，未来 compile 延后到 Host 区间。
- QNN graph 完成后跨 wave resident，容量不足时 LRU 淘汰空闲图，request 完成只清 owner，scheduler stop 才释放全部图。
- 当前代码已经有 Host/QNN command-boundary overlap，但仍受全局 QNN mutex、fallback compile、全局 pool 和无 `OperatorInfo` command 等边界约束。
