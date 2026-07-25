# MNN 三模式服务、双流水线与 QNN 调度代码 Review

本文记录 `mnncli serve` 从启动配置、HTTP/SSE 请求、`RequestCoordinator`、`Llm` batch 分发，到 AR/Eagle 双 lane、Host/QNN stage 和 QNN resident graph 的真实代码路径。Review 基线是合并提交 `f7d890342d4b571b7a707fb0e263f90939deb970`，当前版本是该提交上的完整 worktree；Android 设备验收只覆盖现有 AR QNN 模型，不把主机 Eagle3 测试写成设备结果。

## 0. 模块接口调用总图

```text
`mnncli serve --config ... [--scheduler-mode ...]`
    |
    +-> `GetSpec("serve")`
    |      `apps/mnncli/src/cli_command_spec.cpp:61`
    |
    +-> `ServeCommandHandler::Handle()`
           `apps/mnncli/src/handlers/serve_command_handler.cpp:36`
           |
           +-> 校验 CLI mode
           +-> `LLMManager::CreateLLM(config, ..., scheduler_mode)`
           |      `apps/mnncli/src/llm_manager.cpp:14`
           |      |
           |      +-> `Llm::set_config()`          CLI 覆盖 JSON
           |      +-> `Llm::load()`                校验 mode/packed attention
           |             `transformers/llm/engine/src/llm.cpp:625`
           |             |
           |             +-> `prepareDualPipelineExecutionState()`  仅 dual
           |             +-> `refreshDualPipelineGraphSnapshot()`
           |             +-> 锁定 startup-only mode
           |
           +-> `MnncliServer::Start(..., llm->scheduler_mode())`
                  `apps/mnncli/src/mnncli_server.cpp:520`
                  |
                  +-> `/v1/models`                 readiness
                  +-> `/v1/chat/completions`
                         |
                         +-> `RequestCoordinator::submit()`
                         |      queue ownership / request id / trace
                         |
                         +-> worker thread
                         |      |
                         |      +-> `single_request`
                         |      |      `Llm::response()`
                         |      |        -> token text -> `Task::append()`
                         |      |
                         |      +-> `continuous_batch` / `dual_pipeline`
                         |             2 ms 合批，最多 4 个同 max_tokens 请求
                         |             -> tokenize
                         |             -> `Llm::generate(batch)`
                         |                    `transformers/llm/engine/src/llm.cpp:1271`
                         |                    |
                         |                    +-> packed AR / non-dual Eagle
                         |                    |      `Generation::generateBatch()`
                         |                    |
                         |                    +-> dual AR
                         |                    |      `BatchScheduler::scheduleWave()`
                         |                    |      -> request -> stable lane
                         |                    |      -> QNN compatible bucket
                         |                    |      -> lane-local KV/Module/Runtime
                         |                    |
                         |                    +-> dual Eagle3
                         |                           draft tree -> leaf prune
                         |                           -> target verify wave
                         |                           -> stable lane KV state
                         |                           `transformers/llm/engine/src/speculative_decoding/eagle_batch.cpp:925`
                         |                                  |
                         |                                  +-> graph prefetch wave
                         |                                  +-> stage wave
                         |
                         +-> `DualPipelineScheduler`
                         |      `transformers/llm/engine/include/llm/DualPipelineScheduler.hpp:24`
                         |      |
                         |      +-> graph cursor/window worker
                         |      |      -> load callback
                         |      |      -> QNN raw resident pool
                         |      |
                         |      +-> Host ready queue  -> max active Host = 1
                         |      +-> QNN ready queue   -> max active QNN  = 1
                         |             Host lane A 可与 QNN lane B overlap
                         |
                         +-> `PluginExecuteRaw::compute()`
                         |      `source/backend/qnn/backend/QNNBackend.cpp:1176`
                         |      -> per-shape binary lookup
                         |      -> cache miss 时 transient load/execute/release
                         |      -> QNN graph invoke
                         |      -> output copy
                         |
                         +-> HTTP response
                                non-stream: OpenAI JSON + usage
                                SSE: chunk -> finish frame -> `[DONE]`
                                disconnect: 100 ms 检查连接并取消/suppress output
```

这个总图先给出三个所有权边界：

| 边界 | 所有者 | 不变量 |
|---|---|---|
| HTTP 请求与 SSE 输出 | `RequestCoordinator` | 一个 `Llm` 实例只由 coordinator worker 调用 |
| request 到 lane、KV 和 wave | `BatchScheduler` + generation loop | 同一请求保持 lane 归属；一轮 wave 的 chunk 互不重叠 |
| Host/QNN command 与 QNN 图 | `DualPipelineScheduler` | Host 最多 1 个，QNN 最多 1 个，二者可重叠 |

## 1. Review 版本边界

| 项目 | 版本 |
|---|---|
| Review 基线 | `f7d890342d4b571b7a707fb0e263f90939deb970` |
| 基线组成 | Eagle batch `d37ea6d2` + dual runtime `6b2efd54` 的 merge |
| 当前版本 | `f7d89034` + 本 worktree 的服务模式、Eagle3/QNN、trace、构建与测试修改 |
| Android 模型 | `/data/local/tmp/qnn_fixed_s1_8_128/`，AR，实际 bucket 为 `1/8/128` |
| Android QNN runtime | `/data/local/tmp/mnn-qnn`，原目录只复用，不复制、不删除 |
| Eagle3 覆盖 | 主机单元测试、编译和导出路径；没有 Android Eagle3 实测 |

本文只描述当前可从源码或验证产物证明的行为。目标策略中的 `1/32/256/512`、W4 channel-wise 和 OpenCL attention 已进入导出默认值，但不能反向改写现有 `1/8/128` 设备模型的事实。

## 2. 配置选择与启动锁定

### 2.1 三种 mode

| `scheduler_mode` | 服务队列 | LLM 调用 | packed attention |
|---|---|---|---|
| `single_request` | 每次取 1 个 task | `Llm::response()` | 不强制 |
| `continuous_batch` | 2 ms 窗口，最多 4 个 task | `Llm::generate(batch)` | 强制开启 |
| `dual_pipeline` | 与 continuous 相同的 HTTP 合批 | dual AR/Eagle batch + 两 lane | 强制开启 |

`LlmConfig` 先读显式 `scheduler_mode`，未提供时才兼容旧 `dual_pipeline_mode` / `dual_pipeline` 布尔值。

```cpp
// transformers/llm/engine/src/llmconfig.hpp:341
std::string scheduler_mode() const {
    const auto& document = config_.document;
    if (document.HasMember("scheduler_mode")) {
        if (!document["scheduler_mode"].IsString()) {
            return "";
        }
        return document["scheduler_mode"].GetString();
    }
    const bool legacyDualPipeline =
        config_.value("dual_pipeline_mode", config_.value("dual_pipeline", false));
    return legacyDualPipeline ? "dual_pipeline" : "single_request";
}
```

CLI override 在 `load()` 之前合入 runtime config，因此优先级是 CLI > JSON > legacy fallback。

```cpp
// apps/mnncli/src/llm_manager.cpp:14
std::unique_ptr<MNN::Transformer::Llm> LLMManager::CreateLLM(
    const std::string& config_path, bool use_template, const std::string& scheduler_mode) {
    std::unique_ptr<MNN::Transformer::Llm> llm(MNN::Transformer::Llm::createLLM(config_path));
    // ... build runtime_config with scheduler_mode ...
    llm->set_config(runtime_config);
    if (!llm->load()) {
        return nullptr;
    }
    // ... tuning ...
    return llm;
}
```

`Llm::load()` 在创建 runtime/module 之前拒绝非法 mode 和缺少 packed attention 的 batch/dual 配置，并在成功加载后设置 `mSchedulerModeLocked`。这保证服务运行期间不会通过后续 `set_config()` 改变调度所有权。

## 3. HTTP/SSE 请求生命周期

### 3.1 `Task` 同时响应完成和取消

```cpp
// apps/mnncli/src/mnncli_server.cpp:31
bool wait() {
    std::unique_lock<std::mutex> lock(mutex);
    completed.wait(lock, [this] { return finished || cancelled.load(); });
    return !cancelled.load() && !failed;
}

// apps/mnncli/src/mnncli_server.cpp:66
ChunkState next_chunk(std::string& chunk) {
    std::unique_lock<std::mutex> lock(mutex);
    completed.wait_for(lock, std::chrono::milliseconds(100),
                       [this] { return !chunks.empty() || finished || cancelled.load(); });
    if (cancelled.load() || (finished && chunks.empty())) {
        return ChunkState::Finished;
    }
    if (chunks.empty()) {
        return ChunkState::Pending;
    }
    chunk = std::move(chunks.front());
    chunks.pop_front();
    return ChunkState::Ready;
}
```

待执行任务取消时，`cancel()` 直接从 `mPending` 删除并完成 task。已经进入 engine 的任务设置 `cancelled`，`append()` 会丢弃后续文本；100 ms 的 `Pending` 状态让 SSE provider 在 batch 尚未发布结果时也能检查 `DataSink::is_writable()`。当前公开 LLM API 没有 per-request abort，所以底层 `response()` / `generate()` 仍在当前安全边界自然返回。

### 3.2 single 与 batch 的输出时机不同

`single_request` 把 `Llm::response()` 的文本回调送进 `Utf8StreamProcessor`，每个合法 UTF-8 文本片段立即进入 `Task::chunks`。

`continuous_batch` 和 `dual_pipeline` 则先完成一次 `Llm::generate(batch)`，然后逐请求解码完整 token vector，再追加一个聚合文本 chunk：

```cpp
// apps/mnncli/src/mnncli_server.cpp:310
void runBatch(const std::vector<std::shared_ptr<Task>>& tasks) {
    // ... tokenize active tasks ...
    results = mLlm->generate(input_ids, nullptr, active_tasks.front()->max_tokens);
    // ... validate result count and INTERNAL_ERROR ...
    for (size_t i = 0; i < active_tasks.size(); ++i) {
        Utf8StreamProcessor processor([&decoded](const std::string& text) { decoded += text; });
        // ... decode all returned tokens ...
        active_tasks[i]->append(decoded);
        complete(active_tasks[i], false);
    }
}
```

所以这里的准确表述是：SSE provider 能实时读取已经发布的 chunk；single 模式可以随生成输出，batch/dual 模式目前是 batch 完成后交付，不是逐 token streaming。

### 3.3 SSE 终止协议

`/v1/chat/completions` 先 `submit()`。非流式请求等待 task，返回 OpenAI JSON 和真实 `prompt_tokens/completion_tokens`。SSE 路径先循环 `next_chunk()`，成功完成后才写 `finish_reason=stop` 和 `[DONE]`；连接不可写或写失败时立即取消 task。

关键顺序位于 `apps/mnncli/src/mnncli_server.cpp:631-685`：

```text
submit
  -> next_chunk() 每 100 ms 返回 Pending，检查连接是否仍可写
  -> data: {chat.completion.chunk}
  -> task.wait()
  -> data: {finish_reason: stop}
  -> data: [DONE]
```

## 4. Batch 与 dual request 调度

### 4.1 coordinator 只决定“何时一起进入 engine”

`RequestCoordinator::workerLoop()` 在非 single mode 下等待 2 ms，然后只合并 `max_tokens` 相同的请求，batch 上限为 4。它不推断 lane，也不直接管理 KV；真实 request-to-lane 映射由 `BatchScheduler` 产生。

### 4.2 `scheduleWave()` 只返回同 segment 的互斥 chunk

```cpp
// transformers/llm/engine/src/BatchScheduler.cpp:226
std::vector<std::shared_ptr<BatchScheduler::Chunk>> BatchScheduler::scheduleWave(
    int blockSize, int bs, const std::set<int>& skipReqIds) {
    std::vector<std::shared_ptr<Chunk>> wave;
    auto first = schedule(blockSize, bs, skipReqIds);
    // ... non-dual returns one chunk ...
    const int segmentIndex = first->segmentIndex;
    while (!mPendingChunks.empty() && mPendingChunks.front()->segmentIndex == segmentIndex) {
        wave.push_back(_popPendingChunk());
    }
    // ... trace reqId -> pipelineId ownership ...
    return wave;
}
```

每个 `Chunk` 带 `pipelineId`、`segmentIndex` 和 `reqId`。同一 wave 只排出相同 segment 的 pending chunk，测试覆盖 2/3/4 请求分组、single request wave、split count clamp 和 skip request。

### 4.3 AR batch 分支

`Llm::generate(batch)` 的入口分支在 `transformers/llm/engine/src/llm.cpp:1271-1284`：

```cpp
// transformers/llm/engine/src/llm.cpp:1271
std::vector<std::vector<int>> Llm::generate(
    const std::vector<std::vector<int>>& input_ids, std::ostream* os, int max_new_tokens) {
    if (!mConfig->packed_attention() && !mConfig->dual_pipeline_mode()) {
        // ... reject unsupported batch ...
    }
    if (!mConfig->dual_pipeline_mode() || mConfig->speculative_type() == "eagle") {
        return mGenerationStrategy->generateBatch(input_ids, os, max_new_tokens);
    }
    // ... dual AR wave loop ...
    return output_tokens;
}
```

dual AR 在每轮中做以下工作：

1. `scheduleWave(-1, 4)` 得到最多两条 lane 的互斥 chunk。
2. 计算该 chunk 所需的 QNN compatible bucket。
3. 从 lane-local module pool 取得 `(bucket, all_logits)` 对应 clone。
4. 复制上层 `BatchKVMeta` 的请求条目到 lane-local meta。
5. 启动 graph prefetch wave 和 stage wave。
6. 两条线程分别在独立 `ExecutorScope` 中调用 `Module::onForward()`。
7. join 两条线程，完成 stage/graph wave，再把 KV 条目同步回上层 meta。

当前实现会在第 7 步等待同一 wave 的 lane 全部完成。它不是“lane A 先完成就立即进入下一请求”的异步 admission 方案。

## 5. Eagle3 tree 与 target wave

Eagle packed batch 主入口是 `EagleGeneration::generateBatch()`，位于 `transformers/llm/engine/src/speculative_decoding/eagle_batch.cpp:925`。其主线是：

```text
base model prefill/decode
  -> Eagle draft model 扩展 token tree
  -> `TokenTree::finalize(sampleToken, maxDraftTokens)`
  -> 生成 draft tokens / tree mask / position ids / retrieve paths
  -> target model packed verify
  -> accept path 并更新 scheduler/KV
  -> 下一轮 draft
```

### 5.1 只裁叶子，保持祖先闭包

旧的“全节点按分数排序后截断”可能保留子节点却删掉祖先。当前 `finalize()` 每轮只从选中叶子中删除累计概率最低者，然后按创建顺序重建 mask 和 retrieve path。

```cpp
// transformers/llm/engine/src/speculative_decoding/tokentree.hpp:136
TreeOutputs finalize(int sampleToken, int maxDraftTokens) {
    auto allNodes = getAllNodes();
    while (allNodes.size() > maxDraftTokens) {
        auto leaf = allNodes.end();
        // ... select the lowest-probability retained leaf ...
        allNodes.erase(leaf);
    }
    // ... rebuild verifier inputs, tree mask and retrieve paths ...
    return outputs;
}
```

### 5.2 Eagle dual 的 lane 边界

Eagle target wave 和 AR dual 使用同一个 `DualPipelineScheduler` 接口。`transformers/llm/engine/src/speculative_decoding/eagle_batch.cpp:1189-1273` 构建 per-lane graph requests、启动 prefetch/stage wave、并发 target `onForward()` 并 join；后续 `transformers/llm/engine/src/speculative_decoding/eagle_batch.cpp:1333-1354` 继续用 `scheduleWave()` 维持 request lane 归属。

设备当前没有 Eagle3 QNN 模型，因此这一分支的证据来自主机测试、`spec_eval` 编译和 QNN 导出工具，不来自 Android 三模式表。

## 6. Host/QNN stage grant

`DualPipelineScheduler` 为每个 wave 保存两个 ready queue 和两个 active counter。授予逻辑很小：

```cpp
// transformers/llm/engine/src/DualPipelineScheduler.cpp:527
void DualPipelineScheduler::_grantReadyStagesLocked() {
    if (mActiveHostStages == 0 && !mHostReadyStages.empty()) {
        // ... grant one Host waiter ...
        if (mActiveQnnStages > 0) {
            ++mHostQnnOverlapGrants;
        }
    }
    if (mActiveQnnStages == 0 && !mQnnReadyStages.empty()) {
        // ... grant one QNN waiter ...
        if (mActiveHostStages > 0) {
            ++mHostQnnOverlapGrants;
        }
    }
    mStageCondition.notify_all();
}
```

这给出三个可以直接验收的不变量：

- `maxConcurrentHostStages <= 1`；
- `maxConcurrentQnnStages <= 1`；
- Host 与 QNN 可以同时 active，`hostQnnOverlapGrants` 记录发生次数。

command callback 根据当前 graph snapshot 中导出的 QNN plugin op 名称分类。QNN command 先 `enterGraphStage()` 等自己的 graph resident，再进入 QNN ready queue；Host command 直接进入 Host ready queue。pipeline 只有离开当前 stage 后才能发布下一 stage，所以同 lane 内命令顺序不变。

## 7. QNN bucket、预取与 residency

### 7.1 从 request group 到单 shape binary

`selectQnnCompatibleBucketSize()` 会遍历本轮所有 QNN plugin op，反复提升 padded size，直到每个 op 选择一致；任一 op 没有足够大的 bucket 时返回 `-1`。

构造 graph request 时保留外部选择信息：

```cpp
// transformers/llm/engine/src/DualPipelineGraph.cpp:576
std::vector<DualPipelineScheduler::GraphRequest> buildQnnGraphRequestsForSize(
    const GraphSnapshot& snapshot, int start, int maxK, int reqId, int requestGroupSize) {
    std::vector<DualPipelineScheduler::GraphRequest> requests;
    // ... visit QNN plugin ops ...
    request.graphId = qnnGraphResourceIdForShape(op, shapeIndex);
    request.graphPath = qnnGraphPathForShape(op, shapeIndex);
    request.shapeIndex = shapeIndex;
    request.bucketSize = op.qnn.bucketSizes[shapeIndex];
    request.allGraphName = {op.qnn.allGraphName[shapeIndex]};
    // ...
    return requests;
}
```

带 `allGraphPath` 的新导出模型会把这里的 `shape_index/bucket` 直接写入 trace。当前设备上的旧 combined-binary 模型没有 per-shape path，实测字段因此是 `-1/-1`；设备 bucket 集合 `1/8/128` 来自模型配置与 graph metadata，不能把 `-1` 描述成直接观测到某个 bucket。

### 7.2 执行游标驱动的预取窗口

`beginGraphPrefetchWave()` 只扩展到配置的 lookahead。pipeline 进入 graph `i` 时：

1. 取消被执行顺序跳过的 pending graph。
2. 更新 `currentGraphIndex=i`。
3. 把窗口扩展到 `i + graphPrefetchLookahead`。
4. 只等待 graph `i` 的 `loadFinished`。
5. 获得 QNN stage 后执行，离开时记录 complete。

loader 从可运行任务中按“距当前 cursor 的距离、pipeline id、入队序号”选择任务。未来 graph 不会越过正在等待执行的 current graph 抢占加载机会。

### 7.3 resident graph 与 LRU

graph complete 只减少 `activeUseCount`，不会立即销毁 executor。加载新 graph 达到 `maxResidentGraphs` 时，LRU 只选择 resident、未 pinned、activeUseCount 为 0 的记录；没有安全候选时 resident limit 是软水位，当前活跃图不会被强制释放。

请求结束调用 `releaseRequestGraphs()` 清除 owner metadata。scheduler stop 则对所有 resident graph 发出强制 release，并最终清空 QNN raw pool。

设备上的 bucket-128 combined binary 无法同时保留 30 个 context：旧配置在 graph12/13 返回 QNN validate 1002。最终正确性配置使用 `maxResidentGraphs=2`、`prefetchWindow=0`；16-token HTTP prefill（从 `1/8/128` 可用集合推导为 128 bucket）、dual HTTP 和三请求真实 QNN 路径均通过。这里的结论是“有界 resident 能满足正确性”，不是吞吐最优值。

### 7.4 combined-binary cache miss 也受生命周期约束

新 per-shape 模型优先命中 scheduler resident pool；旧 combined-binary 模型没有 `allGraphPath` 时，同样不能把 fallback executor 永久留在每个 Plugin 实例。当前 cache miss 使用独立 transient alias，执行完成或失败后立即 release：

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:1176
bool compute(CPUKernelContext* ctx) override {
    // ... select combined or per-shape path ...
    if (!findRawGraphExecutor(graphPath, binaryOffset, binarySize, graphNames)) {
        transientGraphId = "transient:" +
            makeRawGraphCacheKey(graphPath, binaryOffset, binarySize, graphNames);
        // ... preload through the shared raw graph pool ...
    }
    auto executor = findRawGraphExecutor(graphPath, binaryOffset, binarySize, graphNames);
    const bool executeSucceeded =
        executor && executor->invokModel(mInputs, mOutputs, executorShapeIndex);
    if (!executeSucceeded) {
        // ... release transient alias ...
        return false;
    }
    // ... copy output ...
    if (!transientGraphId.empty()) {
        releaseRawGraphInternal(transientGraphId, false, false);
    }
    return true;
}
```

这条路径解决 continuous 模式按 30 个 Plugin 累积大 bucket QNN context 的问题；dual 模式正常命中 resident graph 时不创建 transient alias。

`QNNPerf` 也把 provider 的 performance infrastructure 当作可选能力。`deviceGetInfrastructure`、power-config 函数或创建调用不可用时，perf 调整退化为 no-op，graph 正确性路径不会再通过空函数指针崩溃，见 `source/backend/qnn/backend/QNNPerf.cpp:14`。

### 7.5 导出到执行的接口链

```text
`llmexport.py`
  W4 / channel-wise 默认
    -> `generate_llm_qnn.py`
       固定 bucket `1/32/256/512`，默认 Host attention backend 为 OpenCL
       -> `generateLlmIO`
       -> `compilefornpu`
          跳过实际 `OpType_Attention` 名称
          写 `allGraphPath` + `separate_graphs`
          -> `npu_convert.py`
             每 shape 生成独立 QNN context binary
             -> runtime `DualPipelineGraph`
                选 path / shape / bucket
                -> `PluginExecuteRaw::compute()`
                   resident lookup -> transient fallback -> QNN invoke -> copy output/release
```

`PluginExecuteRaw` 对单 shape binary 把内部 `executorShapeIndex` 设为 0。外部原始 `shapeIndex` 仍用于选择 `allGraphPath`，两者不能混为一个索引。

## 8. 取消、失败与回收

| 场景 | 当前行为 | 回收边界 |
|---|---|---|
| queued HTTP task 取消 | 立即从 `mPending` 删除 | task 完成，engine 不会看到请求 |
| executing single 取消 | 立即停止发布文本 | 当前 `Llm::response()` 返回后 coordinator idle |
| executing batch/dual 取消 | provider 100 ms 检查断开，该 task 不再解码/append | 整个 `generate(batch)` 返回后完成 batch |
| module 输出失败 | 取消 stage wave 和 graph wave | join worker 后统一 finish/cleanup |
| QNN graph load 失败 | resident=false，唤醒等待者并使 wave 失败 | active/pin 状态清理 |
| graph 跳过 | 删除 pending load，登记 complete | wave 尾部清理未执行窗口 |
| scheduler stop | join loader，强制 release resident graph | QNN raw pool 最终清空 |

这里最重要的边界是“服务取消可见性”与“engine 抢占”不同。当前实现完成了前者，没有添加不存在的 per-request KV rollback 或 kernel abort API。

## 9. 验收 trace

trace 默认关闭，仅当 `MNN_ACCEPTANCE_TRACE` 是非空且不是 `0` 时写 stderr，入口为 `transformers/llm/engine/include/llm/AcceptanceTrace.hpp:16-47`。

| event | 关键字段 |
|---|---|
| `request_enqueued/started/cancelled/completed` | request id、scope、状态、TTFT、E2E、token、tok/s、bytes |
| `batch_started/batch_member` | mode、batch size、HTTP request 到 batch index |
| `lane_owner/lane_completed` | engine request id、lane、segment、完成时间 |
| `qnn_bucket` | lane、request id、graph index、shape index、bucket |
| `stage_summary` | Host/QNN 完成数、overlap、max Host、max QNN、cancelled |

trace 不参与调度决策，不开启时只保留一次环境变量判断后的快速返回。

## 10. Android 三模式实测

设备是 arm64/V79，模型为 `/data/local/tmp/qnn_fixed_s1_8_128/` 的 AR QNN 产物。最终矩阵在独立 GSM8K 100 题任务自然结束后执行；每个模式只生成 1-4 token，取消用例客户端在 1 秒主动断开，没有再跑 100 题。计时用于说明请求确实完成，不作为并发性能结论。

| 模式 | readiness | JSON | SSE / `[DONE]` | 并发 | 取消后健康 | TTFT / E2E / tok/s | 结果 |
|---|---|---|---|---|---|---|---|
| `single_request` | `/v1/models` 通过 | HTTP 200，OpenAI JSON，2 token | content + finish + `[DONE]` | 3 请求严格串行 | executing cancel，后续 1-token 请求通过 | 2.681 s / 7.384 s / 0.271 | 通过 |
| `continuous_batch` | 通过 | HTTP 200，2 token | 通过；batch 完成后交付 | trace 出现 2 请求同 batch，其余安全排队 | executing cancel，输出 0 byte，健康通过 | 5.024 s / 5.024 s / 0.398 | 通过 |
| `dual_pipeline` | 通过 | HTTP 200，2 token | 通过；batch 完成后交付 | 2 请求分别归属 lane 0/1，Host/QNN 各自不超过 1 | executing cancel，输出 0 byte，健康通过 | 10.019 s / 10.019 s / 0.200 | 通过 |

三种基础响应的 `usage` 都是 `prompt_tokens=16`、`completion_tokens=2`、`total_tokens=18`。single 的并发 trace 中前一个请求完成后下一个才开始；continuous/dual 均出现 `request_count=2` 的真实合批。三个模式的取消 trace 都是 `request_cancelled state=executing`，随后 `request_completed status=cancelled` 且 `response_bytes=0`。

dual 额外使用同一静态 Android 构建运行：

```bash
# 设备端运行；V79 skeleton 需要 ADSP_LIBRARY_PATH
LD_LIBRARY_PATH=/data/local/tmp/mnn-qnn \
ADSP_LIBRARY_PATH=/data/local/tmp/mnn-qnn \
MNN_ACCEPTANCE_TRACE=1 \
./llm_demo_static dual_pipeline_resident2.json --dual-batch-test
```

该命令退出 0，三个 `hello` 请求都生成非空文本，lane 0 持有请求 0/1、lane 1 持有请求 2。30 个真实 QNN graph 多轮执行，典型 `stage_summary` 为 `host_completed=58`、`qnn_completed=60`、`overlap_grants=30/31`、`max_host=1`、`max_qnn=1`、`cancelled=0`。配置使用 resident=2/prefetch=0；设备模型仅提供 AR，因此本表不包含 Eagle3 Android 结论。

attached 模型是 combined-binary 旧格式，`qnn_bucket` trace 为 `shape_index=-1 bucket=-1`。`1/8/128` 是模型 metadata 的实际 bucket 集合；长 prompt 需要的 128 bucket 是从 packed length 和可用 bucket 推导，不伪装成 trace 直接观测值。

磁盘门禁：

| 项目 | 验收前 | 验收后 |
|---|---:|---:|
| `/data` 可用空间 | 118 GiB | 118 GiB |
| model 目录 | 3.2 GiB | 3.2 GiB，未修改 |
| QNN runtime 目录 | 4.4 GiB | 4.4 GiB，未修改 |
| acceptance staging | 0 | 13 MiB（`mnncli`、静态 demo、三配置和摘要），低于 256 MiB |

## 11. 验证证据

当前主机与构建门禁：

| 检查 | 结果 |
|---|---|
| focused `llm` + `run_test.out llm` | 38/38 通过 |
| `run_test.out op/packed_attention` | 1/1 通过 |
| trace stage test | `host=1, qnn=1, overlap=1, max_host=1, max_qnn=1` |
| Android service build | `apps/mnncli/build.sh --android-service`，NDK r29，arm64，Release，8 jobs，成功 |
| deployable `mnncli` | stripped 6.6 MiB PIE |
| 动态依赖 | 仅 `libdl.so`、`libm.so`、`libc.so` |
| Android 三模式 HTTP | JSON、SSE、并发、执行中取消、健康全部通过 |
| Android dual QNN demo | 静态 6.0 MiB demo，3 请求/2 lane/30 graph，退出 0 |
| Python exporter | `py_compile` 与 `--help` 通过 |
| `compilefornpu.cpp` | translation-unit compile 通过 |
| `git diff --check HEAD` | 通过 |

## 12. 逐文件修改版本表

下面每一行都以 `f7d89034` 为基线，`worktree` 为当前版本。

### 12.1 `apps/mnncli` 服务与构建

| 文件 | 基线 | 当前修改 | 接口变化 | 调用者 | 行为影响 / 证据 |
|---|---|---|---|---|---|
| `apps/mnncli/CMakeLists.txt` | 完整 CLI，TLS/video 固定进入构建 | 增加 TLS、video、service-only 开关 | `MNNCLI_ENABLE_TLS`、`MNNCLI_SERVICE_ONLY` | `build.sh` | Android 明文服务不依赖 OpenSSL/libyuv；service build 通过 |
| `apps/mnncli/build.sh` | host 两阶段构建、并行度未受验收约束 | 增加 NDK r29 arm64 service-only 两阶段构建与 strip | `--android-service`、`--clean`、`--check` | 开发/验收命令 | 生成 6.6 MiB 静态链接 app，jobs<=8 |
| `apps/mnncli/include/llm_manager.hpp` | `CreateLLM(config,use_template)` | 接收可选 scheduler override | `scheduler_mode` 参数 | serve/run/benchmark handler | serve 可在 load 前覆盖 JSON |
| `apps/mnncli/include/mnncli_server.hpp` | server 自己持有响应互斥状态 | forward-declare coordinator，Start 接收 mode | `RequestCoordinator` PImpl、`Start(...,mode)` | serve handler | HTTP 所有权集中到 coordinator |
| `apps/mnncli/src/cli_command_spec.cpp` | serve 无 mode 选项 | 注册三模式选项 | `--scheduler-mode` | command parser | CLI surface 可复现实验模式 |
| `apps/mnncli/src/handlers/serve_command_handler.cpp` | 直接 load/start | 校验 mode、传 override、报告 load 失败 | `CreateLLM(...,mode)` | `mnncli serve` | CLI 优先级与错误反馈可验证 |
| `apps/mnncli/src/llm_manager.cpp` | runtime config 后直接 load | 在 load 前 merge mode，传播失败 | startup override | serve handler | 非法/缺 packed 配置启动失败 |
| `apps/mnncli/src/mnncli.cpp` | 完整 CLI 注册 | service-only 时仅注册本地 serve 依赖，更新 help | compile-time service surface | app main | 默认 CLI 不变；Android service 可独立链接 |
| `apps/mnncli/src/mnncli_server.cpp` | handler 直接串行调用 LLM，SSE 等完整结果 | coordinator queue、实时 chunk、batch、100 ms 连接轮询、取消、usage、trace | `submit/cancel/reset`、三态 Task chunk API | HTTP routes | 三模式 JSON/SSE/并发/执行中取消均在 Android 通过 |

### 12.2 Runtime、QNN 与导出

| 文件 | 基线 | 当前修改 | 接口变化 | 调用者 | 行为影响 / 证据 |
|---|---|---|---|---|---|
| `source/backend/cpu/CPUPackedAttention.cpp` | packed attention 基础实现 | 修正 Eagle/packed 输入边界 | 内部 shape/length 处理 | CPU backend | PackedAttention tests 覆盖 |
| `source/backend/qnn/backend/QNNBackend.cpp` | raw graph plugin 基础执行 | per-shape path、resident pool、预取/释放、cache miss transient executor | `preloadRawGraph/releaseRawGraph/releaseAllRawGraphs` | LLM graph callbacks、Plugin | continuous 不再每 Plugin 永久保留 context；三模式与 dual QNN demo 通过 |
| `source/backend/qnn/backend/QNNPerf.hpp` | perf 对象无有效性状态 | 增加 power-config 初始化值与 valid gate | 内部状态 | `QNNPerf.cpp` | 可选 provider 能力安全降级 |
| `source/backend/qnn/backend/QNNPerf.cpp` | 无条件调用 perf 函数指针 | 校验 infrastructure/function/status，invalid 时 no-op | 构造/方法行为 | raw executor、QNN backend | 低 resident probe 不再空指针崩溃；Android QNN 路径通过 |
| `source/backend/qnn/npu_convert.py` | 合并 graph 到一个 context | 可按 graph 生成独立 binary | `separate_graphs` post config | `generate_llm_qnn.py` | 为 per-shape runtime path 提供产物 |
| `tools/cpp/compilefornpu.cpp` | QNN plugin 写单 path | 写 `allGraphPath`，跳过实际 Attention op，启用 separate graph | Plugin metadata/post config | QNN converter | Attention 留 Host/OpenCL，FFN graph 可分 bucket |
| `transformers/llm/engine/include/llm/DualPipelineGraph.hpp` | graph snapshot 基础字段 | 增加 per-shape path/bucket metadata helper | `bucketSizes/graphPaths` | Llm、tests | runtime 可选择单 shape binary |
| `transformers/llm/engine/include/llm/DualPipelineScheduler.hpp` | graph/stage API | GraphRequest 增加 shape/bucket，snapshot 增加 max Host | struct 字段 | graph builder、trace、tests | trace 可报告真实 bucket 与 Host 不变量 |
| `transformers/llm/engine/include/llm/llm.hpp` | legacy dual 配置入口 | 公开 scheduler mode，增加启动锁与 dual helpers | `scheduler_mode()` | mnncli、generation | 模式在 load 后稳定 |
| `transformers/llm/engine/include/llm/AcceptanceTrace.hpp` | 不存在 | 新增默认关闭 trace helper | `enabled/nowMicros/log` | service/scheduler/LLM | 无独立 metrics subsystem；环境开关测试通过 |
| `transformers/llm/engine/src/BatchScheduler.cpp` | dual split/schedule 基础 | trace lane owner，skip-aware wave | `scheduleWave(...,skipReqIds)` | AR/Eagle batch | 真实请求所有权可验收 |
| `transformers/llm/engine/src/DualPipelineGraph.cpp` | snapshot 与 QNN request 基础 | 合并 execution/model order，选 compatible bucket/path | graph request builders | Llm/Eagle | per-lane bucket 选择与 oversized 拒绝测试 |
| `transformers/llm/engine/src/DualPipelineScheduler.cpp` | cursor/stage/resident 调度 | trace bucket/stage/lane，Host max counter | snapshot/trace fields | runtime callbacks/tests | Host/QNN<=1 与 overlap 可观测 |
| `transformers/llm/engine/src/llm.cpp` | dual AR runtime | 三模式校验、lane resources、bucket module、KV 同步、trace | `scheduler_mode()`、dual wave helpers | mnncli/Generation | AR dual 真实执行主线；38 tests |
| `transformers/llm/engine/src/llmconfig.hpp` | legacy dual bool | canonical scheduler mode + packed requirement | config accessors | `Llm::load()`/server | JSON/legacy/invalid tests |
| `transformers/llm/engine/src/speculative_decoding/eagle.cpp` | single Eagle | packed batch所需的 draft/hidden/KV 支持 | Eagle internal methods | Eagle generation | Eagle3 batch 编译路径 |
| `transformers/llm/engine/src/speculative_decoding/eagle_batch.cpp` | Eagle packed batch 基础 | dual target wave、lane KV/module、draft/base 协同 | `EagleGeneration::generateBatch()` | `Llm::generate(batch)` | Host unit/compile evidence，非 Android claim |
| `transformers/llm/engine/src/speculative_decoding/generate.cpp` | continuous AR 使用原始 packed length | embedding、position id、module key 按 QNN compatible bucket padding | `ArGeneration::generateBatch()` 内部 shape | continuous batch | attached `1/8/128` 模型的 continuous HTTP 通过 |
| `transformers/llm/engine/src/speculative_decoding/generate.hpp` | generation strategy 接口 | 扩展 batch/Eagle data structures 和 override | `generateBatch()` | Llm、AR/Eagle | 三模式共用 batch dispatch |
| `transformers/llm/engine/src/speculative_decoding/tokentree.hpp` | 全节点排序截断 | leaf-only prune 后重建 mask/path | `finalize()` 行为 | Eagle batch | TokenTree regression 通过 |
| `transformers/llm/engine/tools/generateLlmIO.cpp` | 单一模型/shape IO 生成 | target/eagle/eagle_fc 组件与多 bucket IO | CLI component/buckets | QNN export script | `1/32/256/512` export 输入 |
| `transformers/llm/export/llmexport.py` | W4 block 64、lm head 策略不同 | W4 channel-wise，lm head 默认 16 bit | exporter defaults | 模型导出命令 | Python syntax/help 通过 |
| `transformers/llm/export/npu/generate_llm_qnn.py` | 单组件/旧默认流程 | 三组件导出、固定 bucket、OpenCL runtime config | component export CLI | QNN offline workflow | py_compile/help 通过；未宣称设备 Eagle |

### 12.3 Demo 与测试

| 文件 | 基线 | 当前修改 | 接口变化 | 调用者 | 行为影响 / 证据 |
|---|---|---|---|---|---|
| `transformers/llm/engine/CMakeLists.txt` | llm/demo targets | 注册 `spec_eval` 等新源 | build target | CMake | focused build 成功 |
| `transformers/llm/engine/demo/llm_demo.cpp` | 普通/已有 batch demo | 增加 dual batch/throughput 入口；正确性模式保留配置 resident 并执行 tuning | `--dual-batch-test` 等 | Android acceptance | resident=2 三请求真实 QNN test 退出 0 |
| `transformers/llm/engine/demo/spec_eval.cpp` | 不存在 | Eagle3 speculative evaluation demo | `spec_eval` CLI | host evaluation | target 编译通过 |
| `test/llm/BatchSchedulerTest.cpp` | 基础 scheduler tests | 增加 dual 分组、wave、lane 稳定和 skip tests | test cases | `run_test.out llm` | 38/38 suite 的一部分 |
| `test/llm/DualPipelineGraphTest.cpp` | graph metadata tests | per-shape path、bucket/shape、order merge tests | test cases | `run_test.out llm` | bucket 选择与 metadata 通过 |
| `test/llm/DualPipelineStageSchedulerTest.cpp` | stage overlap tests | 增加 Host max invariant | snapshot assertion | `run_test.out llm` | trace focus 为 max_host=1/max_qnn=1 |
| `test/llm/SchedulerModeConfigTest.cpp` | 不存在 | legacy/explicit/invalid mode tests | 3 test cases | `run_test.out llm` | config 分支通过 |
| `test/llm/TokenTreeTest.cpp` | 不存在 | 祖先闭包/叶子裁剪回归 | test case | `run_test.out llm` | leaf prune 通过 |
| `test/op/PackedAttentionTest.cpp` | packed attention 基础 case | 增加 Eagle/batch shape 与边界 case | op tests | `run_test.out` | focused op evidence |

### 12.4 文档

| 文件 | 基线 | 当前修改 | 接口变化 | 调用者 | 行为影响 / 证据 |
|---|---|---|---|---|---|
| `dual_pipeline_code_review.md` | 历史追加式 dual review | 重写为三模式到 QNN 的单主线 | 统一调用图、版本表、实测表 | reviewer | Android 结果、实现边界、anchor/fence/diff 检查 |

## 13. 当前实现与目标方案的差异

1. batch/dual SSE 是 batch 完成后交付，不是逐 token streaming。
2. SSE 每 100 ms 检查断开并停止输出，但 in-flight HTTP 取消不会抢占正在执行的 LLM batch/kernel。
3. dual lane 在一个 wave 内并发，下一 wave 仍等待两 lane join；没有即时 lane-completion admission 或请求迁移。
4. coordinator 的 continuous batch 是固定 2 ms 小窗口和最大 4 请求，不是长期驻留、逐 token admission 的 continuous batching engine。
5. QNN stage 串行，Host stage 也限制为 1；收益来自 Host/QNN 跨 lane overlap，不是两个 QNN graph 同时执行。
6. 新导出默认目标是 `1/32/256/512`，本轮 Android 模型实际只有 `1/8/128`。
7. attached combined-binary 模型的 trace 没有 shape/path metadata，bucket 字段是 `-1/-1`；新 per-shape 导出才可直接 trace bucket。
8. Android 验收覆盖 AR。Eagle3 当前只有主机测试、编译和导出证据。

## 小结

- 服务入口是 `ServeCommandHandler::Handle()`，CLI mode 在 `Llm::load()` 前覆盖 JSON，并在加载后锁定。
- `RequestCoordinator` 拥有 HTTP queue、SSE、100 ms 断开检测、取消和一个 LLM worker；batch mode 当前按 batch 完成后发布结果。
- `BatchScheduler::scheduleWave()` 提供互斥 lane chunk，AR/Eagle generation 负责 lane-local Module、Runtime 和 KV 同步。
- `DualPipelineScheduler` 把 graph window 与 Host/QNN stage 分开管理，保证 Host/QNN 各自最多一个并允许二者 overlap。
- QNN 链路从 per-shape export metadata 走到 resident raw executor；combined cache miss 使用 transient executor，设备 bucket 必须按现有 `1/8/128` 产物报告。
- 下一步阅读入口是 `apps/mnncli/src/mnncli_server.cpp:18`、`transformers/llm/engine/src/llm.cpp:1271`、`transformers/llm/engine/src/speculative_decoding/eagle_batch.cpp:925` 和 `transformers/llm/engine/src/DualPipelineScheduler.cpp:223`。
