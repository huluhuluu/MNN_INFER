# `mnncli serve` 三模式逻辑主线 Code Review

本文只跟踪当前 worktree 中一条聊天请求从 `mnncli serve` 到 HTTP/SSE 结束的真实代码路径。重点文件是 `apps/mnncli/src/mnncli_server.cpp`、`transformers/llm/engine/src/llm.cpp`、`transformers/llm/engine/src/BatchScheduler.cpp` 和 `transformers/llm/engine/src/DualPipelineScheduler.cpp`。

审查时先看第 1 节主线图，再按第 2-7 节回到具体状态和源码。这里描述的是当前实现，不把未来的动态 admission、逐 token batch streaming 或纯 QNN 并行当作已实现能力。

## 1. 一页逻辑主线

```text
`mnncli serve --config ... [--scheduler-mode ...]`
    |
    v
`ServeCommandHandler::Handle`
    |  解析 CLI；CLI mode 覆盖 JSON；`CreateLLM()` 在 `load()` 前写入 runtime config
    v
`Llm::load()`
    |  校验 mode；continuous/dual 必须 packed attention；加载完成后锁定 mode
    v
`MnncliServer::Start` -> 一个 `RequestCoordinator` + 一个 worker thread + 一个 `Llm`
    |
    v
`POST /v1/chat/completions`
    |  解析 messages / stream / max_tokens，创建 `Task`
    v
`RequestCoordinator::submit` -> `mPending`
    |
    +-- `single_request` -----------------------------------------------+
    |     只取一个 Task -> `runSingle()`                                |
    |     template/tokenize -> `generate_init(stream)` -> `generate()`  |
    |     token text -> `Task::append()` -> `Task::chunks`              |
    |                                                                    |
    +-- `continuous_batch` / `dual_pipeline` --------------------------+
          2 ms 收集窗口，最多 4 个且 `max_tokens` 相同的 Task
          -> `runBatch()` -> `generate_init(nullptr)`
          -> `Llm::generate(vector<input_ids>)`
          -> 一次完整 batch generation 返回 `vector<vector<int>>`
          -> 每个 result decode 后一次 `Task::append()`
    |
    v
`stream=false`                         `stream=true`
    |                                      |
    | `Task::wait()`                       | 100 ms poll `Task::next_chunk()`
    | OpenAI JSON + usage                  | `data: content` -> finish -> `[DONE]`
    v                                      v
HTTP response                           SSE response

`Llm::generate(vector)` 内部的 mode 分支
    |
    +-- 非 dual：`Generation::generateBatch()`
    |       AR: `BatchScheduler::schedule()` -> packed forward -> sample/update/release
    |       Eagle: `EagleGeneration::generateBatch()`，同样要求 packed attention
    |
    +-- dual AR：`BatchScheduler::scheduleWave()`
            -> 同 segment 的 chunk 组成 wave，并记录 request -> lane
            -> 每 lane 取独立 `executor` / `runtimeManager` / `BatchKVMeta` / module
            -> `beginGraphPrefetchWave()` (有 QNN Plugin 时准备 graph；纯 Host 时为空)
            -> `beginStageWave()`
            -> 两个 lane thread 执行各自 `Module::onForward()`
                 before callback: Host -> `enterStage(Host)`
                                  QNN  -> `enterGraphStage(QNN)` -> `enterStage(QNN)`
                 after callback: 对称 leave
            -> join 所有 lane -> `finishStageWave()` + `finishGraphPrefetchWave()`
            -> sample/update request/KV/release -> 下一 wave
```

这张图的两个关键分界是：

- HTTP 的 batch 是 coordinator 先收集一小组请求，再调用一次完整 batch API；它不是 decode 中途持续插入新请求。
- dual 的两个 lane 有独立 executor/module 资源，但 Host stage 最多同时一个、QNN stage 最多同时一个；lane 并发不等于同类 backend 并发。

## 2. 关键对象与所有权

| 层次 | 对象 | 拥有的状态 | 审查时应确认的边界 |
|---|---|---|---|
| 启动 | `ServeCommandHandler` | config path、CLI mode、host/port | CLI mode 在 `Llm::load()` 前生效 |
| 服务 | `RequestCoordinator` | `mPending`、`mExecutingTasks`、唯一 worker | 只有 worker 调用同一个 `Llm` |
| 请求 | `Task` | prompts、cancel flag、response、chunk queue、metrics | `append()` 在 cancel 后不能再写数据 |
| LLM | `Llm` | `LlmContext`、generation strategy、batch scheduler | mode load 后不可重配；取消是原子请求旗标 |
| batch | `BatchScheduler` | request state、KV、segment、lane chunk | dual wave 只合并同一 `segmentIndex` 的 pending chunk |
| dual stage | `DualPipelineScheduler` | graph window、Host/QNN ready queue、active stages | Host<=1、QNN<=1；不同类可以 overlap |
| SSE | `DataSink` | socket writability、SSE frames | 写失败立即向 coordinator 取消 Task |

## 3. 启动时如何确定 mode

入口 `apps/mnncli/src/handlers/serve_command_handler.cpp:34` 只接受 `single_request`、`continuous_batch`、`dual_pipeline` 三个 CLI 值，并把可选覆盖值传给 `LLMManager::CreateLLM()`。

```cpp
// apps/mnncli/src/llm_manager.cpp:14
std::unique_ptr<MNN::Transformer::Llm> LLMManager::CreateLLM(
    const std::string& config_path,
    bool use_template,
    const std::string& scheduler_mode
) {
    std::unique_ptr<MNN::Transformer::Llm> llm(MNN::Transformer::Llm::createLLM(config_path));
    std::string runtime_config = use_template ? "{\"tmp_path\":\"tmp\"}" :
                                                "{\"tmp_path\":\"tmp\",\"use_template\":false}";
    if (!scheduler_mode.empty()) {
        runtime_config = use_template ?
            "{\"tmp_path\":\"tmp\",\"scheduler_mode\":\"" + scheduler_mode + "\"}" :
            "{\"tmp_path\":\"tmp\",\"use_template\":false,\"scheduler_mode\":\"" + scheduler_mode + "\"}";
    }
    llm->set_config(runtime_config);
    {
        AUTOTIME;
        if (!llm->load()) {
            return nullptr;
        }
    }
    // ...
    {
        AUTOTIME;
        TuningPrepare(llm.get());
    }
    return llm;
}
```

`transformers/llm/engine/src/llmconfig.hpp:341` 的 canonical `scheduler_mode` 优先于 legacy `dual_pipeline_mode`/`dual_pipeline`。`transformers/llm/engine/src/llm.cpp:623` 校验 mode 合法性，并在 batch/dual 未开启 packed attention 时直接拒绝 load；`transformers/llm/engine/src/llm.cpp:771` 之后把 `mSchedulerModeLocked` 设为 true。

审查结论：mode 是启动时策略，不是每个 HTTP 请求可切换的策略。运行中再通过 `set_config()` 写 mode 应被锁定逻辑拒绝，避免 coordinator mode 与 LLM generation mode 分离。

## 4. HTTP、Task 与 coordinator

### 4.1 `Task` 是响应所有权边界

`apps/mnncli/src/mnncli_server.cpp:21` 中每个 `Task` 同时保存请求数据、输出队列和完成状态。非流式路径等待 `Task::wait()`，流式路径重复读取 `Task::next_chunk()`。

```cpp
// apps/mnncli/src/mnncli_server.cpp:67
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

这里的 100 ms timeout 不是生成轮询；它让 SSE provider 在 batch generation 仍在运行时重新检查 `DataSink::is_writable()`，从而尽早发现客户端断开。

### 4.2 coordinator 的批边界

`apps/mnncli/src/mnncli_server.cpp:412` 的 `workerLoop()` 是唯一消费 `mPending` 的线程：

- `single_request` 每轮只取第一个 Task，调用 `runSingle()`。
- 另两个 mode 先等待最多 2 ms，再取最多 4 个、并且 `max_tokens` 与第一个 Task 相同的请求。
- worker 在调用 `runSingle()`/`runBatch()` 前写入 `mExecutingTasks`；退出后才清空。

所以 `continuous_batch` 在服务层的含义是固定入队窗口的 batch。已经开始的 `runBatch()` 不会再吸收新 Task；新 Task 只能等待下一次 workerLoop。

### 4.3 single 和 batch 的输出时序不同

`apps/mnncli/src/mnncli_server.cpp:289` 的 `runSingle()` 使用 `LlmStreamBuffer`，回调会把 UTF-8 片段交给 `Task::append()`。因此单请求的 `chunks` 可以在 `Llm::generate()` 返回前被 SSE provider 读取。

`apps/mnncli/src/mnncli_server.cpp:333` 的 `runBatch()` 则先调用一次 `mLlm->generate(input_ids, nullptr, max_tokens)`，仅在它返回后才把每个 token result decode 并 `append()`。这解释了 batch/dual 的 SSE transport 虽然可用，但内容目前按 batch completion 出现。

### 4.4 HTTP/SSE 的结束条件

`apps/mnncli/src/mnncli_server.cpp:604` 创建 Task。`stream=false` 在 `Task::wait()` 成功后构造 OpenAI JSON；`apps/mnncli/src/mnncli_server.cpp:666` 的 chunk provider 在所有 chunk 消耗完且 `Task::wait()` 成功后才写 finish frame 和 `[DONE]`。

这要求审查时区分三种结束：

| Task 结果 | 非流式 | SSE |
|---|---|---|
| 成功 | HTTP 200 + response/usage | content chunks -> finish -> `[DONE]` |
| failed | HTTP 500 | provider 结束，不写正常 finish |
| cancelled | HTTP 500 或 provider 结束 | provider 结束，不再写新内容 |

## 5. `Llm::generate(vector)` 的三条执行支路

`transformers/llm/engine/src/llm.cpp:1283` 是 coordinator batch API 的 LLM 落点：

1. 未开启 packed attention 且不是 dual 时，逐 input 调用单请求 generate。这是兼容分支；正常服务的 continuous/dual 已在 load 时被要求 packed attention。
2. 非 dual，或 speculative type 为 `eagle` 时，交给当前 `GenerationStrategy::generateBatch()`。
3. dual AR 时，Llm 自己建立 request scheduler、lane wave、图预取与 stage wave。

`transformers/llm/engine/src/speculative_decoding/generate.cpp:48` 的 AR batch 主线是：`addRequest()` -> `schedule()` -> packed embedding/mask/position -> module forward -> sample -> `BatchScheduler::update()` -> 已完成请求释放 KV。`transformers/llm/engine/src/speculative_decoding/eagle_batch.cpp:925` 的 Eagle batch 同样首先检查 packed attention；dual Eagle 在同文件的 `1208`、`1233`、`1270` 使用相同的 graph/stage wave 协议。

这里不要把 AR 与 Eagle 的采样细节混为一谈。服务层只依赖它们共同的 batch API 返回 `vector<vector<int>>`；具体 speculative tree 的维护留在 generation strategy 内部。

## 6. dual AR：request 到 lane wave

`transformers/llm/engine/src/BatchScheduler.cpp:226` 的 `scheduleWave()` 先生成第一个 chunk；只有 `mDualPipelineMode` 为真时才继续从 `mPendingChunks` 取出相同 `segmentIndex` 的 chunk。随后对每个 `chunk->reqId` 写 `event=lane_owner`。

这给出 request/lane 的审查边界：

```text
BatchScheduler request state
    -> scheduleWave(segment S)
        -> chunk 0: pipelineId 0, reqId group A
        -> chunk 1: pipelineId 1, reqId group B
    -> one dual wave

下一 segment 或尚未入队的 request
    -> 不进入本 wave
```

`transformers/llm/engine/src/llm.cpp:1329` 消费这个 wave。每个 `DualWaveTask` 绑定一个 chunk，并取对应 lane 的 `BatchKVMeta`、module 和 executor。`transformers/llm/engine/src/llm.cpp:320` 在初始化时为每个 pipeline 建独立 executor/runtime manager；`transformers/llm/engine/src/llm.cpp:385` 再以 `(paddedCulLen, all_logits)` 为 key 维护该 lane 的 module pool。

这意味着双 lane 不共享 `Module::onForward()` 的对象。共享的是上层 scheduler/control state 和 stage gate，而不是 module runtime 本身。

## 7. dual stage：graph prefetch、Host/QNN gate 与 join

### 7.1 graph 预取是可选分支

`transformers/llm/engine/src/llm.cpp:455` 从 execution/model snapshot 建立 `mDualPipelineQnnGraphRequests` 和 op-name 到 graph-index 的映射。`transformers/llm/engine/src/llm.cpp:1406` 为每个 lane 构造 `PipelineGraphWave` 并调用 `beginGraphPrefetchWave()`。

有 QNN Plugin 时，`transformers/llm/engine/src/DualPipelineScheduler.cpp:183` 登记每条 lane 的 graph 窗口并唤醒 worker；`enterGraphStage()` 等待 graph resident 后再请求 QNN stage。纯 Host 模型的 graph request 为空，`beginGraphPrefetchWave()` 返回成功但不会产生 QNN load；这与设备实测的 `qnn_completed=0` 一致。

### 7.2 stage gate 按 op 回调，而不是按整次 forward

`transformers/llm/engine/src/llm.cpp:350` 通过每 lane executor callback 连接 scheduler：

- 当前 `OperatorInfo::name()` 在 `activeQnnOpIndices` 中，before callback 调用 `enterGraphStage()`，最终进入 `STAGE_QNN`。
- 否则 before callback 调用 `enterStage(pipelineId, STAGE_HOST)`。
- after callback 对称调用 `leaveGraphStage()` 或 `leaveStage()`。

`transformers/llm/engine/src/DualPipelineScheduler.cpp:514` 的 `_grantReadyStagesLocked()` 只在 `mActiveHostStages == 0` 时 grant 一个 Host waiter，只在 `mActiveQnnStages == 0` 时 grant 一个 QNN waiter。两个条件独立，因此允许一个 Host 和一个 QNN 同时存在，并记录 `mHostQnnOverlapGrants`。

```text
lane 0 op callback                 lane 1 op callback
      |                                   |
      +-- Host ready queue -----------+   +-- QNN ready queue -----------+
                                      |                                  |
                           Host active == 0?                   QNN active == 0?
                                      |                                  |
                                  grant one                          grant one
                                      \                                  /
                                       \---- 可跨 backend overlap --------/

同一 backend class：最多一个 active stage
```

`transformers/llm/engine/src/DualPipelineScheduler.cpp:473` 只有在 ready queue 和 active map 都为空时才成功结束 stage wave，并在同一位置输出 `stage_summary`。因此 trace 中的 `max_host`、`max_qnn` 是 stage callback 层的并发上界，不是线程数或 lane 数。

### 7.3 当前 wave 仍是 join barrier

`transformers/llm/engine/src/llm.cpp:1450` 为每个 lane 创建 worker thread，但 `transformers/llm/engine/src/llm.cpp:1466` 会 join 所有 worker，随后才读取 snapshot、finish stage/prefetch wave，并进入采样与下一次 `scheduleWave()`。

所以当前实现具有两个同时成立的事实：

- lane-local executor/module 可以并行推进到各自的 op callback；
- 不会因为一个 lane 先完成整次 forward 就立即接纳新的 request 或启动下一 segment。

这是 review 时最重要的性能/语义边界之一。

## 8. 双流水线调度如何形成 lane wave

这一层先决定谁属于同一个 forward wave，还没有开始争用 Host 或 QNN。入口是 `transformers/llm/engine/src/BatchScheduler.cpp:226` 的 `scheduleWave()`：先调用正常的 `schedule()` 取得第一个 chunk；只有 dual mode 才继续从 `mPendingChunks` 取出与第一个 chunk 同一 `segmentIndex` 的 chunk。

```cpp
// transformers/llm/engine/src/BatchScheduler.cpp:226
std::vector<std::shared_ptr<BatchScheduler::Chunk>> BatchScheduler::scheduleWave(
    int blockSize, int bs, const std::set<int>& skipReqIds) {
    std::vector<std::shared_ptr<Chunk>> wave;
    auto first = schedule(blockSize, bs, skipReqIds);
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
    // 为每个 `chunk->reqId` 写 `lane_owner` trace。
    // ...
    return wave;
}
```

因此拆分单位是 scheduler 已经构造好的 request group，不是把同一 HTTP request 复制给两个 lane。一个典型 wave 可以这样读：

```text
pending request state
    |
    +-> `schedule()` produces chunk A: `pipelineId=0, segmentIndex=S, reqId={r1,r2}`
    |
    +-> same-S pending chunk B: `pipelineId=1, reqId={r3}`
    v
`scheduleWave()` -> {A, B} -> one dual forward wave

next segment or later HTTP Task
    -> cannot join the active wave
```

`transformers/llm/engine/src/llm.cpp:1329` 消费这个 vector。它为每个 chunk 计算 QNN bucket-compatible 的 `paddedCulLen`，构造 embedding、mask、position 和 lane-local `BatchKVMeta`，再以 `chunk->pipelineId` 选择执行资源。上层 `mBatchMeta` 保存整个 batch 的调度/KV 视图；`prepareDualPipelineBatchMeta()` 只生成本次 lane module forward 所需的局部视图。

所以同一 segment 的不同 request group 可以并行前进，但已经开始的 HTTP batch 不会与后来 Task 重新混排。服务层的 2 ms 收集窗口和 LLM 内的 segment wave 是两个不同层面的 batch 边界。

## 9. 两条 lane 为什么能同时执行，何处又会汇合

`transformers/llm/engine/src/llm.cpp:306` 的 `prepareDualPipelineExecutionState()` 为每个 `mDualPipelineRuntimes[i]` 创建独立的 `Express::Executor`、`RuntimeManager` 与 `BatchKVMeta`。`transformers/llm/engine/src/llm.cpp:385` 的 `getDualPipelineModule()` 也把 Module pool 放在对应 lane，而不是共享 `mModule`。

`transformers/llm/engine/src/llm.cpp:1450` 对 `tasks` 中每一条 lane 创建一个 `std::thread`；worker 在该 lane 的 `ExecutorScope` 内调用自己的 `module->onForward(...)`，记录 `completedUs`，并在空输出时取消 stage/prefetch wave。紧接着 `transformers/llm/engine/src/llm.cpp:1466` 逐个 `join()` worker。这里的源码顺序就是“并发发射，然后统一汇合”，而不是将 worker 交给常驻线程池后继续立即调度。

完整的并发边界如下。`before/after` callback 来自 `transformers/llm/engine/src/llm.cpp:350` 的 lane-local executor；只有 callback 放行后，所在 op 才继续执行。

```text
control thread
    -> `scheduleWave()`
    -> graph window + `beginStageWave()`
    -> launch lane 0 worker ----------------------------+
    -> launch lane 1 worker -------------------------+   |
                                                  |   |   |
lane 0 executor/module                         lane 1 executor/module
    -> Host/QNN callback                           -> Host/QNN callback
    -> `onForward()`                               -> `onForward()`
    -> `lane_completed`                            -> `lane_completed`
                                                  |   |   |
                                                  +---+---+
                                                      v
                                               `workers.join()`
                                                      v
                    `finishStageWave()` + `finishGraphPrefetchWave()`
                                                      v
                                     sample/update/KV release/next wave
```

因此“同时执行”成立在两个 lane 的独立 executor/module 和各自 worker thread 上。它不表示同类 backend 可以无上限并发，也不表示 lane 0 完成后可绕过 lane 1 立即开始下一 segment：`join()` 是明确的 wave barrier。`lane_completed.elapsed_us` 能比较 lane 结束时间，但不能证明下一 wave 已启动。

## 10. 算子优先级与并发上限如何控制

这里有两种不同的“优先级”，需要分开审查。

### 10.1 执行阶段：Host/QNN 两条 FIFO ready queue

`transformers/llm/engine/src/llm.cpp:350` 的 before callback 先查 `activeQnnOpIndices`：匹配 QNN Plugin op-name 就调用 `enterGraphStage(pipelineId, graphIndex)`，其余 op 调 `enterStage(pipelineId, STAGE_HOST)`。after callback 对称调用 `leaveGraphStage()` 或 `leaveStage()`，从而保持每条 lane 的 op 顺序。

`transformers/llm/engine/src/DualPipelineScheduler.cpp:414` 把 waiter 放入 `mHostReadyStages` 或 `mQnnReadyStages`。grant 规则不是 layer number、token 位置或固定 lane 优先级，而是各自 FIFO 队列的队首：

```cpp
// transformers/llm/engine/src/DualPipelineScheduler.cpp:514
void DualPipelineScheduler::_grantReadyStagesLocked() {
    if (mActiveHostStages == 0 && !mHostReadyStages.empty()) {
        std::shared_ptr<StageWaiter> waiter = mHostReadyStages.front();
        mHostReadyStages.pop_front();
        mActiveStages[waiter->pipelineId] = waiter->backend;
        ++mActiveHostStages;
        // ... update `max_host` and possible overlap count.
        waiter->granted = true;
    }
    if (mActiveQnnStages == 0 && !mQnnReadyStages.empty()) {
        std::shared_ptr<StageWaiter> waiter = mQnnReadyStages.front();
        mQnnReadyStages.pop_front();
        mActiveStages[waiter->pipelineId] = waiter->backend;
        ++mActiveQnnStages;
        // ... update `max_qnn` and possible overlap count.
        waiter->granted = true;
    }
    mStageCondition.notify_all();
}
```

```text
lane 0 Host callback -> Host FIFO --+-- active Host == 0 ? -> grant one Host op
                                    |
lane 1 Host callback -> Host FIFO --+

lane 0 QNN callback  -> QNN FIFO  --+-- active QNN == 0 ? -> grant one QNN op
                                    |
lane 1 QNN callback  -> QNN FIFO  --+

Host active + QNN active: allowed, `overlap_grants` increments
two Host active or two QNN active: rejected by the gate
```

两个 backend 都空闲时，单次 `_grantReadyStagesLocked()` 可以各放行一个 Host 和一个 QNN waiter。因此 Host/QNN 可 overlap，但不能把它描述为两个 QNN op 同时跑。`finishStageWave()` 的 `stage_summary` 会输出 `max_host`、`max_qnn` 和 `overlap_grants`。

### 10.2 图加载阶段：离执行游标越近，优先级越高

QNN graph load 不使用上面的 stage FIFO。`transformers/llm/engine/src/DualPipelineScheduler.cpp:561` 的 `_popNextTaskLocked()` 在 graph-load task 中按以下顺序选一个：

1. `graphIndex - currentGraphIndex` 的非负距离更小者优先。
2. 距离相同，`pipelineId` 更小者优先。
3. 仍相同，`enqueueSequence` 更早者优先。

这让刚要被某条 lane 用到的图先进入 resident 状态，而不会让较早批量入队的远端图长期堵住当前图。该规则只排序加载任务，不改变 Host/QNN stage 并发上限，也不绕过 `enterGraphStage()` 对 resident 状态的等待。

## 11. QNN 图如何做局部、滑动式预取

### 11.1 先从模型快照得到可加载的具体 graph

`transformers/llm/engine/src/llm.cpp:455` 建立 QNN Plugin op-name 到 graph-index 的映射；dual wave 以 lane 的 `paddedCulLen` 调用 `transformers/llm/engine/src/DualPipelineGraph.cpp:575` 的 `buildQnnGraphRequestsForSize()`。当导出物带有 `graphPaths` 时，函数选择能覆盖当前 request group 的 shape/bucket，再把单 shape 的 path/name 写入 `GraphRequest`：

```cpp
// transformers/llm/engine/src/DualPipelineGraph.cpp:575
std::vector<DualPipelineScheduler::GraphRequest> buildQnnGraphRequestsForSize(
    const GraphSnapshot& snapshot, int start, int maxK, int requestGroupSize) {
    std::vector<DualPipelineScheduler::GraphRequest> requests;
    for (int i = start; i < static_cast<int>(snapshot.ops.size()) &&
         static_cast<int>(requests.size()) < maxK; ++i) {
        const OpInfo& op = snapshot.ops[i];
        if (!isQnnPluginOp(op)) {
            continue;
        }
        DualPipelineScheduler::GraphRequest request;
        // ... initialize common graph metadata.
        if (requestGroupSize > 0 && !op.qnn.graphPaths.empty()) {
            const int shapeIndex = selectQnnShapeIndex(op.qnn, requestGroupSize);
            if (shapeIndex < 0) {
                return {};
            }
            request.graphId = qnnGraphResourceIdForShape(op, shapeIndex);
            request.graphPath = qnnGraphPathForShape(op, shapeIndex);
            request.offset = 0;
            request.size = 0;
            request.shapeIndex = shapeIndex;
            request.bucketSize = op.qnn.bucketSizes[shapeIndex];
            request.allGraphName = {op.qnn.allGraphName[shapeIndex]};
        }
        // ... copy draft/pin metadata.
        requests.push_back(request);
    }
    return requests;
}
```

`qnn_bucket` trace 会记录 lane、graph index、shape index 和 bucket。若当前模型是 combined binary、没有 per-shape `allGraphPath`，`shape_index=-1, bucket=-1` 是元数据边界，不应误读为预取失败。

### 11.2 window 只覆盖当前图附近，而不是整次 forward

`transformers/llm/engine/src/DualPipelineScheduler.cpp:183` 的 `beginGraphPrefetchWave()` 为每条 lane 保存 `graphs`、`ownerRequestIds`、`currentGraphIndex` 与 `requestedUntil`，随后最多扩到 `dual_pipeline_prefetch_window`。当 executor 真正进入某个 QNN op，`enterGraphStage()` 位于 `transformers/llm/engine/src/DualPipelineScheduler.cpp:222`：它推进 `currentGraphIndex`，将 window 扩到 `graphIndex + lookahead`，把图放入 `mPendingQnnGraphs`，并等待 `loadFinished && resident` 后才申请 QNN stage。

```text
lane p, lookahead = 2

begin prefetch wave
    cursor = -1, request graph 0..2
    -> loader sorts by distance and calls `onGraphLoad`

executor reaches QNN graph 0
    cursor = 0, request extends only through graph 2
    wait graph 0 resident -> acquire the single QNN stage

executor leaves QNN graph 0
    -> `leaveGraphStage()` enqueues completion
    -> active use decreases; graph may remain resident or later be LRU-evicted

executor reaches QNN graph 1
    cursor = 1, request extends only through graph 3
```

`leaveGraphStage()` 在 `transformers/llm/engine/src/DualPipelineScheduler.cpp:296` 清除 `executingGraphIndex` 与 `mQnnExecutionActive`，release stage，并为已消费图追加 completion。`_processGraphComplete()` 在 `transformers/llm/engine/src/DualPipelineScheduler.cpp:746` 递减 `activeUseCount`；`_planEvictionsLocked()` 只会选非 pinned、非 active-use 的 LRU resident 图。wave 结束时 `finishGraphPrefetchWave()` 取消未消费的 pending load，并为已登记但未消费的图补 completion。

这就是“局部预取”的资源边界：加载窗口由执行游标推进，已消费图不再因 window owner 而保持 active use；resident cache 仍可跨 wave 保存图，直到 LRU 或 scheduler stop 释放。纯 CPU/OpenCL 模型没有 QNN Plugin graph 时 `graphs` 为空，预取协议成功返回但不会产生 QNN load 或 QNN stage。

## 12. server 模式代码、三种解码方式的使用与轻量测试

### 12.1 服务端只有一个 LLM 调用 worker

HTTP handler 可以并发进来，但 `apps/mnncli/src/mnncli_server.cpp:412` 的 `RequestCoordinator::workerLoop()` 是同一个 `Llm` 的唯一调用者。它在 `single_request` 只取一个 Task；其他两种 mode 最多等待 2 ms，并取至多四个、`max_tokens` 与首 Task 相同的请求：

```cpp
// apps/mnncli/src/mnncli_server.cpp:412
void workerLoop() {
    while (true) {
        std::vector<std::shared_ptr<Task>> batch;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mReady.wait(lock, [this] { return mStopping || !mPending.empty(); });
            // ... skip cancelled pending tasks and take the first Task.
            if (mMode != Mode::SingleRequest) {
                mReady.wait_for(lock, std::chrono::milliseconds(2), [this] {
                    return mStopping || !mPending.empty();
                });
                const int max_tokens = batch.front()->max_tokens;
                while (!mPending.empty() && batch.size() < kMaxBatchSize &&
                       mPending.front()->max_tokens == max_tokens) {
                    batch.push_back(mPending.front());
                    mPending.pop_front();
                }
            }
            mExecuting = true;
            mExecutingTasks = batch;
        }
        if (mMode == Mode::SingleRequest) {
            runSingle(batch.front());
        } else {
            runBatch(batch);
        }
        // ... clear `mExecutingTasks`, then process the next group.
    }
}
```

`runSingle()` 位于 `apps/mnncli/src/mnncli_server.cpp:289`，通过 `LlmStreamBuffer` 在生成回调中 `Task::append()`；`runBatch()` 位于 `apps/mnncli/src/mnncli_server.cpp:333`，在 `Llm::generate(vector)` 返回后才把每个 token result decode/append。这是三种 decode mode 共享 HTTP API、却具有不同响应时序的直接原因：batch/dual 有 SSE transport，但目前是 `delivery=batch_completion`，不是逐 token batch streaming。

### 12.2 配置与启动

现有模型 JSON 中保留 `llm_model`、`llm_weight`、`tokenizer_file` 等模型字段，只覆写下列 mode fragment。`transformers/llm/engine/src/llmconfig.hpp:341` 规定 canonical `scheduler_mode` 优先于 legacy dual flag；`transformers/llm/engine/src/llmconfig.hpp:366` 到 `transformers/llm/engine/src/llmconfig.hpp:379` 定义 dual 的 split/resident/lookahead 字段；`transformers/llm/engine/src/llm.cpp:623` 强制 continuous/dual 使用 `packed_attention_mode=true`。

单请求配置片段：

```json
{
  "backend_type": "cpu",
  "scheduler_mode": "single_request"
}
```

连续 batch 配置片段：

```json
{
  "backend_type": "cpu",
  "scheduler_mode": "continuous_batch",
  "packed_attention_mode": true
}
```

双流水线配置片段：

```json
{
  "backend_type": "cpu",
  "scheduler_mode": "dual_pipeline",
  "packed_attention_mode": true,
  "dual_pipeline_split_count": 2,
  "dual_pipeline_max_resident_graphs": 5,
  "dual_pipeline_prefetch_window": 2
}
```

对纯 CPU/OpenCL 模型，只需把 `backend_type` 改为 `opencl` 并复用同样三组 scheduler 配置；dual 仍可验证 lane 和 Host stage，但预期 `qnn_completed=0`、`max_qnn=0`。对 QNN Plugin 模型，启动环境还必须提供模型所需的 QNN runtime library path；不要把 CPU/OpenCL Host backend 与模型内部 QNN Plugin 图混为一谈。

```bash
# 服务端或 Android 设备端执行；每个 mode 使用独立端口并保存 stderr trace。
MNN_ACCEPTANCE_TRACE=1 mnncli serve --config /path/to/dual.json --host 127.0.0.1 --port 18080 2> /tmp/mnncli-dual.trace

# 覆盖优先级测试：JSON 写 single，但 CLI 在 load 前覆盖为 dual。
MNN_ACCEPTANCE_TRACE=1 mnncli serve --config /path/to/single.json --scheduler-mode dual_pipeline --host 127.0.0.1 --port 18081

# Android 主机侧执行；设备上服务监听 18080 时建立转发。
adb forward tcp:18080 tcp:18080
```

`apps/mnncli/src/handlers/serve_command_handler.cpp:34` 解析并校验三种 CLI 值，`apps/mnncli/src/llm_manager.cpp:14` 在 `load()` 前将 CLI mode 写入 runtime config。因此第二个命令的行为应按 dual 观察，而不是按 JSON 中的 single 观察。

### 12.3 用短请求验收 HTTP、SSE、并发和取消

以下命令在能访问服务的主机执行。每轮只启动一个 mode 的 server；`continuous_batch`/`dual_pipeline` 的并发样本必须使用相同 `max_tokens`，否则 coordinator 会把后续请求留给下一 batch。

```bash
# 客户端主机执行：readiness 和非流式 OpenAI JSON。
BASE=http://127.0.0.1:18080
curl -fsS "$BASE/v1/models"
curl --fail-with-body -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"local","messages":[{"role":"user","content":"Reply with one word."}],"max_tokens":2}'
```

```bash
# 客户端主机执行：SSE 必须看到至少一个 `data:` frame、finish frame 和 `[DONE]`。
curl -N --no-buffer -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"local","messages":[{"role":"user","content":"Reply with one word."}],"max_tokens":3,"stream":true}' | tee /tmp/mnncli-sse.out
grep -q '^data: ' /tmp/mnncli-sse.out
grep -q '"finish_reason":"stop"' /tmp/mnncli-sse.out
grep -q 'data: \[DONE\]' /tmp/mnncli-sse.out
```

```bash
# 客户端主机执行：同一 max_tokens 的两条请求并发提交。
for id in 1 2; do curl --fail-with-body -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"local","messages":[{"role":"user","content":"Reply with one word."}],"max_tokens":3}' > "/tmp/mnncli-response-${id}.json" & done
wait

# 主动断开 SSE，再确认服务仍可处理下一条 health 请求。
curl --max-time 1 -N --no-buffer -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"local","messages":[{"role":"user","content":"Count slowly from one."}],"max_tokens":32,"stream":true}' || true
curl -fsS "$BASE/v1/models"
```

验收时应同时查看 HTTP 输出和 server stderr 中的 `MNN_ACCEPTANCE_TRACE`。`transformers/llm/engine/include/llm/AcceptanceTrace.hpp:18` 表明该 trace 默认关闭，只有环境变量非空且不是 `0` 才输出。

| mode | 正确性重点 | trace 中应看到什么 | 不应声称什么 |
|---|---|---|---|
| `single_request` | JSON 非空；SSE 在生成中有内容；取消后下一请求成功 | 两个并发请求的 `request_started/request_completed` 严格串行 | 不把 HTTP handler 并发误当作 LLM forward 并发 |
| `continuous_batch` | JSON/SSE 正常；两条相同 token budget 请求合为一组；取消不伤害同 batch 同伴 | `batch_started request_count=2`、`batch_member`、`delivery=batch_completion` | 不称作 decode 中动态 admission 或逐 token batch SSE |
| `dual_pipeline` | HTTP 行为同 batch；两个 group 获得不同 lane；QNN/Host 不变量成立 | `lane_owner` 含 lane 0/1；`lane_completed`；`stage_summary max_host<=1,max_qnn<=1` | 不称作双 QNN 并行或 lane 完成即续跑 |

对 QNN Plugin 模型，可检查 `qnn_bucket` 是否符合实际 shape/bucket，并以 `overlap_grants>0` 作为 Host/QNN 跨类 overlap 的证据。对 QNN-free CPU/OpenCL 模型，正确预期是 dual lane 存在、QNN 计数为零；没有 QNN graph/load trace 不能视为失败。

## 13. 取消与失败主线

SSE 写失败或 socket 不可写时，`apps/mnncli/src/mnncli_server.cpp:666` 调用 `RequestCoordinator::cancel(task)`。取消路径分为三层：

1. **queued**：`apps/mnncli/src/mnncli_server.cpp:146` 直接从 `mPending` 移除并 complete。
2. **executing single/batch member**：Task 的 atomic cancel flag 立即生效；`Task::append()` 会丢弃后续文本。
3. **engine stop**：只有 `mExecutingTasks` 的所有成员都已取消时，coordinator 才调用 `Llm::requestCancel()`，避免一个断开客户端中止仍在服务的 batch 同伴。

```cpp
// transformers/llm/engine/src/llm.cpp:1188
void Llm::requestCancel() {
    mCancelRequested.store(true, std::memory_order_release);
}

bool Llm::cancelRequested() {
    if (!mCancelRequested.load(std::memory_order_acquire)) {
        return false;
    }
    if (mContext->status != LlmStatus::INTERNAL_ERROR) {
        mContext->status = LlmStatus::USER_CANCEL;
    }
    return true;
}
```

generation loop 在安全 token/wave 边界查询这个原子旗标，例如 `transformers/llm/engine/src/speculative_decoding/generate.cpp:61`、`transformers/llm/engine/src/llm.cpp:1326` 和 `transformers/llm/engine/src/speculative_decoding/eagle_batch.cpp:1310`。因此取消是协调器可见的立即状态变化，但 engine 退出点取决于当前 generation path 的下一处安全检查。

## 14. 建议按下面的问题做 code review

| 审查问题 | 应阅读的位置 | 当前代码给出的答案 |
|---|---|---|
| CLI/JSON mode 会不会分裂？ | `apps/mnncli/src/handlers/serve_command_handler.cpp:34`、`apps/mnncli/src/llm_manager.cpp:14`、`transformers/llm/engine/src/llm.cpp:623` | CLI 在 load 前写入 canonical mode，load 后锁定 |
| 同一个 Llm 会不会被多个 HTTP handler 并发调用？ | `apps/mnncli/src/mnncli_server.cpp:128`、`apps/mnncli/src/mnncli_server.cpp:412` | coordinator 单 worker 串行调用 `runSingle/runBatch` |
| continuous 是否真的能中途接纳新请求？ | `apps/mnncli/src/mnncli_server.cpp:412`、`runBatch()` | 否；只在 2 ms 入队窗口组 batch，generation 开始后等待返回 |
| batch SSE 是否逐 token？ | `apps/mnncli/src/mnncli_server.cpp:333`、`apps/mnncli/src/mnncli_server.cpp:666` | 否；SSE provider 可轮询，但 batch 文本在完整结果返回后 append |
| dual 是否让两个 lane 共用 Module？ | `transformers/llm/engine/src/llm.cpp:320`、`transformers/llm/engine/src/llm.cpp:385`、`transformers/llm/engine/src/llm.cpp:1450` | 否；每 lane 有 executor/runtime/batch meta/module pool |
| dual 是否允许两个 Host 或两个 QNN op 同时跑？ | `transformers/llm/engine/src/DualPipelineScheduler.cpp:414`、`transformers/llm/engine/src/DualPipelineScheduler.cpp:514` | 否；各 class 最多一个，Host 与 QNN 可以 overlap |
| dual 是否实现先完成 lane 的立即续跑？ | `transformers/llm/engine/src/llm.cpp:1450`、`transformers/llm/engine/src/llm.cpp:1466` | 否；每 wave 有 all-lane join barrier |
| dual wave 如何保证不是同一 request 的重复执行？ | `transformers/llm/engine/src/BatchScheduler.cpp:226`、`transformers/llm/engine/src/llm.cpp:1329` | `scheduleWave()` 只聚合同 segment 的独立 chunk；每个 chunk 带自己的 `reqId/pipelineId` |
| 算子优先级是否偏向某条 lane？ | `transformers/llm/engine/src/DualPipelineScheduler.cpp:414`、`transformers/llm/engine/src/DualPipelineScheduler.cpp:514` | stage 按 Host/QNN 各自 FIFO；无固定 lane 优先级，最多各活动一个 |
| graph load 如何优先当前 lane 的下一张图？ | `transformers/llm/engine/src/DualPipelineScheduler.cpp:548`、`transformers/llm/engine/src/DualPipelineScheduler.cpp:561`、`transformers/llm/engine/src/DualPipelineScheduler.cpp:614` | 按游标距离排序，平局按 pipeline id/入队序列；window 随 cursor 扩展 |
| QNN 图是否在整次 forward 前全部加载？ | `transformers/llm/engine/src/DualPipelineScheduler.cpp:183`、`transformers/llm/engine/src/DualPipelineScheduler.cpp:222`、`transformers/llm/engine/src/DualPipelineScheduler.cpp:335` | 否；初始只到 lookahead，进入图时再滑动扩展；未消费项在 wave 结束清理 |
| 单个 batch 成员取消会不会中止其他成员？ | `apps/mnncli/src/mnncli_server.cpp:146` | 不会；只有全部 active Task 已取消才请求 engine cancel，单成员只抑制输出 |
| QNN-free 模型会不会误走 QNN stage？ | `transformers/llm/engine/src/llm.cpp:455`、`transformers/llm/engine/src/llm.cpp:350` | graph/op map 为空，executor callback 全部走 Host；实测 `max_qnn=0` |

## 15. 审查时可观察的 trace

| trace | 来源 | 用于确认 |
|---|---|---|
| `request_enqueued/request_started/request_completed` | coordinator | 排队、TTFT、取消、最终状态 |
| `batch_started/batch_member` | `runBatch()` | coordinator 实际组成的 HTTP batch |
| `batch_engine_member` | `Llm::generate(vector)` | HTTP Task 到 engine request 的对应 |
| `lane_owner/lane_completed` | `BatchScheduler::scheduleWave()`、dual loop | request group 到 pipeline lane 的归属和 wave 完成 |
| `qnn_bucket` | graph request enqueue | QNN per-shape request 的 metadata；combined binary 可为 `-1/-1` |
| `stage_summary` | `finishStageWave()` | Host/QNN 完成数、overlap grant、最大并发、取消状态 |

## 16. 小结

- 真实服务入口是 `ServeCommandHandler::Handle`，它在 `load()` 前确定 mode，再创建一个绑定该 mode 的 `RequestCoordinator`。
- coordinator 的核心职责是 Task 生命周期和小窗口合批；single 可以实时 append，batch/dual 只在完整 batch 返回后交付内容。
- `Llm::generate(vector)` 是三模式进入 generation 的分叉点；dual AR 额外拥有 request scheduler、lane-local execution resources 和 graph/stage scheduler。
- dual 的 stage 控制按 op callback 执行：Host<=1、QNN<=1、可跨类 overlap；不过每个 dual wave 仍有 all-lane join barrier。
- `scheduleWave()` 只把同一 segment 的独立 request group 组成 wave；lane 的独立 executor/module 和 worker thread 提供并发，但采样前仍统一 join。
- QNN 图按 lane 的执行游标局部预取，loader 以图距离排序；它和 Host/QNN stage gate 是两套互补但不同的调度机制。
- server 的三种 mode 在 `load()` 前由 `scheduler_mode` 固定；短 JSON/SSE/并发/取消矩阵足以验证功能边界，不需要 GSM8K 或长性能任务。
- 取消先解决 HTTP 输出和队列所有权，再在所有 active batch member 都取消时请求 engine stop。下一步若要继续深入，优先阅读 `BatchScheduler::schedule()` 的 request state 转换和 `DualPipelineScheduler::_workerLoop()` 的 graph resident/LRU 路径。
