# MNN 双流水线 / QNN Prefetch 改动 Code Review 说明

本文记录当前 `mnn-pp` 工作区里双流水线相关未提交改动的真实代码路径。主线从 `Llm::configureDualPipelineMode()` 读配置开始，到 `BatchScheduler::scheduleWave()` 生成同一 segment 的两条 pipeline chunk，再到 `Llm::generate(batch)` 用两套 runtime/module 并发执行，期间把 QNN graph prefetch/release 信号交给 `DualPipelineScheduler` worker，最后由 `QNNBackend.cpp` 里的 CPU plugin raw graph pool 复用预加载的 QNN 离线图。

## 1. 改动总览

这组改动分成五层：

| 层级 | 入口 | 主要职责 | 主要文件 |
|---|---|---|---|
| 配置层 | `Llm::configureDualPipelineMode()` | 打开/关闭 dual mode，启动 prefetch worker，安装 QNN load/release callback | `llm.cpp` |
| 请求调度层 | `BatchScheduler::scheduleWave()` | 按同一 `segmentIndex` 返回 disjoint pipeline chunks | `BatchScheduler.*` |
| 执行资源层 | `prepareDualPipelineExecutionState()` | 为 pipeline 0/1 分别准备 executor、runtime manager、module pool、`BatchKVMeta` | `llm.*`, `Executor.cpp`, `kvmeta.hpp` |
| 图预取层 | `buildGraphSnapshot()` / `DualPipelineScheduler` | 只读图快照，构造 CPU/OpenCL resize request 和 QNN graph request，串行处理 load/complete/release | `DualPipelineGraph.*`, `DualPipelineScheduler.*` |
| QNN plugin 层 | `MNN::QNN::preloadRawGraph()` | 预加载 QNN raw graph，CPU plugin `PluginExecuteRaw` 优先复用 resident executor | `QNNPrefetch.hpp`, `QNNBackend.cpp` |

本轮清理已经去掉几类冗余调试/无用逻辑：

- 删除双流水线普通成功路径的 `MNN_DUAL_PIPELINE` enqueue/snapshot 统计日志；
- 删除 QNN raw graph resident 命中/成功预取/复用的普通 `MNN_PRINT`；
- 删除非 dual mode batch 路径里实际只会 no-op 的 prefetch/complete 调用；
- 恢复 Android 脚本中本地验证留下的过高 `make -j256` 改动。

仍然保留错误路径日志和安全边界代码，例如 QNN preload 失败的 `MNN_ERROR`、`StaticModule` session 缺失诊断、CPU/OpenCL resize callback 的 no-op 边界。

## 2. 逻辑主线

```text
`dual_pipeline_mode=true`
    ↓
`Llm::configureDualPipelineMode()`
    ↓
`BatchScheduler::setDualPipelineMode()` + `DualPipelineScheduler::start()`
    ↓
`Llm::load()` 缓存模型 IO / Module config，并准备两套 pipeline runtime
    ↓
`Llm::generate(vector<vector<int>>)`
    ↓
`scheduleBatchWave()` -> `BatchScheduler::scheduleWave()`
    ↓
同一 wave 内 pipeline 0 / pipeline 1 chunk 分别绑定独立 Module
    ↓
chunk 执行前：`enqueueDualPipelineChunkGraphs()`
    ↓
prefetch worker: CPU/OpenCL resize callback + QNN raw graph preload
    ↓
两个线程分别 `Module::onForward()`
    ↓
chunk 执行后：`completeDualPipelineChunkGraphs()`
    ↓
request 完成：release KV / release QNN graph / release scheduler request
```

需要先明确边界：当前实现已经做到“两个 logical pipeline 分别跑两个 request chunk 的并发 `onForward()`”，但还没有把一个模型内部切成 CPU/GPU attention stage 和 QNN FFN stage 后做 stage-level 交错执行。QNN 仍通过 CPU plugin raw graph 路径加载和执行，不是把 MNN op 直接改成 QNN backend stage scheduler。

## 3. 调度层：同一 segment 的 wave

`Chunk` 增加 `pipelineId` 和 `segmentIndex`。`pipelineId` 标识逻辑执行线，`segmentIndex` 标识 prefill split 后的 token 段。

```cpp
// transformers/llm/engine/include/llm/BatchScheduler.hpp:59
struct Chunk {
    std::vector<std::vector<int>> inputs; // only read data to make embedding
    std::vector<int> calLen;        // calculated lengths for each input token in the chunk
    std::vector<int> pos;           // position for each input token in the chunk
    std::vector<int> reqId;         // global request id
    std::vector<int> state;         // request state when this chunk was scheduled
    int culLen = 0;                 // cumulative length of the chunk
    int pipelineId = 0;             // dual-pipeline logical pipeline id
    int segmentIndex = 0;           // token segment index inside a dual-pipeline wave
};
```

`schedule()` 仍是兼容接口：它会构造 ordered chunks，把第一个返回，其余放入 `mPendingChunks`。dual mode 下固定两条 execution pipeline；`dual_pipeline_split_count` 现在只决定 prefill token segment 数，不决定 pipeline 数。

```cpp
// transformers/llm/engine/src/BatchScheduler.cpp:137
std::shared_ptr<BatchScheduler::Chunk> BatchScheduler::schedule(int blockSize, int bs) {
    // ...
    if (mDualPipelineMode && !scheduledItems.empty()) {
        // ...
        const int pipelineCount = 2;
        std::vector<std::vector<int>> pipelineItems(pipelineCount);
        // existing request keeps its previous pipeline assignment
        // new requests are partitioned across pipeline 0 / 1
        // ...
        const int segmentCount = hasSplitRequest ? mDualPipelineSplitCount : 1;
        std::vector<std::shared_ptr<Chunk>> orderedChunks;
        for (int segment = 0; segment < segmentCount; ++segment) {
            for (int pipeline = 0; pipeline < pipelineCount; ++pipeline) {
                auto chunk = std::make_shared<Chunk>();
                chunk->pipelineId = pipeline;
                chunk->segmentIndex = segment;
                // ...
                if (chunk->culLen > 0) {
                    orderedChunks.push_back(chunk);
                }
            }
        }
        // ...
        _commitChunk(task);
        return task;
    }
    // ...
    return task;
}
```

`scheduleWave()` 是 true dual execution 新增的关键 API：它先取一个 chunk，然后只弹出同一 `segmentIndex` 的 pending chunks。这样 coordinator 不需要连续调用 `schedule()` 猜测哪些 chunk 可以并发，而是一次拿到同一 wave。

```cpp
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
```

测试里已经覆盖同一 wave 返回两条 pipeline、单请求只返回一条 pipeline、decode 阶段 pipeline assignment 保持不变、`split_count > 2` 不会产生 pipeline id 2/3。

```cpp
// test/llm/BatchSchedulerTest.cpp:202
class BatchSchedulerDualPipelineScheduleWaveTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        BatchScheduler scheduler;
        scheduler.setDualPipelineMode(true, 2);
        std::vector<int> reqIds = scheduler.addRequest({{1, 2, 3, 4}, {11, 12, 13, 14}});

        std::vector<std::shared_ptr<BatchScheduler::Chunk>> firstWave = scheduler.scheduleWave(4, 2);
        MNNTEST_ASSERT(firstWave.size() == 2);
        MNNTEST_ASSERT(firstWave[0]->segmentIndex == 0);
        MNNTEST_ASSERT(firstWave[1]->segmentIndex == 0);
        MNNTEST_ASSERT(firstWave[0]->pipelineId == 0);
        MNNTEST_ASSERT(firstWave[1]->pipelineId == 1);
        // ...
        return true;
    }
};
```

## 4. LLM 执行层：两套 runtime/module 并发

`configureDualPipelineMode()` 是配置入口。开启时启动 `DualPipelineScheduler`，并把 graph load/release 接到 `MNN::QNN` public prefetch API。CPU/OpenCL resize prefetch 保留 callback 边界，当前不调用 live `Execution::onResize()`。

```cpp
// transformers/llm/engine/src/llm.cpp:182
void Llm::configureDualPipelineMode() {
    const bool enabled = mConfig->dual_pipeline_mode();
    mScheduler->setDualPipelineMode(enabled, mConfig->dual_pipeline_split_count());
    if (!enabled) {
        resetDualPipelineGraphState();
        resetDualPipelineExecutionState();
        stopDualPipelineRequestThread();
        // ...
        return;
    }

    // CPU/OpenCL resize prefetch must use an isolated runtime or backend callback,
    // never the live session that is executing onForward.
    config.callbacks.onPrefetchResize = [](const DualPipelineScheduler::PrefetchResizeRequest&) {
    };
    config.callbacks.onGraphLoad = [](const DualPipelineScheduler::GraphRequest& request) {
        // ...
        if (!QNN::preloadRawGraph(prefetchConfig)) {
            MNN_ERROR("MNN_QNN: Failed to prefetch graph %s from %s.\n",
                      request.graphId.c_str(), prefetchConfig.path.c_str());
            return false;
        }
        return true;
    };
    // ...
}
```

执行资源由 `DualPipelineRuntime` 持有，每条 pipeline 都有自己的 `Executor`、`RuntimeManager`、`BatchKVMeta` 和 module pool，避免两个线程并发调用同一个 `Module` / `Session`。

```cpp
// transformers/llm/engine/include/llm/llm.hpp:229
struct DualPipelineRuntime {
    std::shared_ptr<Express::Executor> executor;
    std::shared_ptr<Express::Executor::RuntimeManager> runtimeManager;
    std::shared_ptr<BatchKVMeta> batchMeta;
    std::map<std::pair<int, bool>, std::shared_ptr<Express::Module>> modulePool;
};
```

```cpp
// transformers/llm/engine/src/llm.cpp:351
bool Llm::prepareDualPipelineExecutionState() {
    if (!mConfig->dual_pipeline_mode() || !mModule || !mRuntimeManager) {
        return false;
    }
    // ...
    for (size_t i = 0; i < mDualPipelineRuntimes.size(); ++i) {
        auto& runtime = mDualPipelineRuntimes[i];
        runtime.executor = Express::Executor::newExecutor(type, backendConfig, numThread);
        // ...
        runtime.runtimeManager.reset(Express::Executor::RuntimeManager::createRuntimeManager(runtimeConfig));
        runtime.batchMeta.reset(new BatchKVMeta);
        setRuntimeHint(runtime.runtimeManager, runtime.batchMeta.get(), mMeta.get());
    }
    mDualPipelineExecutionReady = true;
    return true;
}
```

`generate(batch)` 的 dual 分支按 wave 构建 `DualWaveTask`，先 enqueue 当前 full-module forward 所需的全部 QNN binary，等待 `waitForGraphsReady()` 成功，再创建 pipeline module 并开线程并发 `onForward()`。

```cpp
// transformers/llm/engine/src/llm.cpp:1254
std::vector<std::vector<int>> Llm::generate(const std::vector<std::vector<int> >& input_ids,
                                            std::ostream* os,
                                            int max_new_tokens) {
    // ...
    } else {
        struct DualWaveTask {
            std::shared_ptr<BatchScheduler::Chunk> chunk;
            std::shared_ptr<Module> module;
            std::shared_ptr<BatchKVMeta> batchMeta;
            std::vector<std::string> graphIds;
            std::vector<Express::VARP> outputs;
            // ...
        };
        // ...
        std::vector<std::shared_ptr<BatchScheduler::Chunk>> wave = scheduleBatchWave(-1, 4);
        // build one task per pipeline chunk, then enqueue graph prefetch
        // ...
        enqueueDualPipelineChunkGraphs(*chunk, &task.graphIds);
        // ...
        workers.emplace_back([this, &tasks, i]() {
            Express::ExecutorScope scope(mDualPipelineRuntimes[tasks[i].chunk->pipelineId].executor);
            tasks[i].outputs = tasks[i].module->onForward(
                {tasks[i].hiddenStates, tasks[i].attentionMask, tasks[i].positionIds, tasks[i].logitsIndex});
        });
        // ...
        completeDualPipelineChunkGraphs(task.graphIds);
    }
    // ...
    return ret;
}
```

这里的并发粒度是“两个 request chunk 同时完整跑各自 Module”。它能让请求 1/2 的完整 forward 互相遮盖，但不是 “request A attention stage + request B QNN FFN stage” 的子图级遮盖。

## 5. 图预取层：只读 snapshot + worker 状态机

`refreshDualPipelineGraphSnapshot()` 只从 `StaticModule` 的 session 拿 pipeline metadata。这个函数不调用 resize、encode、alloc、`Execution::onResize()`。

```cpp
// transformers/llm/engine/src/llm.cpp:440
bool Llm::refreshDualPipelineGraphSnapshot() {
    resetDualPipelineGraphState();
    if (!mConfig->dual_pipeline_mode() || !mModule) {
        return false;
    }
    const StaticModule* staticModule = getStaticModuleForDualPipeline(mModule.get());
    if (staticModule == nullptr || staticModule->getSession() == nullptr) {
        MNN_PRINT("MNN_DUAL_PIPELINE: StaticModule session is not available for graph snapshot.\n");
        return false;
    }
    // Snapshot copies Session metadata once; the prefetch worker must not mutate live Session state.
    mDualPipelineGraphSnapshot = buildGraphSnapshot(staticModule->getSession(), 0, mConfig->base_dir_, mConfig->npu_model_dir());
    mDualPipelineGraphSnapshotReady = !mDualPipelineGraphSnapshot.ops.empty();
    return mDualPipelineGraphSnapshotReady;
}
```

`DualPipelineGraph` 从 `Session::getPipelineInfo()` 复制 op 顺序、backend、已知 tensor shape 和 plugin attrs。QNN plugin 判断现在既支持 type/backend 字符串里带 `qnn`，也支持只有 `path + allGraphName` 的导出模型。

```cpp
// transformers/llm/engine/src/DualPipelineGraph.cpp:262
bool isQnnPluginOp(const OpInfo& op) {
    if (!op.isPlugin) {
        return false;
    }
    if (containsToken(op.pluginType, "qnn") || containsToken(effectiveBackend(op), "qnn")) {
        return true;
    }
    return !op.qnn.allGraphName.empty() && (!op.qnn.relativePath.empty() || !op.qnn.path.empty());
}
```

```cpp
// transformers/llm/engine/src/DualPipelineGraph.cpp:338
std::vector<DualPipelineScheduler::GraphRequest> buildQnnGraphRequests(const GraphSnapshot& snapshot,
                                                                       int start,
                                                                       int maxK,
                                                                       int reqId) {
    std::vector<DualPipelineScheduler::GraphRequest> requests;
    // ...
    for (int i = start; i < static_cast<int>(snapshot.ops.size()) && static_cast<int>(requests.size()) < maxK; ++i) {
        const OpInfo& op = snapshot.ops[i];
        if (!isQnnPluginOp(op)) {
            continue;
        }
        DualPipelineScheduler::GraphRequest request;
        request.action = DualPipelineScheduler::GRAPH_LOAD;
        request.requestId = reqId;
        request.graphId = !op.qnn.targetGraphName.empty() ? op.qnn.targetGraphName : op.opName;
        request.graphPath = op.qnn.path;
        request.offset = op.qnn.offset;
        request.size = op.qnn.size;
        request.allGraphName = op.qnn.allGraphName;
        // ...
        requests.push_back(request);
    }
    return requests;
}
```

`DualPipelineScheduler` worker 只做控制面：prefetch resize callback、QNN graph load/release、complete、LRU eviction。`onGraphLoad` 返回 `bool` 后，load 失败会回滚 resident 和 active-use 状态，避免调度层误以为图已驻留。

```cpp
// transformers/llm/engine/src/DualPipelineScheduler.cpp:256
void DualPipelineScheduler::_processGraphRequest(const GraphRequest& request) {
    // ...
    if (shouldLoad) {
        const bool loadSucceeded = !callbacks.onGraphLoad || callbacks.onGraphLoad(callbackRequest);
        std::lock_guard<std::mutex> lock(mMutex);
        std::map<std::string, GraphState>::iterator iter = mGraphs.find(callbackRequest.graphId);
        if (iter != mGraphs.end()) {
            iter->second.loadFinished = true;
            iter->second.record.resident = loadSucceeded;
            if (!loadSucceeded) {
                iter->second.record.pinned = false;
                iter->second.record.activeUseCount = 0;
                iter->second.record.pendingRelease = false;
                iter->second.record.lastUseSequence = mNextSequence++;
            }
        }
    }
}
```

## 6. QNN raw graph prefetch：public API + CPU plugin 复用

新增 public header 给 LLM 层使用。它表达的是 “预加载 CPU plugin 里将要用到的 QNN raw graph”，不是 stage scheduler API。

```cpp
// include/MNN/QNNPrefetch.hpp:17
struct RawGraphPrefetchConfig {
    std::string graphId;
    std::string path;
    uint64_t offset = 0;
    uint64_t size = 0;
    std::vector<std::string> allGraphName;
    bool pinResident = false;
};

MNN_PUBLIC bool preloadRawGraph(const RawGraphPrefetchConfig& config);
MNN_PUBLIC void releaseRawGraph(const std::string& graphId, bool forceRelease = false, bool unpinAfterRelease = false);
```

public prefetch API 可以早于 QNN runtime creator 被调用，所以 `createQnnContext()` 自己补了一次 symbol load guard，避免 `QnnInterface_getProviders` 为空时崩溃。

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:43
static void createQnnContext(){
    std::lock_guard<std::mutex> lck(gQnnContextMutex);
    QNN_INTERFACE_VER_TYPE qnnInterface{};
#ifndef ENABLE_QNN_CONVERT_MODE
    if (QNN::QnnInterface_getProviders == nullptr
#ifdef MNN_WITH_PLUGIN
        || QNN::QnnSystemInterface_getProviders == nullptr
#endif
        ) {
        if (!QNN::loadQNNSymbol()) {
            return;
        }
    }
    // ...
}
```

resident raw graph pool 用 `graphId` 和 `path#offset#size#allGraphName` 两套索引。`preloadRawGraphInternal()` 命中 resident graph 时直接返回，未命中才 `compileModel()`。

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:907
struct RawGraphRecord {
    std::shared_ptr<RawExecutorWrapper> executor;
    std::string cacheKey;
    std::vector<std::string> graphIds;
    bool pinned = false;
};

static std::mutex gRawGraphPoolMutex;
static std::map<std::string, std::shared_ptr<RawGraphRecord>> gRawGraphById;
static std::map<std::string, std::shared_ptr<RawGraphRecord>> gRawGraphByKey;
```

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:952
static bool preloadRawGraphInternal(const MNN::QNN::RawGraphPrefetchConfig& config) {
    if (config.graphId.empty() || config.path.empty() || config.allGraphName.empty()) {
        return false;
    }
    if (QNN::gContext.deviceHandle == nullptr) {
        QNN::createQnnContext();
    }
    if (QNN::gContext.deviceHandle == nullptr) {
        return false;
    }

    const std::string cacheKey = makeRawGraphCacheKey(config.path, config.offset, config.size, config.allGraphName);
    std::lock_guard<std::mutex> lock(gRawGraphPoolMutex);
    // check graphId, then cacheKey
    // compileModel only on cache miss
    // ...
    return true;
}
```

实际执行落点仍是 CPU plugin 的 `PluginExecuteRaw::init()`：它按相同 key 查 resident executor，命中就复用，否则保持原行为现场 compile。

```cpp
// source/backend/qnn/backend/QNNBackend.cpp:1040
bool init(CPUKernelContext* ctx) override {
    if (QNN::gContext.deviceHandle == nullptr){
        QNN::createQnnContext();
    }
    auto path = MNNFilePathConcat(ctx->dir_path(), ctx->getAttr("path")->s()->str());
    // ...
    mRawExecutor = findRawGraphExecutor(path, binaryOffset, binarySize, allGraphName);
    if (mRawExecutor) {
        return true;
    }
    mRawExecutor.reset(new RawExecutorWrapper());
    return mRawExecutor->compileModel(path, binaryOffset, binarySize, allGraphName);
}
```

## 7. 逐文件修改版本

| 文件 | 修改版本摘要 | Review 重点 |
|---|---|---|
| `include/MNN/QNNPrefetch.hpp` | 新增 QNN raw graph prefetch public API。 | public header 与 `MNN_WITH_PLUGIN` / `MNN_QNN` 构建组合是否匹配。 |
| `CMakeLists.txt` | 将 `QNNPrefetch.hpp` 加入 public headers。 | 非 QNN 构建下 header 暴露和符号实现一致性。 |
| `source/backend/qnn/backend/QNNBackend.cpp` | 增加 QNN symbol load guard、resident raw graph pool、prefetch/release/has API、cache-key 路径规范化、`PluginExecuteRaw` 复用 resident executor、raw executor invoke mutex。 | graph alias/pin 语义、resident pool 生命周期、锁粒度、compileModel 在锁内执行是否可接受。 |
| `express/Executor.cpp` | `RuntimeManager::setHintPtr()` 改为写入当前 `RuntimeManager` 自己持有的 runtimes。 | 避免双 pipeline 下把 KV meta 写到全局 current executor。 |
| `transformers/llm/engine/include/llm/BatchScheduler.hpp` | `Chunk` 增加 `pipelineId/segmentIndex`，新增 `scheduleWave()` 和 request->pipeline 记录。 | API 兼容性、pipeline assignment 生命周期。 |
| `transformers/llm/engine/src/BatchScheduler.cpp` | dual mode 按 request group 和 segment 构造 ordered chunks；`scheduleWave()` 返回同一 segment chunks；release request 时清 pipeline assignment。 | 四请求 A/B/C/D 分组、decode pipeline persistence、pending chunk commit 顺序。 |
| `transformers/llm/engine/include/llm/DualPipelineScheduler.hpp` | `GraphRequest` 承载 QNN metadata，`GraphRecord` 增加 `activeUseCount`，`onGraphLoad` 返回 `bool`，新增 readiness wait。 | load failure 传播、共享 graph 使用计数、等待停止语义。 |
| `transformers/llm/engine/src/DualPipelineScheduler.cpp` | worker 状态机处理 prefetch/load/complete/release/evict；callback 成功后才发布 resident；shared graph release。 | load/complete/release 排序、pinned graph release 语义。 |
| `transformers/llm/engine/src/DualPipelineGraph.cpp` | 从 `Session::getPipelineInfo()` 只读 snapshot；解析 QNN attrs；按请求组选择 bucket，并按 binary asset 生成 resource id。 | 不触发 live resize；`path + allGraphName` 判断是否过宽。 |
| `transformers/llm/engine/include/llm/llm.hpp` | 增加 request thread、双 pipeline runtime 数组、module pool、graph snapshot/cursor/graph records。 | 生命周期 reset、并发资源隔离。 |
| `transformers/llm/engine/src/llm.cpp` | 配置入口、双 runtime 准备、wave 并发执行、prefetch/QNN callback、request 完成清理、空 logits guard。 | 线程异常/失败处理、非 dual 行为保持、batch shape 失败时清理路径。 |
| `transformers/llm/engine/src/kvmeta.hpp` | `BatchKVMeta::releaseKV()` 对不存在 req meta 直接 no-op。 | 避免错误 pipeline 的空 meta 触发 release callback。 |
| `test/CMakeLists.txt` | `MNN_BUILD_LLM=OFF` 时排除 `test/llm`；ON 时把 LLM scheduler/graph 源编进 `run_test.out`。 | 默认测试构建不引入 LLM-only 符号。 |
| `test/llm/BatchSchedulerTest.cpp` | 新增 schedule wave、pipeline persistence、split count clamp 等测试。 | 是否覆盖用户要求的双请求/四请求非重叠调度。 |
| `test/llm/DualPipelineGraphTest.cpp` | 新增 QNN metadata without type 测试。 | metadata-based QNN 识别是否误伤非 QNN plugin。 |
| `test/llm/DualPipelineSchedulerTest.cpp` | 新增 graph load failure/readiness 测试；callback 更新为返回 bool。 | failed load 不应被标记 resident 或保留 active use。 |

## 8. Review 重点和剩余风险

### 8.1 当前还不是 attention/FFN stage-level scheduler

当前 dual execution 的并发单位是两个 `Module::onForward()`。它没有新增 tensor dependency、stage plan、command range executor，也没有把 transformer block 拆成 CPU/OpenCL attention stage 与 QNN FFN stage。后续如果要做到真正 stage-level 遮盖，需要新增 stage plan、stage ready queue 和部分图执行接口，不能只继续扩 `BatchScheduler::Chunk`。

### 8.2 CPU/OpenCL resize prefetch 仍是安全 no-op

`onPrefetchResize` 当前故意不调用 live resize。原因是 `Session::resize()` / `Pipeline::encode()` / `Execution::onResize()` 会修改 live runtime、tensor allocator 和 execution 状态。除非有隔离 runtime/scratch backend，否则 prefetch worker 不能并发触碰主执行 session。

### 8.3 QNN public header 的链接语义

`QNNPrefetch.hpp` 已进入 public headers，但实现主要在 `QNNBackend.cpp`。当前 `#ifndef MNN_WITH_PLUGIN` 有 no-op stub；review 时仍要确认 `MNN_QNN=OFF`、`MNN_WITH_PLUGIN=ON/OFF`、静态库/动态库组合下外部 include + link 行为一致。

### 8.4 QNN graph alias 与 pin 是 record 级语义

`graphId` 现在由 binary path/offset/size/allGraphName 构成，shape variant 保留在 `targetGraphName/shapeIndex`。`unpinAfterRelease` 仍会影响共享 `RawGraphRecord`，不是单个 alias 私有状态。

### 8.5 `buildQnnGraphRequests()` 的 QNN 识别变宽

现在只要 plugin 有 `path + allGraphName` metadata，即使 type/backend 字符串不带 QNN，也会当作 QNN plugin。它解决了已有导出模型 type 不明显的问题，但如果其他 plugin 复用这两个 attr，会被误判。

### 8.6 per-request op cursor 仍是顺序窗口

`mDualPipelineRequestOpCursor` 仍只服务 CPU/OpenCL ordered prefetch window；当前 full-module forward 会准备全部 QNN binary。它不是 DAG cursor，也不能证明 stage dependency 调度正确。

### 8.7 设备 batch 真执行依赖导出 shape 覆盖

使用 `qnn_fixed_s1_8_128` 的 ADB 三请求真实 batch 已跑通。临时 runtime trace 证明 decode wave 中两请求 pipeline 的 30 层均选择 shape index 1 / bucket 8，单请求 pipeline 的 30 层均选择 shape index 2 / bucket 1；移除 trace 后 clean run 仍无 QNN 1002/1100/no-logits。

## 9. 验证记录

本轮清理和文档更新后的 fresh verification：

```bash
# 本地执行
rtk g++ -std=c++11 -I. -Iinclude -Itransformers/llm/engine/include -Itransformers/llm/engine/src -c transformers/llm/engine/src/BatchScheduler.cpp -o /tmp/BatchScheduler.o
rtk g++ -std=c++11 -I. -Iinclude -Itransformers/llm/engine/include -c test/llm/BatchSchedulerTest.cpp -o /tmp/BatchSchedulerTest.o
rtk g++ -std=c++11 -I. -Iinclude -Iexpress -Isource -Itools -Ischema/current -Itransformers/llm/engine/include -Itransformers/llm/engine/src -I3rd_party/flatbuffers/include -I3rd_party/rapidjson/include -c transformers/llm/engine/src/llm.cpp -o /tmp/llm.o
rtk g++ -std=c++11 -I. -Iinclude -Iexpress -Isource -Ischema/current -I3rd_party/flatbuffers/include -I3rd_party/rapidjson/include -c express/Executor.cpp -o /tmp/Executor.o
rtk g++ -std=c++11 -DMNN_WITH_PLUGIN -DMNN_QNN_ENABLED=1 -DENABLE_QNN_ONLINE_FINALIZE -I. -Iinclude -Isource -Ischema/current -I3rd_party/flatbuffers/include -I3rd_party/half -Isource/backend/qnn/backend -Isource/backend/qnn/convertor -I/root/qnn/include/QNN -c source/backend/qnn/backend/QNNBackend.cpp -o /tmp/QNNBackend.o
rtk cmake --build /tmp/mnn-pp-test-llm --target run_test.out -j2
rtk /tmp/mnn-pp-test-llm/run_test.out llm
rtk cmake --build /tmp/mnn-pp-test-llm --target llm -j2
rtk proxy git diff --check
```

结果：上述命令均 exit 0；`run_test.out llm` 显示 `25/25` passed。Android `max_new_tokens=2` 三请求 clean probe 退出 0，三个请求均有输出且无 QNN validate/execute/prefetch/no-logits 错误。`QNNBackend.cpp` standalone compile 仍有既有 `FUNC_PRINT(size)` format warning，不是本轮新增。

## 10. 小结

- 入口是 `Llm::configureDualPipelineMode()`，它把 config、scheduler、prefetch worker 和 QNN callbacks 串起来。
- 调度核心是 `BatchScheduler::scheduleWave()`，它返回同一 segment 的 pipeline 0/1 chunks。
- 执行核心是 `Llm::generate(batch)` dual 分支，两个 pipeline 使用独立 runtime/module pool 并发 `onForward()`。
- 图预取核心是 `DualPipelineGraph` 只读 snapshot 加 `DualPipelineScheduler` worker；CPU/OpenCL resize 仍不碰 live `onResize`。
- QNN 预加载核心是 `QNNBackend.cpp` 的 resident raw graph pool，实际复用点仍在 CPU plugin `PluginExecuteRaw`。
- 最需要 review 的边界是 public QNN API 构建组合、QNN graph alias/pin 语义、metadata-based QNN 识别范围，以及后续 stage-level attention/FFN 切图还未实现。
