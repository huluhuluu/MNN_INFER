# Sliding Window

本文说明当前 `feat/sliding-window` 分支中 LLM 和 Eagle3 推测解码的 sliding window 实现。这里的 sliding window 不只是改 Mask，而是会通过 `KVMeta` 裁剪后端实际保存的 KV cache。

## 1. 配置

相关配置有两个：

- `sliding_window`: LLM 的 KV cache 窗口大小。
- `eagle_sliding_window`: Eagle draft model 的 KV cache 窗口大小。

配置读取在 `transformers/llm/engine/src/llmconfig.hpp`：

```cpp
int sliding_window() const {
    return config_.value("sliding_window", 0);
}

int eagle_sliding_window() const {
    return config_.value("eagle_sliding_window", 0);
}
```

两者分开配置，是因为 LLM 和 Eagle draft model 使用两套 KV 信息：LLM 使用 `mMeta`，Eagle 使用 `mEagleMeta`。

## 2. KVMeta 控制 KV cache

MNN 通过 `KVMeta` 把 KV cache 的增删信息传给后端。几个关键字段：

- `previous`: 当前已经保存的 KV 长度。
- `add`: 本次 forward 要新增的 token 数。
- `remove`: 本次 forward 前要删除的旧 KV 数。
- `reserve` / `n_reserve`: 删除后还要保留的 KV 片段。

forward 后会调用 `KVMeta::sync()` 更新状态：

```cpp
void KVMeta::sync() {
    int revertNumber = 0;
    for (int i=0; i<n_reserve; ++i) {
        revertNumber += reserve[2*i+1];
    }
    previous = previous - remove + add + revertNumber;
    n_reserve = 0;
    reserve = nullptr;
    remove = 0;
    add = 0;
}
```

所以 sliding window 的核心是设置 `remove` 和 `reserve`，让后端只保留窗口内的 KV cache，而不是仅靠 Mask 屏蔽旧 token。

## 3. LLM sliding window

LLM 的裁剪逻辑在 `Llm::applySlidingWindowKVCache(size_t add)` 中：

```cpp
void Llm::applySlidingWindowKVCache(size_t add) {
    if (mMeta == nullptr) {
        return;
    }
    auto slidingWindow = mConfig->sliding_window();
    if (slidingWindow <= 0) {
        mMeta->add = add;
        return;
    }
    if (mMeta->remove > 0 || mMeta->n_reserve > 0 || mMeta->file_flag == KVMeta::PendingRead) {
        mMeta->add = add;
        return;
    }
    auto keep = add >= (size_t)slidingWindow ? 0 : (size_t)slidingWindow - add;
    if (mMeta->previous <= keep) {
        setKVCacheInfo(add, 0);
        return;
    }
    if (keep == 0) {
        setKVCacheInfo(add, mMeta->previous);
        return;
    }
    mMeta->reserveHost.resize(2);
    mMeta->reserveHost[0] = (int)(mMeta->previous - keep);
    mMeta->reserveHost[1] = (int)keep;
    setKVCacheInfo(add, mMeta->previous, mMeta->reserveHost.data(), 1);
}
```

这里的逻辑是：

1. `sliding_window <= 0` 时关闭 sliding window。
2. 本次 forward 会新增 `add` 个 token。
3. 旧 KV 最多保留 `sliding_window - add` 个。
4. 如果旧 KV 太长，就删除旧 KV，并用 `reserveHost` 保留尾部 `keep` 个 KV。

公式如下：

```text
keep = max(sliding_window - add, 0)
```

`forwardVec(...)` 在实际 forward 前调用这个函数：

```cpp
if (0 == mBlockSize) {
    applySlidingWindowKVCache(seq_len);
    auto attention_mask = gen_attention_mask(seq_len);
    auto position_ids = gen_position_ids(seq_len);
    auto res = forwardRaw(input_embeds, attention_mask, position_ids);
    return res;
}
```

分块 forward 时，每个 block 也会先更新 KV cache 信息：

```cpp
for (int i=0; i<blockNumber; ++i) {
    logits.clear();
    applySlidingWindowKVCache(blockSize);
    auto embed = embeddings[i];
    auto attention_mask = gen_attention_mask(blockSize);
    auto position_ids = gen_position_ids(blockSize);
    logits = forwardRaw(embed, attention_mask, position_ids);
}

if (blockRemain != 0) {
    logits.clear();
    applySlidingWindowKVCache(blockRemain);
    ...
}
```

### 3.1 LLM 的 Mask 对齐

KV cache 被裁剪后，Mask 的长度不能再按完整历史长度计算。代码通过 `pendingKVCacheLength(...)` 计算本次 forward 实际会看到的 KV 长度：

```cpp
size_t Llm::pendingKVCacheLength(size_t add, const std::shared_ptr<KVMeta>& meta) const {
    if (meta == nullptr) {
        return mContext->all_seq_len + add;
    }
    auto remove = ALIMIN(meta->remove, meta->previous);
    return meta->previous - remove + _kvMetaReserveSize(meta.get()) + add;
}
```

`gen_attention_mask(...)` 使用这个长度生成 Mask。`attention_type == "mix"` 时，还会把裁剪后的物理 KV 下标映射回原来的逻辑位置：

```cpp
auto add = mMeta != nullptr && mMeta->add > 0 ? mMeta->add : (size_t)seq_len;
auto pad = seq_len > add ? (size_t)seq_len - add : 0;
int kv_seq_len = (int)(pendingKVCacheLength(add) + pad);

const int sliding_window = mConfig->sliding_window();
const int past_kv_len = kv_seq_len - seq_len;
const int logical_past_start = mContext->all_seq_len - past_kv_len;

for (int i = 0; i < seq_len; i++) {
    const int query_pos = i + mContext->all_seq_len;
    for (int j = 0; j < kv_seq_len; j++) {
        const int key_pos = j < past_kv_len
            ? logical_past_start + j
            : mContext->all_seq_len + j - past_kv_len;
        bool is_allowed = (key_pos <= query_pos) && (key_pos > query_pos - sliding_window);
        sliding_attn_ptr[kv_seq_len * i + j] = is_allowed
            ? 0.0f
            : std::numeric_limits<float>::lowest();
    }
}
```

这样做的目的，是让裁剪后的 KV cache 仍然按原始 token 位置判断 causal 和 sliding window 可见性。

## 4. Eagle sliding window

Eagle draft model 有自己的 `mEagleMeta`。它的裁剪逻辑在 `EagleGeneration::applySlidingWindowKVCache(size_t add)`：

```cpp
void EagleGeneration::applySlidingWindowKVCache(size_t add) {
    if (mEagleMeta == nullptr) {
        return;
    }
    auto slidingWindow = mLlm->mConfig->eagle_sliding_window();
    auto remove = std::min(mEagleMeta->remove, mEagleMeta->previous);
    if (slidingWindow <= 0 || mEagleMeta->n_reserve > 0 || mEagleMeta->file_flag == KVMeta::PendingRead) {
        mEagleMeta->remove = remove;
        mEagleMeta->add = add;
        return;
    }
    auto previous = mEagleMeta->previous;
    auto stablePrevious = previous - remove;
    auto keep = add >= (size_t)slidingWindow ? 0 : (size_t)slidingWindow - add;
    if (stablePrevious <= keep) {
        mEagleMeta->remove = remove;
        mEagleMeta->add = add;
        return;
    }
    if (keep == 0) {
        mEagleMeta->remove = previous;
        mEagleMeta->reserve = nullptr;
        mEagleMeta->n_reserve = 0;
        mEagleMeta->add = add;
        return;
    }
    mEagleMeta->reserveHost.resize(2);
    mEagleMeta->reserveHost[0] = (int)(stablePrevious - keep);
    mEagleMeta->reserveHost[1] = (int)keep;
    mEagleMeta->remove = previous;
    mEagleMeta->reserve = mEagleMeta->reserveHost.data();
    mEagleMeta->n_reserve = 1;
    mEagleMeta->add = add;
}
```

Eagle 的特殊点是 tree decoding 会产生临时 KV。这里先合并已有的 `remove`，得到 `stablePrevious = previous - remove`，再从稳定历史里保留尾部窗口。

### 4.1 Eagle 的 Mask 对齐

Eagle 使用 `genAttentionMask(int seqLen)` 生成自己的 Mask。它同样使用 `pendingKVCacheLength(seqLen, mEagleMeta)` 得到实际 KV 长度，再用 `mEaglePastLen` 映射逻辑位置：

```cpp
auto kvSeqLen = (int)mLlm->pendingKVCacheLength(seqLen, mEagleMeta);
auto pastKvLen = kvSeqLen - seqLen;
auto logicalPastStart = mEaglePastLen - pastKvLen;
auto attentionMask = _Input({1, 1, seqLen, kvSeqLen}, NCHW, halide_type_of<float>());
auto ptr = attentionMask->writeMap<float>();
for (int i = 0; i < seqLen; ++i) {
    auto queryPos = mEaglePastLen + i;
    for (int j = 0; j < kvSeqLen; ++j) {
        auto keyPos = j < pastKvLen
            ? logicalPastStart + j
            : mEaglePastLen + j - pastKvLen;
        ptr[i * kvSeqLen + j] = keyPos > queryPos
            ? std::numeric_limits<float>::lowest()
            : 0.0f;
    }
}
```

这保证了 Eagle 的 Mask 和 `mEagleMeta` 裁剪后的 KV cache 长度一致。

### 4.2. Eagle tree decoding

`topkGenerate(...)` 中会扩展 Eagle token tree。tree 内部的可见性由 `TokenTree::getMask()` 给出，但 sliding window 后还需要把保留下来的历史 KV 拼到 tree Mask 前面：

```cpp
auto treeMask = tokenTree.getMask();
applySlidingWindowKVCache(inputEmbeds->getInfo()->dim[0]);
auto kvSeqLen = (int)mLlm->pendingKVCacheLength(inputEmbeds->getInfo()->dim[0], mEagleMeta);
auto historyLen = std::max(0, kvSeqLen - (int)treeMask[0].size());
auto attentionMask = getMask(treeMask, historyLen);
outputs = eagleForwardRaw({inputEmbeds, inputHidden, attentionMask, mTreePosition, mLlm->logitsAllIdx});
```

这里 `historyLen` 是当前还能看到的历史 KV 长度。`getMask(treeMask, historyLen)` 会生成包含历史 KV 和 tree token 的完整 Mask。

### 4.3 Eagle3 prefill 输入截断

Eagle3 第一次进入 `topkGenerate(...)` 时，会把完整 prompt 的 `inputEmbeds` 和 `hiddenStates` 传给 draft model，prefill 计算仍会按完整 prompt 展开。

滑动窗口可以在进入 Eagle draft model 前截断输入，只保留 `eagle_sliding_window` 范围内的尾部内容。

辅助函数：

```cpp
static inline VARP _sliceTail(VARP x, int axis, int keep) {
    auto info = x->getInfo();
    MNN_ASSERT(info != nullptr);
    MNN_ASSERT(axis >= 0 && axis < info->dim.size());
    auto start = info->dim[axis] - keep;
    std::vector<int> indices(keep);
    std::iota(indices.begin(), indices.end(), start);
    return _GatherV2(x, _var<int>(indices, {keep}), _Scalar<int>(axis));
}
```

实际截断逻辑：

```cpp
auto eagleWindow = mLlm->mConfig->eagle_sliding_window();
if (eagleWindow > 0 && inputEmbeds->getInfo()->dim[0] > eagleWindow) {
    inputEmbeds = _sliceTail(inputEmbeds, 0, eagleWindow);
    hiddenStates = _sliceTail(hiddenStates, 1, eagleWindow);
    if (inputIds.size() > static_cast<size_t>(eagleWindow)) {
        inputIds.erase(inputIds.begin(), inputIds.end() - eagleWindow);
    }
    inputEmbeds->readMap<void>();
    hiddenStates->readMap<void>();
}
```

这里同步处理三份输入：

- `inputEmbeds`: 保留输入 embedding 的尾部窗口。
- `hiddenStates`: 保留 hidden states 序列维的尾部窗口。
- `inputIds`: 保留 token id 的尾部窗口。

`readMap<void>()` 用于触发表达式执行，避免后续 draft model prefill 继续挂着完整长序列的上游表达式。


## 5. 注意事项

- `sliding_window <= 0` 或 `eagle_sliding_window <= 0` 时，对应 sliding window 关闭。
- `attention_type == "mix"` 时，LLM Mask 会同时生成 full attention 和 sliding attention 两路。
- Eagle tree decoding 会产生临时 tree KV，`mEagleRemove` 和 `mEagleMeta->remove` 用于下一轮清理这些临时 KV。
- 当前分支没有看到这些路径的单元测试。建议至少覆盖长 prompt、`eagle_sliding_window < prompt_len`、多轮 Eagle draft decode 和 `mix` attention。
