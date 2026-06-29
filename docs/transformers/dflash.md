# DFlash 使用说明

这份说明描述如何把 DFlash 从导出一路跑到 Android 设备上的 `llm_demo`。

## 1. 导出

在 `transformers/llm/export` 下执行。下面示例会导出 MNN 模型，并使用 HQQ 量化：

```bash
cd transformers/llm/export
python llmexport.py \
  --path /path/to/base_model \
  --dflash_path /path/to/dflash_model \
  --export mnn \
  --dst_path ./model_dflash \
  --no_thinking \
  --hqq
```

主要参数说明：

- `--path`：目标模型目录。
- `--dflash_path`：DFlash draft 模型目录。
- `--export mnn`：导出 MNN 模型。
- `--dst_path`：导出产物目录。
- `--no_thinking`：仅对 Qwen3 生效，写入 `enable_thinking=false` 的 chat template 上下文。
- `--hqq`：使用 HQQ 量化。

如果导出 [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16)，需要保留 `--no_thinking`。该参数会让运行时默认使用 `enable_thinking=false` 的 chat template，并在 `llm_config.json` 的 `jinja.context` 中写入：

```json
{
  "enable_thinking": false
}
```

使用更高精度导出通常有助于提升平均接受长度，相关参数示例：

```bash
--quant_bit 16 --lm_quant_bit 16 --embed_bit 16
```

导出完成后，目录里通常会有：

- `config.json`
- `llm_config.json`
- `llm.mnn`
- `llm.mnn.weight`
- `dflash.mnn`
- `dflash.mnn.weight`

其中 `config.json` 会包含 DFlash 运行时字段(使用贪心解码通常可以获得更好的平均接受长度, 需要手动修改采样配置)，`llm_config.json` 会包含模型结构和 chat template 相关字段。

## 2. 配置含义

MNN 模型配置 `config.json` 需要包含下面配置项：

- `speculative_type`：推测解码类型，需要设置为 `dflash`。
- `dflash_model`：DFlash draft 模型文件路径，通常是 `dflash.mnn`。
- `dflash_block_size`：DFlash block 总长度，可以**近似**理解为每次打的草稿长度。
- `dflash_mask_token_id`：DFlash 训练和推理使用的 mask token id。
- `dflash_target_layer_ids`：目标模型中提供给 DFlash draft 模型的 hidden states 层编号。
- `dflash_ddtree_topk`：DDTree 每层候选宽度的可选上限。默认 `0` 表示不单独限制，按 budget 决定。
- `dflash_ddtree_budget`：DDTree 的非根节点预算 `B`。导出后的 DFlash 配置默认会把它写成 `0`；如果不设置它或显式设为 `0`，runtime 会保持线性 DFlash。

`dflash_block_size` 这里表示 **一个 block 的总长度**，不是“额外 draft 个数”。
DDTree 的 `dflash_ddtree_budget` 表示 **树中非根节点数量**，对应论文里的 budget `B`。
当前实现里，运行时每一层的展开宽度会取：

```text
topk = min(vocab_size, dflash_ddtree_budget)                      # dflash_ddtree_topk <= 0
topk = min(vocab_size, dflash_ddtree_budget, dflash_ddtree_topk) # dflash_ddtree_topk > 0
```

也就是说：

- block 第 1 个 token 是当前 token
- 后面 `block_size - 1` 个 token 是 draft token

运行时会把 draft 长度理解成：

```text
draft_len = dflash_block_size - 1
```

## 3. 编译 Android

先在 Android 工程里编译 64 位包：

```bash
cd project/android
mkdir -p build_64
../build_64.sh -DMNN_BUILD_LLM=true -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true  -DMNN_USE_LOGCAT=true  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

编译成功后，`llm_demo` 和 `libllm.so` 会在 `project/android/build_64` 相关目录下生成。

## 4. 推送到设备

先建好设备目录：

```bash
adb shell "mkdir -p /data/local/tmp/mnn-dflash"
```

然后把运行所需文件推上去, 注意文件路径：

```bash
adb push project/android/build_64/spec_eval /data/local/tmp/mnn-dflash/
adb push project/android/build_64/llm_demo /data/local/tmp/mnn-dflash/llm_demo
adb push project/android/build_64/libllm.so /data/local/tmp/mnn-dflash/libllm.so
adb push project/android/build_64/libMNN.so /data/local/tmp/mnn-dflash/libMNN.so
adb push project/android/build_64/libMNN_Express.so /data/local/tmp/mnn-dflash/libMNN_Express.so
adb push /path/to/model_dflash /data/local/tmp/
```

## 5. 设备运行

进入设备目录后,交互式运行：

```bash
adb shell
cd /data/local/tmp/mnn-dflash
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
./llm_demo /path/to/model_dflash/config.json
```

## 6. DFlash 评测

推测解码统计可以使用 `spec_eval` 工具, 这里的模板和测试数据需要注意路径：

```bash
./spec_eval /path-to-model/config.json samples.txt \
  --template-file=prompt_templates.json \
  --template-name=gsm8k \
  --limit=10 \
  --max-tokens=512
```

主要参数说明：

- `/path-to-model/config.json`：导出模型目录下的运行配置。
- `samples.txt`：评测样本文件，每行一个输入样本。
- `--template-file`：prompt template 配置文件。
- `--template-name`：使用的 template 名称，例如 `gsm8k`。
- `--limit`：最多评测的样本数。
- `--max-tokens`：每个样本最多生成的新 token 数。

实测结果(参考测试样本，问答模板在根目录)：
```text
✗ ./spec_eval /data/HUGGINGFACE/Qwen3-4B-DFlash-MNN/config.json ../gsm8k_eval_samples.txt \
    --template-file=../prompt_templates.json \
    --template-name=gsm8k \
    --limit=3 --no-thinking
Can't open file:/sys/devices/system/cpu/cpufreq/boost/affected_cpus
Can't open file:/sys/devices/system/cpu/cpufreq/ondemand/affected_cpus
CPU Group: [ 95  17  83  55  118  27  93  65  5  37  75  47  19  85  57  29  108  67  7  39  101  10  77  49  111  20  87  59  121  30  97  69  33  112  21  88  122  31  98  41  104  13  51  114  23  61  1  124  9  71  43  106  15  81  53  116  25  91  63  3  126  35  73  45  94  117  26  92  64  4  127  36  74  46  109  18  84  56  119  28  54  66  6  38  100  76  48  110  86  58  120  96  68  8  102  11  105  40  103  12  79  50  113  22  89  60  0  123  32  99  70  42  78  14  80  52  115  24  90  62  2  125  34  72  44  107  16  82 ], 1500000 - 2900000
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0

================================================
     Speculative Decoding Evaluation
================================================

Config: /data/HUGGINGFACE/Qwen3-4B-DFlash-MNN/config.json
Data: ../gsm8k_eval_samples.txt
Loaded 3 test samples

Template: gsm8k (../prompt_templates.json)
Loading model...
Model loaded in 22871.4 ms
Draft length: 15

=== Running Evaluation ===

[   3/3] (100%) | tokens= 488, steps=182, avg_accept=4.99, time=81842.1ms
================================================
              Evaluation Results
================================================

--- Statistics ---
Total samples:          3
Total prompt len:       249
Total decoding steps:   182
Total draft tokens:     2912
Total accepted tokens:  909
Total draft time:       45328.23 ms
Total target time:      96878.63 ms

--- Key Metrics ---
Average Accept Length:  4.995 tokens/step
Accept Rate:            31.216%
Compression Ratio:      3.204 tokens/step
Avg Draft Time:         249.056 ms/step
Avg Target Time:        532.300 ms/step
Theoretical Speedup:    3.403x

--- Accept Length Frequency ---
Accept Length   1: 24
Accept Length   2: 28
Accept Length   3: 29
Accept Length   4: 25
Accept Length   5: 18
Accept Length   6: 12
Accept Length   7: 5
Accept Length   8: 3
Accept Length   9: 12
Accept Length  10: 5
Accept Length  11: 7
Accept Length  12: 7
Accept Length  13: 2
Accept Length  14: 1
Accept Length  15: 2
Accept Length  16: 2

================================================
```
## 7. DDTree 运行时配置

当前仓库里的 DFlash 有两种运行方式：

- 线性 DFlash：每轮只验证一条长度为 `block_size` 的草稿路径。
- DDTree DFlash：先用 DFlash draft 模型给出每个未来位置的分布，再把这些分布编译成一棵 draft tree，最后由 target 一次性验证整棵树。

### 7.1 怎么开启

在 `config.json` 里设置：

```json
{
  "speculative_type": "dflash",
  "dflash_block_size": 16,
  "dflash_ddtree_budget": 15,
  "dflash_ddtree_topk": 10
}
```

其中：

- `dflash_ddtree_budget = 0` 或缺省：保持线性 DFlash。
- `dflash_ddtree_budget > 0`：启用 DDTree，如`dflash_ddtree_budget = 15` 表示这棵树最多扩展 `15` 个非根节点；target verify 时看到的是“根节点 + 若干候选 token 节点”的树结构。
- `dflash_ddtree_topk = 0` 时，每一层的候选宽度按 `topk = min(dflash_ddtree_budget, vocab_size)` 计算；设置 `"dflash_ddtree_topk": 10`，那么每一层最多只保留 `top10` 个候选

### 7.2 对比
```bash
# 不开启 DDTree DFlash
➜ ./spec_eval /data/HUGGINGFACE/Qwen3-4B-DFlash-MNN/config-low.json ../gsm8k_eval_samples.txt --template-file=../prompt_templates.json \
    --template-name=gsm8k \
    --limit=10
Can't open file:/sys/devices/system/cpu/cpufreq/boost/affected_cpus
Can't open file:/sys/devices/system/cpu/cpufreq/ondemand/affected_cpus
CPU Group: [ 95  17  83  55  118  27  93  65  5  37  75  47  19  85  57  29  108  67  7  39  101  10  77  49  111  20  87  59  121  30  97  69  33  112  21  88  122  31  98  41  104  13  51  114  23  61  1  124  9  71  43  106  15  81  53  116  25  91  63  3  126  35  73  45  94  117  26  92  64  4  127  36  74  46  109  18  84  56  119  28  54  66  6  38  100  76  48  110  86  58  120  96  68  8  102  11  105  40  103  12  79  50  113  22  89  60  0  123  32  99  70  42  78  14  80  52  115  24  90  62  2  125  34  72  44  107  16  82 ], 1500000 - 2900000
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0

================================================
     Speculative Decoding Evaluation
================================================

Config: /data/HUGGINGFACE/Qwen3-4B-DFlash-MNN/config-low.json
Data: ../gsm8k_eval_samples.txt
Loaded 10 test samples

Template: gsm8k (../prompt_templates.json)
Loading model...
Model loaded in 22397.7 ms
Draft length: 15

=== Running Evaluation ===

[  10/10] (100%) | tokens= 218, steps=688, avg_accept=4.76, time=30888.9ms
================================================
              Evaluation Results
================================================

--- Statistics ---
Total samples:          10
Total prompt len:       858
Total decoding steps:   688
Total draft tokens:     11008
Total accepted tokens:  3275
Total draft time:       159396.03 ms
Total target time:      330669.69 ms

--- Key Metrics ---
Average Accept Length:  4.760 tokens/step
Accept Rate:            29.751%
Compression Ratio:      3.361 tokens/step
Avg Draft Time:         231.680 ms/step
Avg Target Time:        480.625 ms/step
Theoretical Speedup:    3.212x

--- Accept Length Frequency ---
Accept Length   1: 96
Accept Length   2: 127
Accept Length   3: 95
Accept Length   4: 83
Accept Length   5: 71
Accept Length   6: 43
Accept Length   7: 37
Accept Length   8: 26
Accept Length   9: 33
Accept Length  10: 21
Accept Length  11: 17
Accept Length  12: 14
Accept Length  13: 4
Accept Length  14: 9
Accept Length  15: 4
Accept Length  16: 8

================================================


# 开启 DDTree DFlash
➜ ./spec_eval /data/HUGGINGFACE/Qwen3-4B-DFlash-MNN/config-d512-t10.json /workspace/code/mnn-dflash/MNN_INFER/gsm8k_eval_samples.txt \
    --template-file=/workspace/code/mnn-dflash/MNN_INFER/prompt_templates.json \
    --template-name=gsm8k \
    --limit=10
Can't open file:/sys/devices/system/cpu/cpufreq/boost/affected_cpus
Can't open file:/sys/devices/system/cpu/cpufreq/ondemand/affected_cpus
CPU Group: [ 95  17  83  55  118  27  93  65  5  37  75  47  19  85  57  29  108  67  7  39  101  10  77  49  111  20  87  59  121  30  97  69  33  112  21  88  122  31  98  41  104  13  51  114  23  61  1  124  9  71  43  106  15  81  53  116  25  91  63  3  126  35  73  45  94  117  26  92  64  4  127  36  74  46  109  18  84  56  119  28  54  66  6  38  100  76  48  110  86  58  120  96  68  8  102  11  105  40  103  12  79  50  113  22  89  60  0  123  32  99  70  42  78  14  80  52  115  24  90  62  2  125  34  72  44  107  16  82 ], 1500000 - 2900000
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0

================================================
     Speculative Decoding Evaluation
================================================

Config: /data/HUGGINGFACE/Qwen3-4B-DFlash-MNN/config-d512-t10.json
Data: /workspace/code/mnn-dflash/MNN_INFER/gsm8k_eval_samples.txt
Loaded 10 test samples

Template: gsm8k (/workspace/code/mnn-dflash/MNN_INFER/prompt_templates.json)
Loading model...
Model loaded in 22498.4 ms
Draft length: 15

=== Running Evaluation ===

Warning: module need new clone, cloning now.
[  10/10] (100%) | tokens= 512, steps=796, avg_accept=5.97, time=1311848.9ms
================================================
              Evaluation Results
================================================

--- Statistics ---
Total samples:          10
Total prompt len:       858
Total decoding steps:   796
Total draft tokens:     408348
Total accepted tokens:  4752
Total draft time:       205511.66 ms
Total target time:      13042404.00 ms

--- Key Metrics ---
Average Accept Length:  5.970 tokens/step
Accept Rate:            1.164%
Compression Ratio:      85.932 tokens/step
Avg Draft Time:         258.180 ms/step
Avg Target Time:        16384.930 ms/step
Theoretical Speedup:    5.877x

--- Accept Length Frequency ---
Accept Length   1: 103
Accept Length   2: 106
Accept Length   3: 110
Accept Length   4: 107
Accept Length   5: 73
Accept Length   6: 61
Accept Length   7: 27
Accept Length   8: 31
Accept Length   9: 20
Accept Length  10: 15
Accept Length  11: 9
Accept Length  12: 10
Accept Length  13: 3
Accept Length  14: 4
Accept Length  15: 5
Accept Length  16: 112

================================================
```

## 8. 论文引用

本仓库当前 DFlash / DDTree 路径主要参考下面两篇论文：

1. DFlash: Block Diffusion for Flash Speculative Decoding. arXiv:2602.06036.  
   链接：<https://arxiv.org/abs/2602.06036>

2. Accelerating Speculative Decoding with Block Diffusion Draft Trees. arXiv:2604.12989.  
   链接：<https://arxiv.org/abs/2604.12989>

如果需要对照实现细节，建议重点看：

- DFlash 论文中关于 block diffusion drafting、target hidden feature conditioning、block size 的部分；
- DDTree 论文中关于 budgeted draft tree、best-first tree construction、tree verification、posterior tree walk 的部分。
