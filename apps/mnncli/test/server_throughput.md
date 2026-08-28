# Android Server 三模式吞吐测试

`server_throughput.py` 用固定 GSM8K JSONL 测试 Android `mnncli serve`。Android GPU 在本文中专指 MNN OpenCL。主要比较同一 backend 内 `single_request`、`continuous_batch` 和 `dual_pipeline` 的相对吞吐，不把 Host 模型与 QNN Plugin 导出物之间的差值解释为纯 kernel 加速。

脚本只依赖 Python 标准库。设备模型和 QNN runtime 原地复用，不会复制或删除。

## 准备

数据集必须是 JSONL，每行至少包含 `id` 和非空 `prompt`。推荐使用既有 `gsm8k_eval.py prepare` 生成的固定 100 题数据。

按 `apps/mnncli/AGENTS.md` 使用标准两阶段脚本增量构建。不要加 `--clean`，并行度最多为 8：

```bash
cd apps/mnncli
MNN_BUILD_JOBS=8 ./build.sh --android-service
```

默认二进制为 `apps/mnncli/build_mnncli_android_service/mnncli`。设备默认复用：

| 用途 | 路径 |
|---|---|
| CPU/OpenCL Host 模型 | `/data/local/tmp/Qwen3-1.7B-MNN-int4` |
| QNN Plugin 模型 | `/data/local/tmp/qnn_fixed_s1_8_128` |
| QNN runtime | `/data/local/tmp/mnn-qnn` |
| 临时 staging | `/data/local/tmp/mnn-server-throughput` |

脚本读取设备已有 `config.json` / `config_qnn.json`，保留模型字段，并写入正确的绝对 `base_dir`。生成配置统一使用 greedy、`precision=low`、`memory=low`、`power=high`。CPU/OpenCL 使用 4 线程，QNN 使用 1 线程；continuous/dual 开 packed attention；dual 使用 `split_count=2`，QNN dual 另设 resident=40、prefetch=2。resident 是硬预算；活跃图受保护，target 图由 LRU 管理，Eagle FC/draft 当前 bucket 只在 wave 内 pin，并在 bucket 切换和 batch 结束后显式释放，避免跨 wave 累积固定图耗尽容量。

## Dry Run

dry-run 检查本地二进制和数据集，打印九份配置、执行顺序和空间预算。它不会创建 ADB client，也不会连接设备或写结果目录。

```bash
python3 apps/mnncli/test/server_throughput.py matrix \
  --adb-serial 192.168.124.101:47954 \
  --binary apps/mnncli/build_mnncli_android_service/mnncli \
  --dataset /path/to/gsm8k_100.jsonl \
  --phase dev \
  --output-dir /path/to/results-dev \
  --dry-run
```

## 开发矩阵

dev 共 9 个 cell：`CPU / OpenCL / QNN` 乘以三种 mode。每个 cell 先 warmup 4 请求，再用前 8 题、16 completion token、C4 测量两个 wave。dev 开启 `MNN_ACCEPTANCE_TRACE=1`，自动检查：

- single 的请求执行区间严格串行；
- continuous 出现四请求真实 batch；
- dual 使用 lane 0/1；
- CPU/OpenCL dual 的 `max_qnn=0`；
- QNN dual 有 QNN stage，且 `max_host<=1`、`max_qnn<=1`；
- 无 `INTERNAL_ERROR`、crash、QNN graph/device 或 OpenCL kernel/runtime 错误。

```bash
python3 apps/mnncli/test/server_throughput.py matrix \
  --adb-serial 192.168.124.101:47954 \
  --binary apps/mnncli/build_mnncli_android_service/mnncli \
  --dataset /path/to/gsm8k_100.jsonl \
  --phase dev \
  --output-dir /path/to/results-dev
```

dev 只验证 HTTP、配置和调度，不能形成性能结论。

## QNN Dual C6/C8

`dual-bench` 按顺序测试 C6 和 C8。这里 C6 是总并发 6，双 lane 分为
3+3；C8 是总并发 8，双 lane 分为 4+4。两组不会同时启动。

`full` 小规模验收每个 cell 精确使用前 20 条数据、每条最多 512 token。
由于 20 不能被 6 或 8 整除，C6 执行 `6+6+6+2`，C8 执行
`8+8+4`；最后一个 partial wave 会单独记录。任一 wave 失败后会立即停止当前
cell，工具仍会执行另一个 cell，最后在汇总中统一标记失败。

数据字段默认使用 `prompt`。当 QNN 导出不能容纳 few-shot 长前缀时，可用
`--prompt-field question` 测试同一 JSONL 的短题面。实际字段会记录为结果元数据
`dataset_prompt_field`，原始 JSONL 和其他字段不修改。

`--disable-resource-guard` 会关闭模型加载、warmup 和测量期间的温度、电量、可用
内存、进程 RSS 与 ADB 失败自动停止，同时跳过 full 的冷却门和 32°C 起始温度
限制。默认不启用该选项；启用时结果元数据记录
`resource_guard_enabled=false`，且不生成 `.resources.jsonl`。

```bash
python3 apps/mnncli/test/server_throughput.py dual-bench \
  --adb-serial 192.168.124.101:47954 \
  --binary apps/mnncli/build_mnncli_android_service/mnncli \
  --dataset /path/to/gsm8k_100.jsonl \
  --prompt-field question \
  --phase full \
  --output-dir /path/to/results-dual-full
```

## 最终矩阵

full 对每个 backend/mode 分别启动独立的 C1 和 C4 server，共 18 个 cell：

- C1：warmup 1 题，测前 20 题，每题最多 512 token；
- C4：warmup 4 题，测完整 100 题，每题最多 512 token，共 25 个 wave；
- CPU mode 顺序为 single/continuous/dual；OpenCL 向前轮换一次；QNN 再轮换一次；
- server load、模型初始化、warmup 和温控等待不进入测量 wall time。

warmup 后，full 要求连续一分钟的 `PhoneTemp` 都不超过 32.0 C 且波动不超过 1.0 C。测量期间每 5 秒检查一次；超过 42.0 C，或设备未充电且电量不高于 10%，都会立即停止本 cell。漂移使用开头和结尾最多五个且互不重叠的 wave；绝对值超过 10% 时，冷却后用新的 server 重跑，最多三次。传入 `--disable-drift-retry` 后仍记录漂移，但不拒绝、冷却或重跑该 cell。

```bash
python3 apps/mnncli/test/server_throughput.py matrix \
  --adb-serial 192.168.124.101:47954 \
  --binary apps/mnncli/build_mnncli_android_service/mnncli \
  --dataset /path/to/gsm8k_100.jsonl \
  --phase full \
  --output-dir /path/to/results-full \
  --scorer /path/to/transformers/llm/eval/gsm8k_eval.py
```

`--scorer` 可省略。提供后，每个 cell 使用与其 limit 相同的输入子集调用旧 `gsm8k_eval.py score`，避免 C1 被误报为缺失 80 题。server 会对完整 few-shot prompt 再应用 chat template，因此 accuracy 只用于检查输出是否损坏，不与旧 `llm_demo` 的 46/100 直接比较。

## 单独 Load

`load` 可测试已启动的任意 OpenAI-compatible server。每个 wave 的 worker 先通过 barrier，再同时 POST 非流式 `/v1/chat/completions`。请求数和 warmup 数必须能被 concurrency 整除；脚本拒绝部分 wave。

```bash
python3 apps/mnncli/test/server_throughput.py load \
  --base-url http://127.0.0.1:18080 \
  --dataset /path/to/gsm8k_100.jsonl \
  --limit 8 \
  --max-tokens 16 \
  --concurrency 4 \
  --warmup 4 \
  --output /path/to/cell.json
```

输出文本只写主机的 `cell.responses.jsonl`。每个 wave 完成后立即追加响应，并同步写入 `cell.waves.jsonl`；中断时已完成 wave 仍然保留。其中响应保留旧 scorer 需要的 `id`、`response` 和 `status`：生成 token 达到请求上限时为 `max_tokens_finished`，否则为 `normal_finished`。HTTP、超时、空输出或零 usage 记录为 `error`。

## 结果与判定

每次 attempt 生成：

- `<cell>-attemptN.json`：metadata、aggregate 和 wave 统计；
- `<cell>-attemptN.responses.jsonl`：逐请求响应与错误；
- `<cell>-attemptN.waves.jsonl`：逐 wave checkpoint；
- `<cell>-attemptN.server.log.gz`：压缩 server 日志；
- `configs/<cell>-attemptN.json`：实际部署配置；
- 可选 `.score.json` 和对应 scorer 输入子集。

矩阵最终生成 `summary.json`、`summary.csv` 和 `summary.md`。核心字段包括：

- `aggregate_completion_tokens_per_s`：成功请求的实际 completion token / 整段测量 wall time；
- requests/s、prompt/completion token 总数和 wall time；
- 请求 latency p50/p90/p95/max；
- wave token/s p50/p90、首尾不重叠窗口均值与漂移；
- HTTP、timeout、transport、protocol、空输出和零 usage 数；
- 开始/结束温度以及测量期间温度采样；
- config、模型、二进制和 dataset SHA-256；
- 同 backend、同 concurrency 下相对 single 的 speedup。

失败请求不计入 token、请求延迟或成功请求计数，但 wall time 仍是整个测量区间；只要有一个失败请求，整个 cell 即失败并停止矩阵。dev 必须完成 8/8，full C1 必须完成 20/20，full C4 必须完成 100/100。

## 清理和磁盘

开始前脚本记录 `/data` 的 `df` 与三个现有目录的 `du`，并要求至少 10 GiB 可用。它只 push 一个 `mnncli-throughput`，每次只部署一个小配置。设备 staging 硬限制 64 MiB，主机结果硬限制 256 MiB。

每个 cell 启动前先删除同名旧 PID 文件，只 kill 本次新进程记录的 PID，并只移除本次 `adb forward`、PID 文件和临时配置；不调用 `pkill`，不终止其他推理进程。日志和响应始终留在主机；结束后设备 staging 仅保留一个二进制和小型 `throughput-summary.json`。现有 Host/QNN 模型、QNN runtime 和其他用户目录不会被删除或修改。
