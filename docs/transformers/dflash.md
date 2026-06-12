# DFlash 使用说明

这份说明描述如何把 DFlash 从导出一路跑到 Android 设备上的 `llm_demo`。

## 1. 约定

- `base_model`：被增强的原始 LLM
- `dflash_path`：DFlash draft 模型目录
- `export_dir`：导出产物目录
- `speculative_type`：需要设置为 `dflash`

当前仓库里的 DFlash 运行时要求这些配置项至少可用：

- `dflash_model`
- `dflash_block_size`
- `dflash_mask_token_id`
- `dflash_target_layer_ids`

## 2. 导出

在 `transformers/llm/export` 下执行：

```bash
cd transformers/llm/export
python llmexport.py \
  --path /path/to/base_model \
  --dflash_path /path/to/dflash_model \
  --export mnn \
  --dst_path ./model_dflash
```

导出完成后，目录里通常会有：

- `config.json`
- `llm.mnn`
- `llm.mnn.weight`
- `dflash.mnn`
- `dflash.mnn.weight`

其中 `config.json` 会包含 DFlash 相关字段。

## 3. 配置含义

`dflash_block_size` 这里表示 **一个 block 的总长度**，不是“额外 draft 个数”。

也就是说：

- block 第 1 个 token 是当前 token
- 后面 `block_size - 1` 个 token 才是 draft token

所以运行时会把 draft 长度理解成：

```text
draft_len = dflash_block_size - 1
```

## 4. 编译 Android

先在 Android 工程里编译 64 位包：

```bash
cd project/android
mkdir -p build_64
../build_64.sh -DMNN_BUILD_LLM=true -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true  -DMNN_USE_LOGCAT=true  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

编译成功后，`llm_demo` 和 `libllm.so` 会在 `project/android/build_64` 相关目录下生成。

## 5. 推送到设备

先建好设备目录：

```bash
adb shell "mkdir -p /data/local/tmp/mnn-dflash"
```

然后把运行所需文件推上去：

```bash
adb push project/android/build_64/eagle_eval /data/local/tmp/mnn-dflash/
adb push project/android/build_64/llm_demo /data/local/tmp/mnn-dflash/llm_demo
adb push project/android/build_64/libllm.so /data/local/tmp/mnn-dflash/libllm.so
adb push project/android/build_64/libMNN.so /data/local/tmp/mnn-dflash/libMNN.so
adb push project/android/build_64/libMNN_Express.so /data/local/tmp/mnn-dflash/libMNN_Express.so
adb push /path/to/model_dflash/* /data/local/tmp/
```

如果你已经有 `project/android/updateTest.sh` 的完整推送流程，也可以直接复用它，再补上模型目录。

## 6. 设备运行

进入设备目录后运行：

```bash
adb shell "
cd /data/local/tmp/mnn-dflash &&
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH &&
./llm_demo ./model/config.json ./model/prompt.txt
"
```

如果只想看单轮生成，也可以直接传 prompt 文件。

## 7. DFlash 评测

如果要看推测解码统计，仓库里还有 `eagle_eval`：

```bash
./eagle_eval ./model/config.json ./eval_samples.txt --max-tokens=128 --min-avg-accept=3
```

这个工具主要用于本机评测，不是 Android 设备上的主运行入口。


