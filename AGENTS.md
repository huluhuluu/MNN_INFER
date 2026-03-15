# MNN (Mobile Neural Network) 项目上下文

## 项目概述

MNN 是阿里巴巴开源的高效轻量级深度学习框架，支持深度学习模型的推理与训练，专注于在端侧设备（手机/PC/嵌入式设备）上的高性能部署。

**当前版本**: 3.2.2

**主要特点**:
- **轻量性**: 主体功能无依赖，iOS 静态库约 12MB，Android core so 约 800KB
- **通用性**: 支持 TensorFlow、Caffe、ONNX、TorchScripts 模型格式
- **高性能**: 针对 ARM/x86 CPU 进行深度优化，支持 GPU (Metal/OpenCL/Vulkan/CUDA) 加速
- **跨平台**: 支持 iOS 8.0+、Android 4.3+、Linux、Windows、macOS

### 子项目
- **MNN-LLM**: 大语言模型运行方案，支持 Qwen、Baichuan、LLaMA 等主流 LLM
- **MNN-Diffusion**: Stable Diffusion 文生图模型运行方案

## 目录结构

```
MNN-profile/
├── include/MNN/          # 公共 API 头文件
│   ├── Interpreter.hpp   # 模型解释器主接口
│   ├── Tensor.hpp        # 张量定义
│   ├── MNNDefine.h       # 版本和宏定义
│   └── expr/             # 表达式模块头文件
├── source/               # 核心源代码
│   ├── backend/          # 各硬件后端实现
│   │   ├── cpu/          # CPU 后端
│   │   ├── metal/        # Metal (iOS/macOS)
│   │   ├── opencl/       # OpenCL
│   │   ├── vulkan/       # Vulkan
│   │   ├── cuda/         # CUDA
│   │   ├── opengl/       # OpenGL
│   │   ├── coreml/       # CoreML
│   │   ├── nnapi/        # Android NNAPI
│   │   └── hiai/         # 华为 NPU
│   ├── core/             # 核心模块
│   ├── math/             # 数学运算
│   ├── cv/               # 图像处理
│   ├── geometry/         # 几何变换
│   └── shape/            # 形状推理
├── express/              # MNN Express 模块 (动态图)
├── transformers/         # LLM 和 Diffusion 引擎
│   ├── llm/              # 大语言模型引擎
│   └── diffusion/        # 扩散模型引擎
├── tools/                # 工具集
│   ├── converter/        # 模型转换工具
│   ├── quantization/     # 量化工具
│   ├── train/            # 训练工具
│   ├── cv/               # OpenCV 兼容 API
│   └── audio/            # 音频处理 API
├── pymnn/                # Python 绑定
├── test/                 # 测试代码
├── benchmark/            # 性能基准测试
├── apps/                 # 移动端应用示例
│   ├── Android/          # Android 应用
│   └── iOS/              # iOS 应用
├── docs/                 # 文档源码
└── project/              # 跨平台编译配置
    ├── android/          # Android 编译脚本
    └── ios/              # iOS 编译脚本
```

## 构建和运行

### Linux/macOS 构建

```bash
# 基础构建
mkdir build && cd build
cmake ..
make -j$(nproc)

# 启用 LLM 支持
cmake .. -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true \
         -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
make -j$(nproc)

# 启用 OpenCL GPU 支持
cmake .. -DMNN_OPENCL=true

# 启用 CUDA 支持
cmake .. -DMNN_CUDA=true

# 启用训练功能
cmake .. -DMNN_BUILD_TRAIN=true

# 启用模型转换工具
cmake .. -DMNN_BUILD_CONVERTER=true
```

### Android 构建

```bash
cd project/android
mkdir build_64 && cd build_64
../build_64.sh "-DMNN_OPENCL=true -DMNN_ARM82=true -DMNN_LOW_MEMORY=true"
```

### iOS 构建

```bash
sh package_scripts/ios/buildiOS.sh "-DMNN_ARM82=true -DMNN_LOW_MEMORY=true"
```

### 主要 CMake 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `MNN_BUILD_SHARED_LIBS` | 构建动态库 | ON |
| `MNN_OPENCL` | 启用 OpenCL 后端 | OFF |
| `MNN_VULKAN` | 启用 Vulkan 后端 | OFF |
| `MNN_METAL` | 启用 Metal 后端 (Apple) | OFF |
| `MNN_CUDA` | 启用 CUDA 后端 | OFF |
| `MNN_ARM82` | 启用 ARMv8.2 FP16 计算 | ON |
| `MNN_BUILD_TRAIN` | 构建训练模块 | OFF |
| `MNN_BUILD_CONVERTER` | 构建模型转换工具 | OFF |
| `MNN_BUILD_LLM` | 构建 LLM 引擎 | OFF |
| `MNN_LOW_MEMORY` | 低内存优化 | OFF |
| `MNN_BUILD_TEST` | 构建测试 | OFF |
| `MNN_BUILD_OPENCV` | 构建 OpenCV API | OFF |
| `MNN_JNI` | 构建 JNI 绑定 | OFF |

### 运行测试

```bash
cd build
./run_test.out           # 运行所有单元测试
./run_test.out op 0 0 4  # 运行算子测试 (CPU, 单线程, 4线程)
./run_test.out op 3 1 4  # 运行 OpenCL 测试
```

## 开发约定

### 代码风格
- C++ 标准: C++11 (部分模块支持 C++17)
- 使用 `MNN_PUBLIC` 宏导出公共 API
- 命名空间: `MNN`
- 类名使用大驼峰命名 (如 `Interpreter`, `Tensor`)
- 函数名使用小驼峰命名 (如 `createSession`, `runSession`)
- 常量使用全大写下划线 (如 `MNN_VERSION_MAJOR`)

### 错误处理
- 使用 `ErrorCode` 枚举返回错误码
- 调试模式下使用 `MNN_ASSERT` 宏
- 日志输出使用 `MNN_PRINT` 和 `MNN_ERROR` 宏

### 内存管理
- 使用 `Interpreter::destroy()` 销毁解释器实例
- 使用 `releaseSession()` 释放会话
- 调用 `releaseModel()` 可在创建会话后释放模型缓冲区

### 后端架构
每个硬件后端继承自 `Backend` 基类，实现:
- 内存分配 (`onAcquireBuffer`)
- 执行创建 (`onCreate`)
- 执行调度 (`onExecute`)

### 添加新算子
1. 在 `schema/current/MNN.fbs` 定义算子参数
2. 在 `source/geometry/` 实现几何计算
3. 在 `source/shape/` 实现形状推理
4. 在对应后端目录实现执行逻辑

## 常用 API

### 基础推理流程

```cpp
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

// 1. 创建解释器
auto net = MNN::Interpreter::createFromFile("model.mnn");

// 2. 配置调度
MNN::ScheduleConfig config;
config.type = MNN_FORWARD_CPU;
config.numThread = 4;

// 3. 创建会话
auto session = net->createSession(config);

// 4. 获取输入输出
auto input = net->getSessionInput(session, nullptr);
auto output = net->getSessionOutput(session, nullptr);

// 5. 调整输入尺寸 (如需要)
net->resizeTensor(input, {1, 3, 224, 224});
net->resizeSession(session);

// 6. 填充数据并执行
// ... 填充 input 数据 ...
net->runSession(session);

// 7. 读取输出
// ... 读取 output 数据 ...

// 8. 清理
net->releaseSession(session);
MNN::Interpreter::destroy(net);
```

### 使用 Express 模块

```cpp
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/MathOp.hpp>

using namespace MNN::Express;

// 创建变量
auto x = _Input({1, 3, 224, 224}, NCHW, halide_type_of<float>());

// 构建计算图
auto y = _Conv(0.0f, 0.0f, x, 3, 32, {3, 3});  // 卷积
y = _MaxPool(y, {2, 2}, {2, 2});               // 池化
y = _Relu(y);                                  // 激活

// 执行计算
auto output = y->execute();
```

## 测试说明

项目使用自定义测试框架 (`test/MNNTestSuite.h`)。

```bash
# 完整测试流程 (Linux)
./test.sh linux

# 本地测试
./test.sh local

# Android 测试
./test.sh android
```

测试类型:
- 单元测试 (`test/`)
- 模型测试 (`tools/script/modelTest.py`)
- 转换测试 (`tools/script/convertOnnxTest.py` 等)
- Python 绑定测试 (`pymnn/test/`)

## 相关文档

- 官方文档: https://mnn-docs.readthedocs.io/en/latest/
- GitHub: https://github.com/alibaba/MNN
- 本地文档构建: `cd docs && make html`

## 分支信息

- 当前分支: `feature/profile`
- 默认分支: `master` (或 `main`)
- Git Remote: https://github.com/huluhuluu/MNN_INFER.git
