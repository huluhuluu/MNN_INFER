//
//  AttentionTest.cpp
//  MNNTests
//
//  Created by MNN on 2024/07/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "core/OpCommonUtils.hpp"
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include <stdlib.h>
#include <vector>
#include <MNN/AutoTime.hpp>

using namespace MNN::Express;

int NumHead   = 16;
int KvNumHead = 2;
int HeadDim   = 128;
const float diff_threshold = 0.001;
const float diff_percent_threshold = 0.1;
const int pastLength = 101;
#define GENERATE_TOKENS 128
struct KVMeta {
    enum {
        NoChange,
        PendingWrite,
        PendingRead
    } file_operation;
    size_t block = 4096;
    size_t previous = 0;
    size_t remove = 0;
    int* reserve = nullptr;
    int n_reserve = 0;
    size_t add = 0;
    std::string file_name = "";
    int file_flag = NoChange;
    int seqlen_in_disk = 0;
    int layer_index = 0;
    int layer_nums = 0;
    std::vector<int> reserveHost;
    void sync() {
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
};

static KVMeta gMeta;
static std::shared_ptr<Module> _makeAttentionModule(int attentionMode = 8) {
    auto Q = _Input();
    auto K = _Input();
    auto V = _Input();
    auto mask = _Input();
    std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
    attention->type = MNN::OpType_Attention;
    attention->main.type = MNN::OpParameter_AttentionParam;
    attention->main.value = new MNN::AttentionParamT;
    attention->main.AsAttentionParam()->kv_cache = true;
    auto o = Variable::create(Expr::create(attention.get(), {Q, K, V, mask}));
    auto buffer = Variable::save({o});
    MNN::ScheduleConfig config;
    auto status = MNNTestSuite::get()->pStaus;
    config.type = (MNNForwardType)status.forwardType;
    MNN::BackendConfig bnConfig;
    bnConfig.memory = (MNN::BackendConfig::MemoryMode)status.memory;
    bnConfig.precision = (MNN::BackendConfig::PrecisionMode)status.precision;
    bnConfig.power = (MNN::BackendConfig::PowerMode)status.power;
    config.backendConfig = &bnConfig;
    config.numThread = 1;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, &gMeta);
    rtmgr->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
    std::shared_ptr<Module> m(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
    return m;
}

struct KVCache {
    VARP pastK;
    VARP pastV;
    VARP pastMask;
    int current = 0;
    KVCache() {
        pastK = _Input({1, KvNumHead, 1, pastLength, HeadDim}, NCHW);
        pastV = _Input({1, KvNumHead, 1, pastLength, HeadDim}, NCHW);
        pastMask = _Input({pastLength}, NCHW);
        ::memset(pastK->writeMap<float>(), 0, pastK->getInfo()->size * sizeof(float));
        ::memset(pastV->writeMap<float>(), 0, pastK->getInfo()->size * sizeof(float));
        for (int v=0; v<pastLength; ++v) {
            pastMask->writeMap<float>()[v] = std::numeric_limits<float>::lowest();
        }
    }
};

static VARP _computeAttentionExpr(VARP Q, VARP K, VARP V, VARP mask, KVCache cache) {
    auto qinfo = Q->getInfo();
    auto kinfo = K->getInfo();
    auto vinfo = V->getInfo();
    auto seqLength = qinfo->dim[1];
    auto numHead = qinfo->dim[2];
    auto headDim = qinfo->dim[3];
    auto kvNumHead = kinfo->dim[2];
    auto batch = qinfo->dim[0];
    auto group = numHead / kvNumHead;
    if (mask->getInfo()->type.code == halide_type_int) {
        mask = (_Scalar<float>(1.0) - _Cast<float>(mask)) * _Scalar<float>(std::numeric_limits<float>::lowest());
    }

    Q = _Reshape(Q, {batch, seqLength, kvNumHead,group, headDim});
    Q = _Transpose(Q, {0, 2, 3, 1, 4});
    K = _Reshape(K, {batch, seqLength, kvNumHead, 1, headDim});
    K = _Transpose(K, {0, 2, 3, 1, 4});

    auto scale = 1.0f / sqrtf(headDim);
    K = K * _Scalar<float>(scale);
    K.fix(VARP::CONSTANT);
    auto QK = _MatMul(Q, K, false, true); // [batch, kvNumHead, group , seq_len, seq_len]
    QK = QK + mask;
    auto QKPast = _MatMul(Q, cache.pastK, false, true);
    QKPast = QKPast + cache.pastMask;
    QK = _Concat({QKPast, QK}, -1);
    QK = _Softmax(QK, -1);
    V = _Reshape(V, {batch, seqLength, kvNumHead, 1, headDim});
    V = _Transpose(V, {0, 2, 3, 1, 4});
    V.fix(VARP::CONSTANT);
    auto totalV = _Concat({cache.pastV, V}, 3);
    auto QKV = _MatMul(QK, totalV, false, false);
    auto info = QKV->getInfo();
    auto O = _Transpose(QKV, {0, 3, 1, 2, 4});
    O = _Reshape(O, {batch, seqLength, -1});
    O.fix(VARP::CONSTANT);
    // Update KVCache
    for (int y=0; y<kvNumHead; ++y) {
        ::memcpy(cache.pastK->writeMap<float>() + y * pastLength * headDim + cache.current * headDim, K->readMap<float>() + y * seqLength * headDim, seqLength * headDim * sizeof(float));
        ::memcpy(cache.pastV->writeMap<float>() + y * pastLength * headDim + cache.current * headDim, V->readMap<float>() + y * seqLength * headDim, seqLength * headDim * sizeof(float));
    }
    for (int i=0; i<seqLength; ++i) {
        cache.pastMask->writeMap<float>()[i+cache.current] = 0.0f;
    }
    cache.current += seqLength;
    return O;
}

static std::vector< std::vector< std::vector<float> > > generateRandTensor(int C, int H, int W, int precision) {
    std::vector< std::vector< std::vector<float> > > a;
    a.resize(C);
    for (int i = 0; i < C; i++) {
        a[i].resize(H);
        for (int j = 0; j < H; j++) {
            a[i][j].resize(W);
            for (int k = 0; k < W; k++) {
                if (precision == 2) {
                    a[i][j][k] = ((i + j + k) % 10) * 0.002;
                } else {
                    a[i][j][k] = ((i + j + k) % 10) * 0.16 - 5.6;
                }
            }
        }
    }
    return a;
}

VARP vector_to_var(std::vector< std::vector< std::vector<float> > > & a) {
    int C = a.size();
    int H = a[0].size();
    int W = a[0][0].size();
    VARP var = _Input({1, C, H, W}, NCHW, halide_type_of<float>());
    float * ptr = var->writeMap<float>();
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < H; j++) {
            for (int k = 0; k < W; k++) {
                ptr[i * H * W + j * W + k] = a[i][j][k];
            }
        }
    }
    var->unMap();
    return var;
}

VARP vector_to_var(std::vector< std::vector<int> > & a) {
    int H = a.size();
    int W = a[0].size();
    VARP var = _Input({1, 1, H, W}, NCHW, halide_type_of<int>());
    int * ptr = var->writeMap<int>();
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            ptr[i * W + j] = a[i][j];
        }
    }
    var->unMap();
    return var;
}

static std::vector< std::vector< std::vector<float> > >
computeAttention (
    std::vector< std::vector< std::vector<float> > > & query,
    std::vector< std::vector< std::vector<float> > > & key,
    std::vector< std::vector< std::vector<float> > > & value,
    std::vector< std::vector<int> > & mask,
    int seq_len, int kv_seq_len )
{
    int group_size = NumHead / KvNumHead;
    std::vector< std::vector< std::vector<float> > > output(seq_len);
    for (int i = 0; i < seq_len; i++) {
        output[i].resize(NumHead);
        for (int j = 0; j < NumHead; j++) {
            output[i][j].resize(HeadDim);
        }
    }
    for (int h = 0; h < NumHead; h++) {
        int kv_h = h / group_size;
        /*---- Q * K ----*/
        std::vector< std::vector<float> > qk(seq_len, std::vector<float>(kv_seq_len, 0.0f));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < kv_seq_len; j++) {
                qk[i][j] = 0.0f;
                for (int k = 0; k < HeadDim; k++) {
                    qk[i][j] += query[i][h][k] * key[j][kv_h][k];
                }
            }
        }
        /*---- Mask QK ----*/
        if(mask.size() > 0) {
            float scale = 1.0 / sqrt(HeadDim);
            if (mask[0].size() == seq_len) {
                auto diff = kv_seq_len - seq_len;
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < seq_len; j++) {
                        qk[i][j+diff] = qk[i][j+diff] * scale + (1.f - mask[i][j]) * std::numeric_limits<float>::lowest();
                    }
                }
            } else {
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < kv_seq_len; j++) {
                        qk[i][j] = qk[i][j] * scale + (1.f - mask[i][j]) * std::numeric_limits<float>::lowest();
                    }
                }
            }
        } else {
            float scale = 1.0 / sqrt(HeadDim);
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < kv_seq_len; j++) {
                    qk[i][j] *= scale;
                }
            }
        }
        /*---- Softmax QK ----*/
        for (int i = 0; i < seq_len; i++) {
            float maxValue = qk[i][0];
            for (int j = 1; j < kv_seq_len; j++) {
                maxValue = ALIMAX(maxValue, qk[i][j]);
            }
            for (int j = 0; j < kv_seq_len; j++) {
                qk[i][j] -= maxValue;
            }
            float sum = 0.0f;
            for (int j = 0; j < kv_seq_len; j++) {
                sum += exp(qk[i][j]);
            }
            for (int j = 0; j < kv_seq_len; j++) {
                qk[i][j] = exp(qk[i][j]) / sum;
            }
        }
        /*---- QK * V ----*/
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < HeadDim; j++) {
                output[i][h][j] = 0.0f;
                for (int k = 0; k < kv_seq_len; k++) {
                    output[i][h][j] += qk[i][k] * value[k][kv_h][j];
                }
            }
        }
    }
    return output;
}

class NaiveAttention {
    private:
        std::vector< std::vector< std::vector<float> > >  mPastKey, mPastValue;
        int mPastLen;
    public:
        NaiveAttention() : mPastLen(0) {}
        ~NaiveAttention() = default;
        std::vector< std::vector< std::vector<float> > > onExecute (
            std::vector< std::vector< std::vector<float> > > & query,
            std::vector< std::vector< std::vector<float> > > & key,
            std::vector< std::vector< std::vector<float> > > & value,
            std::vector< std::vector<int> > & mask,
            int seq_len )
        {
            for (int i = 0; i < seq_len; i++) {
                mPastKey.push_back(key[i]);
                mPastValue.push_back(value[i]);
            }
            mPastLen += seq_len;
            return computeAttention(query, mPastKey, mPastValue, mask, seq_len, mPastLen);
        }
};

class AttentionTest : public MNNTestCase {
protected:
    std::vector< std::vector< std::vector<float> > > query;
    std::vector< std::vector< std::vector<float> > > key;
    std::vector< std::vector< std::vector<float> > > value;
    std::vector< std::vector<int> > mask;
    std::vector< std::vector< std::vector<float> > > expected_result;
    VARP Query, Key, Value, Mask, Output;
    VARP Query1, Key1, Value1, Mask1;
public:
    AttentionTest() = default;
    virtual ~AttentionTest() = default;
    void generateInput(int seq_len, int precision, bool genDecodeInput = false) {
        query = generateRandTensor(seq_len, NumHead, HeadDim, precision);
        key   = generateRandTensor(seq_len, KvNumHead, HeadDim, precision);
        value = generateRandTensor(seq_len, KvNumHead, HeadDim, precision);
        Query = vector_to_var(query);
        Key   = vector_to_var(key);
        Value = vector_to_var(value);
        if (genDecodeInput) {
            auto vecquery = generateRandTensor(1, NumHead, HeadDim, precision);
            auto veckey   = generateRandTensor(1, KvNumHead, HeadDim, precision);
            auto vecvalue = generateRandTensor(1, KvNumHead, HeadDim, precision);
            Query1 = vector_to_var(vecquery);
            Key1   = vector_to_var(veckey);
            Value1 = vector_to_var(vecvalue);
        }
    }
    void generateChunkMask(int seq_len, int kv_seq_len, int chunk_size, bool genDecodeInput = false) {
        // 防止除以0
        if (chunk_size <= 0) chunk_size = 1;

        mask.resize(seq_len);

        // 计算历史长度 (Gap)，用于处理 KV 长度大于 Seq 长度的情况 (Right Alignment)
        // j < gap 的部分通常被视为 History，默认可见
        int gap = kv_seq_len - seq_len;

        for (int i = 0; i < seq_len; i++) {
            mask[i].resize(kv_seq_len);

            // --- 核心逻辑对应 ---
            // MNN Expr: auto N = _Divide(i, rankVar) * rankVar + rankVar;
            // i 是当前行 (Query)，计算当前块的右边界 (不包含)
            // 比如 rank=2, i=0, block_end_rel=2; i=2, block_end_rel=4
            int block_end_rel = (i / chunk_size) * chunk_size + chunk_size;

            for (int j = 0; j < kv_seq_len; j++) {
                // 将 j 转换为相对于当前 seq_len 的坐标
                int j_rel = j - gap;

                if (j_rel < 0) {
                    // 情况 1: j 在 Gap 区域 (历史 KV Cache)
                    // 通常历史数据对当前所有 Token 都是可见的
                    mask[i][j] = 1;
                } else {
                    // 情况 2: j 在当前处理的序列范围内
                    // 对应 MNN Expr: _Less(j, N)
                    if (j_rel < block_end_rel) {
                        mask[i][j] = 1;
                    } else {
                        mask[i][j] = 0;
                    }
                }
            }
        }

        // 转为 VARP 并处理成 -inf / 0.0 格式
        Mask = vector_to_var(mask);
        Mask = (_Scalar<float>(1.0) - _Cast<float>(Mask)) * _Scalar<float>(std::numeric_limits<float>::lowest());

        // Decode Input 部分通常保持全 1 (即看清所有历史)，或者根据需求修改
        if (genDecodeInput) {
            std::vector<std::vector<int>> vecmask;
            vecmask.resize(1);
            vecmask[0].resize(gMeta.previous + 1);
            for (int i = 0; i < gMeta.previous + 1; ++i) {
                vecmask[0][i] = 1;
            }
            Mask1 = vector_to_var(vecmask);
            Mask1 = (_Scalar<float>(1.0) - _Cast<float>(Mask1)) * _Scalar<float>(std::numeric_limits<float>::lowest());
        }
    }

    void generateMask(int seq_len, int kv_seq_len, bool genDecodeInput = false) {
        mask.resize(seq_len);
        for (int i = 0; i < seq_len; i++) {
            mask[i].resize(kv_seq_len);
            for (int j = 0; j < kv_seq_len; j++) {
                if (j - i <= kv_seq_len - seq_len) {
                    mask[i][j] = 1;
                } else {
                    mask[i][j] = 0;
                }
            }
        }
        Mask = _Input({}, NCHW, halide_type_of<float>());
        Mask1 = _Input({}, NCHW, halide_type_of<float>());
        Mask->writeMap<float>()[0] = 0.0f;
        Mask1->writeMap<float>()[0] = 0.0f;
    }

    bool compareResult(int seq_len) {
        const float * resultPtr = Output->readMap<float>();
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < NumHead; j++) {
                for (int k = 0; k < HeadDim; k++) {
                    float diff = fabs(resultPtr[i * NumHead * HeadDim + j * HeadDim + k] - expected_result[i][j][k]);
                    float diff_percent = fabs(diff / expected_result[i][j][k]);
                    if (diff > diff_threshold && diff_percent > diff_percent_threshold) {
                        printf("Result Mismatch: expected %lf but got %lf in CPU Attention Test\n", expected_result[i][j][k], resultPtr[i * NumHead * HeadDim + j * HeadDim + k]);
                        printf("Error Position: Output[%d][%d][%d]\n", i, j, k);
                        return false;
                    }
                }
            }
        }
        Output->unMap();
        return true;
    }

    virtual bool run(int precision) {
        srand(2024);
        // unit test 1
        {
            std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
            std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
            attention->type = MNN::OpType_Attention;
            attention->main.type = MNN::OpParameter_AttentionParam;
            attention->main.value = new MNN::AttentionParamT;
            attention->main.AsAttentionParam()->kv_cache = true;
            int seq_len = 10;
            generateInput(seq_len, precision);
            generateMask(seq_len, seq_len);
            expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
            auto attn = _makeAttentionModule();
            gMeta.add = seq_len;
            Output = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            KVCache kvCache;
            bool pass = compareResult(seq_len);
            if (!pass) {
                printf("Error: LowerTriangular Attention with kv_cache unit test failed!\n");
                return false;
            }

            /* generate mask expr */
            /* generate mask expr */
            auto MaskExpr = vector_to_var(mask);
            MaskExpr = (_Scalar<float>(1.0) - _Cast<float>(MaskExpr)) * _Scalar<float>(std::numeric_limits<float>::lowest());
            Output = _computeAttentionExpr(Query, Key, Value, MaskExpr, kvCache);
            pass = compareResult(seq_len);
            if (!pass) {
                FUNC_PRINT(1);
                return false;
            }
            // naiveAttention with history is error, use expr to test
            Output = _computeAttentionExpr(Query, Key, Value, MaskExpr, kvCache);
            gMeta.add = seq_len;
            auto output2 = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            auto diff = _ReduceMax(output2 - Output)->readMap<float>()[0];
            if (diff >= 0.01f) {                 FUNC_PRINT_ALL(diff, f);
                return false;
            }
        }
        // test2
        {
            std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
            std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
            attention->type = MNN::OpType_Attention;
            attention->main.type = MNN::OpParameter_AttentionParam;
            attention->main.value = new MNN::AttentionParamT;
            attention->main.AsAttentionParam()->kv_cache = true;
            int seq_len = 10;
            generateInput(seq_len, precision);
            generateChunkMask(seq_len, seq_len, 2);
            expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
            auto attn = _makeAttentionModule();
            gMeta.previous = 0;
            gMeta.add = seq_len;
            Output = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            KVCache kvCache;
            bool pass = compareResult(seq_len);
            if (!pass) {
                printf("Error: Not LowerTriangular Attention with kv_cache unit test failed!\n");
                return false;
            }
            Output = _computeAttentionExpr(Query, Key, Value, Mask, kvCache);
            pass = compareResult(seq_len);
            if (!pass) {
                FUNC_PRINT(1);
                return false;
            }
            // naiveAttention with history is error, use expr to test
            Output = _computeAttentionExpr(Query, Key, Value, Mask, kvCache);
            gMeta.add = seq_len;
            auto output2 = attn->onForward({Query, Key, Value, Mask})[0];
            gMeta.sync();
            auto diff = _ReduceMax(output2 - Output)->readMap<float>()[0];
            if (diff >= 0.01f) {
                FUNC_PRINT_ALL(diff, f);
                return false;
            }
        }
        // unit test 3
        {
            auto rtInfo = ExecutorScope::Current()->getRuntime().first;
            bool cpuInfer = true;
            for(auto &rt : rtInfo) {
                if(rt.first != MNN_FORWARD_CPU) {
                    cpuInfer = false;
                    break;
                }
            }
            if(cpuInfer) {
                // TODO: CPU support kv_cache == false
                return true;
            }
            std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
            std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
            attention->type = MNN::OpType_Attention;
            attention->main.type = MNN::OpParameter_AttentionParam;
            attention->main.value = new MNN::AttentionParamT;
            attention->main.AsAttentionParam()->kv_cache = false;
            int seq_len = 128;
            generateInput(seq_len, precision);
            mask.clear();
            expected_result = naiveAttention->onExecute(query, key, value, mask, seq_len);
            Output = Variable::create(Expr::create(attention.get(), {Query, Key, Value}));
            bool pass = compareResult(seq_len);
            if (!pass) {
                printf("Error: Attention without kv_cacheunit test failed!\n");
                return false;
            }
        }
        return true;
    }
};

class SpeedAttentionTest : public AttentionTest {
    protected:
        std::vector< std::vector< std::vector<float> > > query;
        std::vector< std::vector< std::vector<float> > > key;
        std::vector< std::vector< std::vector<float> > > value;
        std::vector< std::vector<int> > mask;
        std::vector< std::vector< std::vector<float> > > expected_result;

public:
SpeedAttentionTest() = default;
    virtual ~SpeedAttentionTest() = default;

    virtual bool run(int precision) {
        std::vector<int> seqs = {4096};
        std::shared_ptr<NaiveAttention> naiveAttention(new NaiveAttention);
        std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
        attention->type = MNN::OpType_Attention;
        attention->main.type = MNN::OpParameter_AttentionParam;
        attention->main.value = new MNN::AttentionParamT;
        attention->main.AsAttentionParam()->kv_cache = true;
        /* 3 attention module */
        std::vector<int> quantQKV = {8, 9, 10};
        std::vector<std::string> testNames = {"float qkv", "quant qk", "quant qkv"};
        for (int n = 0; n < seqs.size(); ++n) {
            int seq_len = seqs[n];
            MNN_PRINT(">>> seq_len=%d, decode_len=%d\n", seq_len, GENERATE_TOKENS);
            generateInput(seqs[n], precision, true);
            generateMask(seqs[n], seq_len, true);
            for (int m = 0; m < testNames.size(); ++m) {
                gMeta.previous = 0;
                gMeta.add = seq_len;
                auto _module = _makeAttentionModule(quantQKV[m]);
                MNN::Timer t1;
                for (int x = 0; x < 5; ++x) {
                    Output = _module->onForward({Query, Key, Value, Mask})[0];
                }
                auto time = (float)t1.durationInUs() / 1000.0f / 5.f;
                MNN_PRINT("%s: prefill cost = %.2f\n", testNames[m].c_str(), time);
                gMeta.sync();
                MNN::Timer t2;
                for (int x = 0; x < GENERATE_TOKENS; ++x) {
                    gMeta.add = 1;
                    auto output2 = _module->onForward({Query1, Key1, Value1, Mask1})[0];
                    gMeta.sync();
                }
                time = (float)t2.durationInUs() / 1000.0f;
                MNN_PRINT("%s: decode cost = %f\n", testNames[m].c_str(), time);
            }
        }
        return true;
    }
};


// ---------------------------------------------------------------------------
// FlashAttention correctness test (OpenCL only).
//
// Runs the same Attention op three ways on identical inputs and compares:
//   A) default prefillResize / longPrefillResize path
//   B) MNN_OPENCL_FLASH_ATTENTION=1 fused path
//   C) a double-precision host reference over head 0
// The inputs are pre-rounded to fp16 so the host reference sees exactly the
// values the GPU sees; the only remaining differences are accumulation order
// and accumulator precision. Pass criterion is "FlashAttention is at least as
// close to the double reference as the default path is", which is the property
// that actually matters and does not depend on the default path's own error.
// ---------------------------------------------------------------------------
class FlashAttentionTest : public MNNTestCase {
public:
    virtual ~FlashAttentionTest() = default;

    // round-to-nearest-even into IEEE fp16 precision, kept in a float
    static float quantHalf(float f) {
        uint32_t x;
        ::memcpy(&x, &f, 4);
        const uint32_t sign = x & 0x80000000u;
        int exp = (int)((x >> 23) & 0xffu) - 127;
        uint32_t man = x & 0x7fffffu;
        if (exp < -14) {
            float z = 0.0f;
            return sign ? -z : z;
        }
        uint32_t keep = man >> 13;
        const uint32_t rem = man & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (keep & 1u))) {
            keep++;
        }
        if (keep == 0x400u) { keep = 0; exp++; }
        const uint32_t y = sign | ((uint32_t)(exp + 127) << 23) | (keep << 13);
        float r;
        ::memcpy(&r, &y, 4);
        return r;
    }

    static VARP makeQKV(int seqLen, int heads, int headDim, int seed, std::vector<float>& host) {
        VARP v = _Input({1, seqLen, heads, headDim}, NCHW, halide_type_of<float>());
        auto ptr = v->writeMap<float>();
        const int total = seqLen * heads * headDim;
        host.resize(total);
        unsigned int st = (unsigned int)(seed * 2654435761u + 12345u);
        for (int i = 0; i < total; ++i) {
            st = st * 1103515245u + 12345u;
            float val = (float)((st >> 16) & 0x7fffu) / 32767.0f * 0.5f - 0.25f;
            val = quantHalf(val);
            ptr[i] = val;
            host[i] = val;
        }
        v->unMap();
        return v;
    }
    static VARP makeCausalMask(int seqLen) {
        VARP m = _Input({1, 1, seqLen, seqLen}, NCHW, halide_type_of<float>());
        auto ptr = m->writeMap<float>();
        for (int i = 0; i < seqLen; ++i) {
            for (int j = 0; j < seqLen; ++j) {
                ptr[i * seqLen + j] = (j > i) ? std::numeric_limits<float>::lowest() : 0.0f;
            }
        }
        m->unMap();
        return m;
    }

    // The shared _makeAttentionModule uses numThread = 1, but the OpenCL Attention creator
    // is only registered for BUFFER while OpenCLBackend defaults to IMAGE on Adreno, so the
    // op would silently land on the CPU backup backend. Llm sets numThread |= 64
    // (MNN_GPU_MEMORY_BUFFER) for exactly this reason.
    static std::shared_ptr<Module> makeModuleBuffer() {
        auto Q = _Input();
        auto K = _Input();
        auto V = _Input();
        auto mask = _Input();
        std::shared_ptr<MNN::OpT> attention(new MNN::OpT);
        attention->type = MNN::OpType_Attention;
        attention->main.type = MNN::OpParameter_AttentionParam;
        attention->main.value = new MNN::AttentionParamT;
        attention->main.AsAttentionParam()->kv_cache = true;
        auto o = Variable::create(Expr::create(attention.get(), {Q, K, V, mask}));
        auto buffer = Variable::save({o});
        MNN::ScheduleConfig config;
        auto status = MNNTestSuite::get()->pStaus;
        config.type = (MNNForwardType)status.forwardType;
        MNN::BackendConfig bnConfig;
        bnConfig.memory = (MNN::BackendConfig::MemoryMode)status.memory;
        bnConfig.precision = (MNN::BackendConfig::PrecisionMode)status.precision;
        bnConfig.power = (MNN::BackendConfig::PowerMode)status.power;
        config.backendConfig = &bnConfig;
        config.numThread = 1;
        if (config.type == MNN_FORWARD_OPENCL) {
            config.numThread |= 64;
        }
        std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(config));
        rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, &gMeta);
        rtmgr->setHint(MNN::Interpreter::ATTENTION_OPTION, 8);
        std::shared_ptr<Module> m(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
        return m;
    }

    static void setFlash(int on) {
#if defined(_WIN32)
        _putenv_s("MNN_OPENCL_FLASH_ATTENTION", on ? "1" : "0");
#else
        setenv("MNN_OPENCL_FLASH_ATTENTION", on ? "1" : "0", 1);
#endif
    }

    // double-precision attention for one query head, causal, over the whole KV history
    static void goldHead(const std::vector<float>& q, int qLen, int numHead, int headDim,
                         const std::vector<float>& kAll, const std::vector<float>& vAll,
                         int kvLen, int kvNumHead, int pastLen, int head,
                         std::vector<double>& out) {
        const int group = numHead / kvNumHead;
        const int kvh = head / group;
        const double scale = 1.0 / sqrt((double)headDim);
        out.assign((size_t)qLen * headDim, 0.0);
        std::vector<double> logits(kvLen);
        for (int i = 0; i < qLen; ++i) {
            const float* qp = q.data() + ((size_t)i * numHead + head) * headDim;
            const int last = pastLen + i;
            double mx = -1e300;
            for (int j = 0; j <= last && j < kvLen; ++j) {
                const float* kp = kAll.data() + ((size_t)j * kvNumHead + kvh) * headDim;
                double acc = 0.0;
                for (int d = 0; d < headDim; ++d) {
                    acc += (double)qp[d] * (double)kp[d];
                }
                logits[j] = acc * scale;
                if (logits[j] > mx) { mx = logits[j]; }
            }
            double sum = 0.0;
            for (int j = 0; j <= last && j < kvLen; ++j) {
                logits[j] = exp(logits[j] - mx);
                sum += logits[j];
            }
            double* op = out.data() + (size_t)i * headDim;
            for (int j = 0; j <= last && j < kvLen; ++j) {
                const double p = logits[j] / sum;
                const float* vp = vAll.data() + ((size_t)j * kvNumHead + kvh) * headDim;
                for (int d = 0; d < headDim; ++d) {
                    op[d] += p * (double)vp[d];
                }
            }
        }
    }

    struct Err { double maxAbs = 0.0; double relL2 = 0.0; };
    static Err compareHead(const std::vector<float>& got, const std::vector<double>& gold,
                           int qLen, int numHead, int headDim, int head) {
        Err e;
        double sumSq = 0.0, sumRef = 0.0;
        for (int i = 0; i < qLen; ++i) {
            for (int d = 0; d < headDim; ++d) {
                const double g = gold[(size_t)i * headDim + d];
                const double v = (double)got[((size_t)i * numHead + head) * headDim + d];
                const double diff = v - g;
                e.maxAbs = fmax(e.maxAbs, fabs(diff));
                sumSq += diff * diff;
                sumRef += g * g;
            }
        }
        e.relL2 = (sumRef > 0.0) ? sqrt(sumSq / sumRef) : sqrt(sumSq);
        return e;
    }

    // chunks: sequence of prefill lengths run back to back against one KV cache.
    // The output of the LAST chunk is what gets compared.
    bool runCase(const char* name, std::vector<int> chunks, int numHead, int kvNumHead, int headDim, bool doGold) {
        NumHead = numHead; KvNumHead = kvNumHead; HeadDim = headDim;
        const int nChunk = (int)chunks.size();
        std::vector<VARP> Qs(nChunk), Ks(nChunk), Vs(nChunk), Ms(nChunk);
        std::vector<std::vector<float>> qh(nChunk), kh(nChunk), vh(nChunk);
        for (int c = 0; c < nChunk; ++c) {
            Qs[c] = makeQKV(chunks[c], numHead, headDim, 11 + c * 7, qh[c]);
            Ks[c] = makeQKV(chunks[c], kvNumHead, headDim, 12 + c * 7, kh[c]);
            Vs[c] = makeQKV(chunks[c], kvNumHead, headDim, 13 + c * 7, vh[c]);
            Ms[c] = makeCausalMask(chunks[c]);
        }
        std::vector<float> res[2];
        for (int pass = 0; pass < 2; ++pass) {
            setFlash(pass);
            gMeta.previous = 0; gMeta.remove = 0; gMeta.n_reserve = 0; gMeta.reserve = nullptr;
            auto m = makeModuleBuffer();
            if (nullptr == m) {
                MNN_ERROR("FlashAttentionTest[%s]: module load failed\n", name);
                return false;
            }
            VARP out;
            for (int c = 0; c < nChunk; ++c) {
                gMeta.add = chunks[c];
                out = m->onForward({Qs[c], Ks[c], Vs[c], Ms[c]})[0];
                if (c + 1 < nChunk) {
                    out->readMap<float>();
                    gMeta.sync();
                }
            }
            auto info = out->getInfo();
            auto ptr = out->readMap<float>();
            if (nullptr == info || nullptr == ptr) {
                MNN_ERROR("FlashAttentionTest[%s]: null output\n", name);
                return false;
            }
            res[pass].assign(ptr, ptr + info->size);
            gMeta.sync();
        }
        setFlash(0);
        if (res[0].size() != res[1].size() || res[0].empty()) {
            MNN_ERROR("FlashAttentionTest[%s]: output size mismatch\n", name);
            return false;
        }
        // direct A/B diff over the whole tensor
        double abMax = 0.0, abSq = 0.0, refSq = 0.0;
        for (size_t i = 0; i < res[0].size(); ++i) {
            const double d = (double)res[1][i] - (double)res[0][i];
            abMax = fmax(abMax, fabs(d));
            abSq += d * d;
            refSq += (double)res[0][i] * (double)res[0][i];
        }
        const double abRel = (refSq > 0.0) ? sqrt(abSq / refSq) : sqrt(abSq);
        MNN_PRINT("[%s] default-vs-flash: maxAbs=%.3e relL2=%.3e\n", name, abMax, abRel);

        bool ok = true;
        if (doGold) {
            const int qLen = chunks[nChunk - 1];
            int pastLen = 0;
            for (int c = 0; c + 1 < nChunk; ++c) { pastLen += chunks[c]; }
            std::vector<float> kAll, vAll;
            for (int c = 0; c < nChunk; ++c) {
                kAll.insert(kAll.end(), kh[c].begin(), kh[c].end());
                vAll.insert(vAll.end(), vh[c].begin(), vh[c].end());
            }
            const int kvLen = pastLen + qLen;
            const int heads[2] = {0, numHead - 1};
            for (int hi = 0; hi < 2; ++hi) {
                const int head = heads[hi];
                std::vector<double> gold;
                goldHead(qh[nChunk - 1], qLen, numHead, headDim, kAll, vAll, kvLen, kvNumHead, pastLen, head, gold);
                auto eRef = compareHead(res[0], gold, qLen, numHead, headDim, head);
                auto eFa  = compareHead(res[1], gold, qLen, numHead, headDim, head);
                MNN_PRINT("[%s] head %2d vs fp64: default maxAbs=%.3e relL2=%.3e | flash maxAbs=%.3e relL2=%.3e\n",
                          name, head, eRef.maxAbs, eRef.relL2, eFa.maxAbs, eFa.relL2);
                // FlashAttention must not be worse than the path it replaces (small slack
                // for the fp16 output rounding), and must be accurate in absolute terms.
                if (!(eFa.relL2 <= eRef.relL2 * 1.5 + 1e-4)) {
                    MNN_ERROR("[%s] head %d: flash relL2 %.3e worse than default %.3e\n", name, head, eFa.relL2, eRef.relL2);
                    ok = false;
                }
                if (!(eFa.relL2 < 1e-2)) {
                    MNN_ERROR("[%s] head %d: flash relL2 %.3e too large\n", name, head, eFa.relL2);
                    ok = false;
                }
            }
        } else {
            if (!(abRel < 1e-2)) {
                MNN_ERROR("[%s] default-vs-flash relL2 %.3e too large\n", name, abRel);
                ok = false;
            }
        }
        return ok;
    }

    virtual bool run(int precision) {
        auto type = MNNTestSuite::get()->pStaus.forwardType;
        if (type != MNN_FORWARD_OPENCL) {
            MNN_PRINT("FlashAttentionTest: OpenCL only, skip (forwardType=%d)\n", type);
            return true;
        }
        bool ok = true;
        ok = runCase("seq64",       {64},       16, 8, 128, true)  && ok;
        ok = runCase("seq128",      {128},      16, 8, 128, true)  && ok;
        ok = runCase("seq200_tail", {200},      16, 8, 128, true)  && ok;
        ok = runCase("seq640_long", {640},      16, 8, 128, true)  && ok;
        ok = runCase("seq1024",     {1024},     16, 8, 128, false) && ok;
        ok = runCase("chunk64+32",  {64, 32},   16, 8, 128, true)  && ok;
        ok = runCase("chunk512+128",{512, 128}, 16, 8, 128, true)  && ok;
        return ok;
    }
};

MNNTestSuiteRegister(FlashAttentionTest, "op/flashattention");


MNNTestSuiteRegister(AttentionTest, "op/attention");
MNNTestSuiteRegister(SpeedAttentionTest, "speed/attention");
#endif
