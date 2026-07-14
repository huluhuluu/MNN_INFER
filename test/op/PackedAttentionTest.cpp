//
//  PackedAttentionTest.cpp
//  MNNTests
//
//  Created by MNN on 2025/03/20.
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

using namespace MNN::Express;

// Test parameters (use different names to avoid conflict with AttentionTest.cpp)
static int gPackedNumHead   = 32;
static int gPackedKvNumHead = 8;
static int gPackedHeadDim   = 128;
static const float gPackedDiffThreshold = 0.01;
static const float gPackedDiffPercentThreshold = 0.1;

static void syncKVMeta(MNN::KVMeta& meta) {
    int revertNumber = 0;
    for (int i = 0; i < meta.n_reserve; ++i) {
        revertNumber += meta.reserve[2 * i + 1];
    }
    meta.previous = meta.previous - meta.remove + meta.add + revertNumber;
    meta.n_reserve = 0;
    meta.reserve = nullptr;
    meta.remove = 0;
    meta.add = 0;
}

static void syncBatchMeta(MNN::BatchKVMeta& batchMeta) {
    for (int id : batchMeta.calId) {
        auto iter = batchMeta.mMetas.find(id);
        if (iter != batchMeta.mMetas.end() && iter->second != nullptr) {
            syncKVMeta(*iter->second);
        }
    }
    batchMeta.calId.clear();
}

static void setKVCacheInfo(MNN::BatchKVMeta& batchMeta, int reqId, size_t add = 0, size_t remove = 0, int* reserve = nullptr, int nReserve = 0) {
    auto& meta = batchMeta.mMetas[reqId];
    if (meta == nullptr) {
        meta = new MNN::KVMeta();
        meta->previous = 0;
    }
    if (remove > meta->previous) {
        remove = meta->previous;
    }
    meta->remove = remove;
    meta->reserve = reserve;
    meta->n_reserve = nReserve;
    meta->add = add;
    if (add > 0) {
        batchMeta.calId.push_back(reqId);
    }
}

static void clearBatchMeta(MNN::BatchKVMeta& batchMeta) {
    for (auto& kv : batchMeta.mMetas) {
        delete kv.second;
    }
    batchMeta.mMetas.clear();
    batchMeta.calId.clear();
}

static MNN::BatchKVMeta gBatchMeta;
struct BatchMetaCleaner {
    ~BatchMetaCleaner() {
        clearBatchMeta(gBatchMeta);
    }
};
static BatchMetaCleaner gBatchMetaCleaner;

static std::shared_ptr<Module> _makePackedAttentionModule(int attentionMode = 8) {
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
    rtmgr->setHintPtr(MNN::Interpreter::KVCACHE_INFO, &gBatchMeta);
    rtmgr->setHint(MNN::Interpreter::ATTENTION_OPTION, attentionMode);
    rtmgr->setHint(MNN::Interpreter::PACKED_ATTENTION_MODE, 1); // Enable packed mode
    std::shared_ptr<Module> m(Module::load({}, {}, (uint8_t*)buffer.data(), buffer.size(), rtmgr));
    return m;
}

static std::vector<std::vector<std::vector<float>>> generateRandTensor(int seqLen, int numHead, int headDim, int precision) {
    std::vector<std::vector<std::vector<float>>> a;
    a.resize(seqLen);
    for (int i = 0; i < seqLen; i++) {
        a[i].resize(numHead);
        for (int j = 0; j < numHead; j++) {
            a[i][j].resize(headDim);
            for (int k = 0; k < headDim; k++) {
                if (precision == 2) {
                    a[i][j][k] = ((i + j + k) % 10) * 0.002;
                } else {
                    a[i][j][k] = ((i + j + k) % 10) * 0.16 - 0.8;
                }
            }
        }
    }
    return a;
}

static std::vector<std::vector<std::vector<float>>> generateSensitiveTensor(int seqLen, int numHead, int headDim, int seed) {
    std::vector<std::vector<std::vector<float>>> a(seqLen, std::vector<std::vector<float>>(numHead, std::vector<float>(headDim)));
    uint32_t state = static_cast<uint32_t>(seed);
    for (int i = 0; i < seqLen; i++) {
        for (int j = 0; j < numHead; j++) {
            for (int k = 0; k < headDim; k++) {
                state = state * 1664525u + 1013904223u;
                int v = static_cast<int>((state >> 8) & 0xFFFF);
                a[i][j][k] = (static_cast<float>(v) / 32768.0f - 1.0f) * 0.35f;
            }
        }
    }
    return a;
}

static std::vector<std::vector<std::vector<float>>> sliceTensor(const std::vector<std::vector<std::vector<float>>>& src, int start, int length) {
    std::vector<std::vector<std::vector<float>>> dst(length);
    for (int i = 0; i < length; ++i) {
        dst[i] = src[start + i];
    }
    return dst;
}

static std::vector<std::vector<std::vector<float>>> concatTensor(const std::vector<std::vector<std::vector<float>>>& first,
                                                                 const std::vector<std::vector<std::vector<float>>>& second) {
    std::vector<std::vector<std::vector<float>>> dst;
    dst.reserve(first.size() + second.size());
    dst.insert(dst.end(), first.begin(), first.end());
    dst.insert(dst.end(), second.begin(), second.end());
    return dst;
}

// Pack multiple requests into single tensor along seq dimension
VARP packRequests(const std::vector<std::vector<std::vector<std::vector<float>>>>& requests) {
    int totalSeqLen = 0;
    int numHead = requests[0][0].size();
    int headDim = requests[0][0][0].size();
    
    for (auto& req : requests) {
        totalSeqLen += req.size();
    }
    
    VARP var = _Input({1, totalSeqLen, numHead, headDim}, NCHW, halide_type_of<float>());
    float* ptr = var->writeMap<float>();
    
    int offset = 0;
    for (auto& req : requests) {
        int seqLen = req.size();
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < numHead; j++) {
                for (int k = 0; k < headDim; k++) {
                    ptr[(offset + i) * numHead * headDim + j * headDim + k] = req[i][j][k];
                }
            }
        }
        offset += seqLen;
    }
    var->unMap();
    return var;
}

// Compute attention for single request (reference implementation)
static std::vector<std::vector<std::vector<float>>> computeSingleAttention(
    std::vector<std::vector<std::vector<float>>>& query,
    std::vector<std::vector<std::vector<float>>>& key,
    std::vector<std::vector<std::vector<float>>>& value,
    const std::vector<std::vector<int>>& mask,
    int seqLen, int kvSeqLen)
{
    int groupSize = gPackedNumHead / gPackedKvNumHead;
    std::vector<std::vector<std::vector<float>>> output(seqLen);
    for (int i = 0; i < seqLen; i++) {
        output[i].resize(gPackedNumHead);
        for (int j = 0; j < gPackedNumHead; j++) {
            output[i][j].resize(gPackedHeadDim);
        }
    }
    
    for (int h = 0; h < gPackedNumHead; h++) {
        int kvH = h / groupSize;
        // Q * K
        std::vector<std::vector<float>> qk(seqLen, std::vector<float>(kvSeqLen, 0.0f));
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < kvSeqLen; j++) {
                for (int k = 0; k < gPackedHeadDim; k++) {
                    qk[i][j] += query[i][h][k] * key[j][kvH][k];
                }
            }
        }
        
        // Scale and mask
        float scale = 1.0 / sqrt(gPackedHeadDim);
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < kvSeqLen; j++) {
                qk[i][j] *= scale;
                if (mask.size() > 0 && mask[i].size() > j) {
                    if (mask[i][j] == 0) {
                        qk[i][j] = std::numeric_limits<float>::lowest();
                    }
                }
            }
        }
        
        // Softmax
        for (int i = 0; i < seqLen; i++) {
            float maxVal = qk[i][0];
            for (int j = 1; j < kvSeqLen; j++) {
                maxVal = std::max(maxVal, qk[i][j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < kvSeqLen; j++) {
                qk[i][j] = exp(qk[i][j] - maxVal);
                sum += qk[i][j];
            }
            for (int j = 0; j < kvSeqLen; j++) {
                qk[i][j] /= sum;
            }
        }
        
        // QK * V
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < gPackedHeadDim; j++) {
                output[i][h][j] = 0.0f;
                for (int k = 0; k < kvSeqLen; k++) {
                    output[i][h][j] += qk[i][k] * value[k][kvH][j];
                }
            }
        }
    }
    return output;
}

// Generate block diagonal mask for packed requests
VARP generateBlockDiagonalMask(const std::vector<int>& seqLens) {
    int maskSize = 0;
    for (int len : seqLens) {
        maskSize += len * len;
    }

    VARP maskVar = _Input({1, 1, 1, maskSize}, NCHW, halide_type_of<float>());
    float* ptr = maskVar->writeMap<float>();
    int offset = 0;
    for (int len : seqLens) {
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                ptr[offset + i * len + j] = (j <= i) ? 0.0f : std::numeric_limits<float>::lowest();
            }
        }
        offset += len * len;
    }
    maskVar->unMap();
    return maskVar;
}

VARP generateBlockDiagonalIntMask(const std::vector<int>& seqLens) {
    int maskSize = 0;
    for (int len : seqLens) {
        maskSize += len * len;
    }

    VARP maskVar = _Input({1, 1, 1, maskSize}, NCHW, halide_type_of<int>());
    int* ptr = maskVar->writeMap<int>();
    int offset = 0;
    for (int len : seqLens) {
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                ptr[offset + i * len + j] = j <= i ? 1 : 0;
            }
        }
        offset += len * len;
    }
    maskVar->unMap();
    return maskVar;
}

VARP generatePackedFullCausalMask(const std::vector<int>& seqLens, const std::vector<int>& kvSeqLens) {
    int maskSize = 0;
    for (int i = 0; i < seqLens.size(); ++i) {
        maskSize += seqLens[i] * kvSeqLens[i];
    }

    VARP maskVar = _Input({1, 1, 1, maskSize}, NCHW, halide_type_of<float>());
    float* ptr = maskVar->writeMap<float>();
    int offset = 0;
    for (int r = 0; r < seqLens.size(); ++r) {
        const int seqLen = seqLens[r];
        const int kvSeqLen = kvSeqLens[r];
        const int historyLen = kvSeqLen - seqLen;
        for (int i = 0; i < seqLen; ++i) {
            for (int j = 0; j < kvSeqLen; ++j) {
                ptr[offset + i * kvSeqLen + j] = (j <= historyLen + i) ? 0.0f : std::numeric_limits<float>::lowest();
            }
        }
        offset += seqLen * kvSeqLen;
    }
    maskVar->unMap();
    return maskVar;
}

VARP generatePackedFloatMask(const std::vector<std::vector<std::vector<int>>>& masks) {
    int maskSize = 0;
    for (auto& mask : masks) {
        maskSize += static_cast<int>(mask.size() * mask[0].size());
    }

    VARP maskVar = _Input({1, 1, 1, maskSize}, NCHW, halide_type_of<float>());
    float* ptr = maskVar->writeMap<float>();
    const float minVal = std::numeric_limits<float>::lowest();
    int offset = 0;
    for (auto& mask : masks) {
        const int rows = static_cast<int>(mask.size());
        const int cols = static_cast<int>(mask[0].size());
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ptr[offset + i * cols + j] = mask[i][j] ? 0.0f : minVal;
            }
        }
        offset += rows * cols;
    }
    maskVar->unMap();
    return maskVar;
}

VARP generatePackedIntMask(const std::vector<std::vector<std::vector<int>>>& masks) {
    int maskSize = 0;
    for (auto& mask : masks) {
        maskSize += static_cast<int>(mask.size() * mask[0].size());
    }

    VARP maskVar = _Input({1, 1, 1, maskSize}, NCHW, halide_type_of<int>());
    int* ptr = maskVar->writeMap<int>();
    int offset = 0;
    for (auto& mask : masks) {
        const int rows = static_cast<int>(mask.size());
        const int cols = static_cast<int>(mask[0].size());
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ptr[offset + i * cols + j] = mask[i][j] ? 1 : 0;
            }
        }
        offset += rows * cols;
    }
    maskVar->unMap();
    return maskVar;
}

static std::vector<std::vector<int>> generateCausalMaskHost(int seqLen, int kvSeqLen) {
    std::vector<std::vector<int>> mask(seqLen, std::vector<int>(kvSeqLen, 0));
    const int historyLen = kvSeqLen - seqLen;
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j <= historyLen + i && j < kvSeqLen; ++j) {
            mask[i][j] = 1;
        }
    }
    return mask;
}

static bool compareRequestOutput(const float* outPtr,
                                 int packedOffset,
                                 const std::vector<std::vector<std::vector<float>>>& expected,
                                 const std::string& reqName) {
    for (int i = 0; i < expected.size(); ++i) {
        for (int j = 0; j < gPackedNumHead; ++j) {
            for (int k = 0; k < gPackedHeadDim; ++k) {
                const int packedIndex = (packedOffset + i) * gPackedNumHead * gPackedHeadDim + j * gPackedHeadDim + k;
                const float actual = outPtr[packedIndex];
                const float target = expected[i][j][k];
                const float diff = fabs(actual - target);
                const float diffPercent = fabs(diff / (fabs(target) + 1e-6f));
                if (diff > gPackedDiffThreshold && diffPercent > gPackedDiffPercentThreshold) {
                    MNN_PRINT("%s mismatch at [%d][%d][%d]: expected %f, got %f, diff %f\n",
                              reqName.c_str(), i, j, k, target, actual, diff);
                    return false;
                }
            }
        }
    }
    return true;
}

static bool compareRequestOutputStrict(const float* outPtr,
                                       int packedOffset,
                                       const std::vector<std::vector<std::vector<float>>>& expected,
                                       const std::string& reqName,
                                       float threshold) {
    for (int i = 0; i < expected.size(); ++i) {
        for (int j = 0; j < gPackedNumHead; ++j) {
            for (int k = 0; k < gPackedHeadDim; ++k) {
                const int packedIndex = (packedOffset + i) * gPackedNumHead * gPackedHeadDim + j * gPackedHeadDim + k;
                const float actual = outPtr[packedIndex];
                const float target = expected[i][j][k];
                const float diff = fabs(actual - target);
                if (diff > threshold) {
                    MNN_PRINT("%s strict mismatch at [%d][%d][%d]: expected %f, got %f, diff %f\n",
                              reqName.c_str(), i, j, k, target, actual, diff);
                    return false;
                }
            }
        }
    }
    return true;
}

class PackedAttentionTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        srand(2025);

        // Test 0: Single Qwen-shaped prefill with non-periodic input.
        // This catches layout/softmax mistakes that periodic data can hide.
        {
            const int reqLen = 14;
            clearBatchMeta(gBatchMeta);

            auto query = generateSensitiveTensor(reqLen, gPackedNumHead, gPackedHeadDim, 17);
            auto key = generateSensitiveTensor(reqLen, gPackedKvNumHead, gPackedHeadDim, 29);
            auto value = generateSensitiveTensor(reqLen, gPackedKvNumHead, gPackedHeadDim, 41);

            VARP packedQ = packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{query});
            VARP packedK = packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{key});
            VARP packedV = packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{value});
            VARP mask = generateBlockDiagonalMask(std::vector<int>{reqLen});

            setKVCacheInfo(gBatchMeta, 0, reqLen);
            auto module = _makePackedAttentionModule();
            auto output = module->onForward({packedQ, packedK, packedV, mask})[0];
            syncBatchMeta(gBatchMeta);

            auto ref = computeSingleAttention(query, key, value, generateCausalMaskHost(reqLen, reqLen), reqLen, reqLen);
            const float* outPtr = output->readMap<float>();
            float threshold = precision == 2 ? 0.02f : 0.002f;
            bool pass = compareRequestOutputStrict(outPtr, 0, ref, "SensitivePrefill", threshold);
            output->unMap();
            if (!pass) {
                return false;
            }
        }

        // Test 1: Two requests with different lengths
        {
            int req0Len = 14;
            int req1Len = 7;
            std::vector<int> seqLens = {req0Len, req1Len};
            int totalLen = req0Len + req1Len;
            
            // Generate data for each request
            auto query0 = generateRandTensor(req0Len, gPackedNumHead, gPackedHeadDim, precision);
            auto key0 = generateRandTensor(req0Len, gPackedKvNumHead, gPackedHeadDim, precision);
            auto value0 = generateRandTensor(req0Len, gPackedKvNumHead, gPackedHeadDim, precision);
            
            auto query1 = generateRandTensor(req1Len, gPackedNumHead, gPackedHeadDim, precision);
            auto key1 = generateRandTensor(req1Len, gPackedKvNumHead, gPackedHeadDim, precision);
            auto value1 = generateRandTensor(req1Len, gPackedKvNumHead, gPackedHeadDim, precision);
            
            // Pack requests
            std::vector<std::vector<std::vector<std::vector<float>>>> queries = {query0, query1};
            std::vector<std::vector<std::vector<std::vector<float>>>> keys = {key0, key1};
            std::vector<std::vector<std::vector<std::vector<float>>>> values = {value0, value1};
            
            VARP packedQ = packRequests(queries);
            VARP packedK = packRequests(keys);
            VARP packedV = packRequests(values);
            VARP mask = generateBlockDiagonalIntMask(seqLens);
            
            // Setup BatchKVMeta
            clearBatchMeta(gBatchMeta);
            setKVCacheInfo(gBatchMeta, 0, req0Len);
            setKVCacheInfo(gBatchMeta, 1, req1Len);
            
            // Run packed attention
            auto module = _makePackedAttentionModule();
            auto output = module->onForward({packedQ, packedK, packedV, mask})[0];
            syncBatchMeta(gBatchMeta);
            
            // Verify output shape
            auto info = output->getInfo();
            if (info->dim[1] != totalLen) {
                MNN_PRINT("Error: Output seq_len mismatch. Expected %d, got %d\n", totalLen, (int)info->dim[1]);
                return false;
            }
            
            // Compute reference outputs for each request separately
            auto ref0 = computeSingleAttention(query0, key0, value0, generateCausalMaskHost(req0Len, req0Len), req0Len, req0Len);
            auto ref1 = computeSingleAttention(query1, key1, value1, generateCausalMaskHost(req1Len, req1Len), req1Len, req1Len);
            
            // Compare results
            const float* outPtr = output->readMap<float>();
            bool pass = compareRequestOutput(outPtr, 0, ref0, "Req0");
            if (pass) {
                pass = compareRequestOutput(outPtr, req0Len, ref1, "Req1");
            }
            
            output->unMap();
            
            if (!pass) {
                return false;
            }
        }
        
        // Test 2: Three requests
        {
            std::vector<int> seqLens = {256, 128, 1};
            int totalLen = 0;
            for (int len : seqLens) {
                totalLen += len;
            }
            
            std::vector<std::vector<std::vector<std::vector<float>>>> queries, keys, values;
            for (int r = 0; r < 3; r++) {
                queries.push_back(generateRandTensor(seqLens[r], gPackedNumHead, gPackedHeadDim, precision));
                keys.push_back(generateRandTensor(seqLens[r], gPackedKvNumHead, gPackedHeadDim, precision));
                values.push_back(generateRandTensor(seqLens[r], gPackedKvNumHead, gPackedHeadDim, precision));
            }
            
            VARP packedQ = packRequests(queries);
            VARP packedK = packRequests(keys);
            VARP packedV = packRequests(values);
            VARP mask = generateBlockDiagonalMask(seqLens);
            
            clearBatchMeta(gBatchMeta);
            for (int r = 0; r < 3; r++) {
                setKVCacheInfo(gBatchMeta, r, seqLens[r]);
            }
            
            auto module = _makePackedAttentionModule();
            auto output = module->onForward({packedQ, packedK, packedV, mask})[0];
            syncBatchMeta(gBatchMeta);
            
            auto info = output->getInfo();
            const size_t expectedSize = (size_t)totalLen * gPackedNumHead * gPackedHeadDim;
            if (info->dim.size() < 2 || info->dim[1] != totalLen || info->size != expectedSize) {
                MNN_PRINT("Error: Output shape mismatch. Expected seq_len=%d, element_size=%zu, got seq_len=%d, element_size=%zu\n",
                          totalLen, expectedSize,
                          info->dim.size() > 1 ? (int)info->dim[1] : -1, info->size);
                return false;
            }

            std::vector<std::vector<std::vector<std::vector<float>>>> refs;
            refs.reserve(seqLens.size());
            for (int r = 0; r < seqLens.size(); ++r) {
                refs.push_back(computeSingleAttention(queries[r], keys[r], values[r],
                                                      generateCausalMaskHost(seqLens[r], seqLens[r]),
                                                      seqLens[r], seqLens[r]));
            }

            const float* outPtr = output->readMap<float>();
            bool pass = true;
            int offset = 0;
            for (int r = 0; r < seqLens.size() && pass; ++r) {
                pass = compareRequestOutput(outPtr, offset, refs[r], std::string("Req") + std::to_string(r));
                offset += seqLens[r];
            }
            output->unMap();

            if (!pass) {
                return false;
            }
        }

        // Test 3: Mixed prefill and decode with existing KV cache
        {
            const int historyLen = 128;
            const int req0Len = 256;
            const int req1DecodeLen = 1;
            clearBatchMeta(gBatchMeta);

            auto module = _makePackedAttentionModule();

            auto req1QueryAll = generateRandTensor(historyLen + req1DecodeLen, gPackedNumHead, gPackedHeadDim, precision);
            auto req1KeyAll = generateRandTensor(historyLen + req1DecodeLen, gPackedKvNumHead, gPackedHeadDim, precision);
            auto req1ValueAll = generateRandTensor(historyLen + req1DecodeLen, gPackedKvNumHead, gPackedHeadDim, precision);

            auto req1QueryHistory = sliceTensor(req1QueryAll, 0, historyLen);
            auto req1KeyHistory = sliceTensor(req1KeyAll, 0, historyLen);
            auto req1ValueHistory = sliceTensor(req1ValueAll, 0, historyLen);
            auto req1QueryDecode = sliceTensor(req1QueryAll, historyLen, req1DecodeLen);
            auto req1KeyDecode = sliceTensor(req1KeyAll, historyLen, req1DecodeLen);
            auto req1ValueDecode = sliceTensor(req1ValueAll, historyLen, req1DecodeLen);

            {
                std::vector<int> seqLens = {historyLen};
                std::vector<std::vector<std::vector<std::vector<float>>>> queries = {req1QueryHistory};
                std::vector<std::vector<std::vector<std::vector<float>>>> keys = {req1KeyHistory};
                std::vector<std::vector<std::vector<std::vector<float>>>> values = {req1ValueHistory};

                setKVCacheInfo(gBatchMeta, 1, historyLen);
                auto seedOutput = module->onForward({packRequests(queries), packRequests(keys), packRequests(values), generateBlockDiagonalMask(seqLens)})[0];
                (void)seedOutput;
                syncBatchMeta(gBatchMeta);
                if (gBatchMeta.mMetas[1] == nullptr || gBatchMeta.mMetas[1]->previous != (size_t)historyLen) {
                    MNN_PRINT("Error: Failed to seed KV cache for req1, expected previous=%d, got %zu\n",
                              historyLen, gBatchMeta.mMetas[1] == nullptr ? 0 : gBatchMeta.mMetas[1]->previous);
                    return false;
                }
            }

            auto req0Query = generateRandTensor(req0Len, gPackedNumHead, gPackedHeadDim, precision);
            auto req0Key = generateRandTensor(req0Len, gPackedKvNumHead, gPackedHeadDim, precision);
            auto req0Value = generateRandTensor(req0Len, gPackedKvNumHead, gPackedHeadDim, precision);

            std::vector<int> seqLens = {req0Len, req1DecodeLen};
            std::vector<std::vector<std::vector<std::vector<float>>>> queries = {req0Query, req1QueryDecode};
            std::vector<std::vector<std::vector<std::vector<float>>>> keys = {req0Key, req1KeyDecode};
            std::vector<std::vector<std::vector<std::vector<float>>>> values = {req0Value, req1ValueDecode};

            setKVCacheInfo(gBatchMeta, 0, req0Len);
            setKVCacheInfo(gBatchMeta, 1, req1DecodeLen);
            auto output = module->onForward({packRequests(queries), packRequests(keys), packRequests(values), generateBlockDiagonalMask(seqLens)})[0];
            syncBatchMeta(gBatchMeta);

            auto info = output->getInfo();
            const int totalLen = req0Len + req1DecodeLen;
            if (info->dim.size() < 2 || info->dim[1] != totalLen) {
                MNN_PRINT("Error: Test3 output seq_len mismatch. Expected %d, got %d\n",
                          totalLen, info->dim.size() > 1 ? (int)info->dim[1] : -1);
                return false;
            }

            auto req0Ref = computeSingleAttention(req0Query, req0Key, req0Value,
                                                  generateCausalMaskHost(req0Len, req0Len), req0Len, req0Len);
            auto req1KeyTotal = concatTensor(req1KeyHistory, req1KeyDecode);
            auto req1ValueTotal = concatTensor(req1ValueHistory, req1ValueDecode);
            auto req1Ref = computeSingleAttention(req1QueryDecode, req1KeyTotal, req1ValueTotal,
                                                  generateCausalMaskHost(req1DecodeLen, historyLen + req1DecodeLen),
                                                  req1DecodeLen, historyLen + req1DecodeLen);

            const float* outPtr = output->readMap<float>();
            bool pass = compareRequestOutput(outPtr, 0, req0Ref, "Req0");
            if (pass) {
                pass = compareRequestOutput(outPtr, req0Len, req1Ref, "Req1Decode");
            }
            output->unMap();

            if (!pass) {
                return false;
            }
            if (gBatchMeta.mMetas[1] == nullptr || gBatchMeta.mMetas[1]->previous != (size_t)(historyLen + req1DecodeLen)) {
                MNN_PRINT("Error: Test3 KV cache length mismatch for req1, expected %d, got %zu\n",
                          historyLen + req1DecodeLen, gBatchMeta.mMetas[1] == nullptr ? 0 : gBatchMeta.mMetas[1]->previous);
                return false;
            }
        }

        // Test 4: Multi-request multi-step decode with existing KV cache
        {
            const std::vector<int> historyLens = {33, 65, 17, 96};
            const int decodeSteps = 4;
            const int reqCount = (int)historyLens.size();
            clearBatchMeta(gBatchMeta);

            auto module = _makePackedAttentionModule();

            std::vector<std::vector<std::vector<std::vector<float>>>> allQuery(reqCount);
            std::vector<std::vector<std::vector<std::vector<float>>>> allKey(reqCount);
            std::vector<std::vector<std::vector<std::vector<float>>>> allValue(reqCount);
            std::vector<std::vector<std::vector<std::vector<float>>>> kvKeyPrefix(reqCount);
            std::vector<std::vector<std::vector<std::vector<float>>>> kvValuePrefix(reqCount);

            for (int r = 0; r < reqCount; ++r) {
                allQuery[r] = generateRandTensor(historyLens[r] + decodeSteps, gPackedNumHead, gPackedHeadDim, precision);
                allKey[r] = generateRandTensor(historyLens[r] + decodeSteps, gPackedKvNumHead, gPackedHeadDim, precision);
                allValue[r] = generateRandTensor(historyLens[r] + decodeSteps, gPackedKvNumHead, gPackedHeadDim, precision);

                auto historyQuery = sliceTensor(allQuery[r], 0, historyLens[r]);
                auto historyKey = sliceTensor(allKey[r], 0, historyLens[r]);
                auto historyValue = sliceTensor(allValue[r], 0, historyLens[r]);

                setKVCacheInfo(gBatchMeta, r, historyLens[r]);
                auto seedOutput = module->onForward({
                    packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{historyQuery}),
                    packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{historyKey}),
                    packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{historyValue}),
                    generateBlockDiagonalMask(std::vector<int>{historyLens[r]})
                })[0];
                (void)seedOutput;
                syncBatchMeta(gBatchMeta);

                kvKeyPrefix[r] = historyKey;
                kvValuePrefix[r] = historyValue;
                if (gBatchMeta.mMetas[r] == nullptr || gBatchMeta.mMetas[r]->previous != (size_t)historyLens[r]) {
                    MNN_PRINT("Error: Failed to seed KV cache for req%d, expected previous=%d, got %zu\n",
                              r, historyLens[r], gBatchMeta.mMetas[r] == nullptr ? 0 : gBatchMeta.mMetas[r]->previous);
                    return false;
                }
            }

            for (int step = 0; step < decodeSteps; ++step) {
                std::vector<int> seqLens(reqCount, 1);
                std::vector<std::vector<std::vector<std::vector<float>>>> decodeQueries;
                std::vector<std::vector<std::vector<std::vector<float>>>> decodeKeys;
                std::vector<std::vector<std::vector<std::vector<float>>>> decodeValues;
                decodeQueries.reserve(reqCount);
                decodeKeys.reserve(reqCount);
                decodeValues.reserve(reqCount);

                for (int r = 0; r < reqCount; ++r) {
                    decodeQueries.push_back(sliceTensor(allQuery[r], historyLens[r] + step, 1));
                    decodeKeys.push_back(sliceTensor(allKey[r], historyLens[r] + step, 1));
                    decodeValues.push_back(sliceTensor(allValue[r], historyLens[r] + step, 1));
                    setKVCacheInfo(gBatchMeta, r, 1);
                }

                auto output = module->onForward({
                    packRequests(decodeQueries),
                    packRequests(decodeKeys),
                    packRequests(decodeValues),
                    generateBlockDiagonalMask(seqLens)
                })[0];
                syncBatchMeta(gBatchMeta);

                const float* outPtr = output->readMap<float>();
                bool pass = true;
                for (int r = 0; r < reqCount && pass; ++r) {
                    auto currentKeyTotal = concatTensor(kvKeyPrefix[r], decodeKeys[r]);
                    auto currentValueTotal = concatTensor(kvValuePrefix[r], decodeValues[r]);
                    auto ref = computeSingleAttention(decodeQueries[r], currentKeyTotal, currentValueTotal,
                                                      generateCausalMaskHost(1, (int)currentKeyTotal.size()),
                                                      1, (int)currentKeyTotal.size());
                    pass = compareRequestOutput(outPtr, r, ref,
                                                std::string("Req") + std::to_string(r) + "_Step" + std::to_string(step));
                    kvKeyPrefix[r] = currentKeyTotal;
                    kvValuePrefix[r] = currentValueTotal;
                    if (gBatchMeta.mMetas[r] == nullptr ||
                        gBatchMeta.mMetas[r]->previous != (size_t)(historyLens[r] + step + 1)) {
                        MNN_PRINT("Error: Test4 KV cache length mismatch for req%d at step %d, expected %d, got %zu\n",
                                  r, step, historyLens[r] + step + 1,
                                  gBatchMeta.mMetas[r] == nullptr ? 0 : gBatchMeta.mMetas[r]->previous);
                        output->unMap();
                        return false;
                    }
                }
                output->unMap();

                if (!pass) {
                    return false;
                }
            }
        }

        // Test 5: Full KV mask after history prefill.
        // Eagle draft/tree steps can pass rectangular reqLen x kvSeqLen masks.
        {
            const int historyLen = 32;
            const int stepLen = 5;
            clearBatchMeta(gBatchMeta);

            auto module = _makePackedAttentionModule();

            auto allQuery = generateRandTensor(historyLen + stepLen, gPackedNumHead, gPackedHeadDim, precision);
            auto allKey = generateRandTensor(historyLen + stepLen, gPackedKvNumHead, gPackedHeadDim, precision);
            auto allValue = generateRandTensor(historyLen + stepLen, gPackedKvNumHead, gPackedHeadDim, precision);

            auto historyQuery = sliceTensor(allQuery, 0, historyLen);
            auto historyKey = sliceTensor(allKey, 0, historyLen);
            auto historyValue = sliceTensor(allValue, 0, historyLen);
            auto stepQuery = sliceTensor(allQuery, historyLen, stepLen);
            auto stepKey = sliceTensor(allKey, historyLen, stepLen);
            auto stepValue = sliceTensor(allValue, historyLen, stepLen);

            setKVCacheInfo(gBatchMeta, 0, historyLen);
            auto seedOutput = module->onForward({
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{historyQuery}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{historyKey}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{historyValue}),
                generateBlockDiagonalMask(std::vector<int>{historyLen})
            })[0];
            (void)seedOutput;
            syncBatchMeta(gBatchMeta);

            setKVCacheInfo(gBatchMeta, 0, stepLen);
            auto output = module->onForward({
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{stepQuery}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{stepKey}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{stepValue}),
                generatePackedFullCausalMask(std::vector<int>{stepLen}, std::vector<int>{historyLen + stepLen})
            })[0];
            syncBatchMeta(gBatchMeta);

            auto expectedKey = concatTensor(historyKey, stepKey);
            auto expectedValue = concatTensor(historyValue, stepValue);
            auto ref = computeSingleAttention(stepQuery, expectedKey, expectedValue,
                                              generateCausalMaskHost(stepLen, historyLen + stepLen),
                                              stepLen, historyLen + stepLen);

            const float* outPtr = output->readMap<float>();
            bool pass = compareRequestOutput(outPtr, 0, ref, "FullMaskStep");
            output->unMap();
            if (!pass) {
                return false;
            }
        }

        // Test 6: Full-KV tree mask with irregular KV length.
        // Eagle draft verification needs a non-causal tree mask over current
        // tokens while all retained history remains visible.
        {
            const int historyLen = 6;
            const int stepLen = 5;
            const int kvSeqLen = historyLen + stepLen;
            clearBatchMeta(gBatchMeta);

            auto module = _makePackedAttentionModule();

            auto allQuery = generateRandTensor(kvSeqLen, gPackedNumHead, gPackedHeadDim, precision);
            auto allKey = generateRandTensor(kvSeqLen, gPackedKvNumHead, gPackedHeadDim, precision);
            auto allValue = generateRandTensor(kvSeqLen, gPackedKvNumHead, gPackedHeadDim, precision);

            auto historyQuery = sliceTensor(allQuery, 0, historyLen);
            auto historyKey = sliceTensor(allKey, 0, historyLen);
            auto historyValue = sliceTensor(allValue, 0, historyLen);
            auto stepQuery = sliceTensor(allQuery, historyLen, stepLen);
            auto stepKey = sliceTensor(allKey, historyLen, stepLen);
            auto stepValue = sliceTensor(allValue, historyLen, stepLen);

            setKVCacheInfo(gBatchMeta, 0, historyLen);
            auto seedOutput = module->onForward({
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{historyQuery}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{historyKey}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{historyValue}),
                generateBlockDiagonalMask(std::vector<int>{historyLen})
            })[0];
            (void)seedOutput;
            syncBatchMeta(gBatchMeta);

            std::vector<std::vector<int>> treeMask(stepLen, std::vector<int>(kvSeqLen, 0));
            for (int i = 0; i < stepLen; ++i) {
                for (int j = 0; j < historyLen; ++j) {
                    treeMask[i][j] = 1;
                }
                treeMask[i][historyLen + i] = 1;
            }
            treeMask[1][historyLen + 0] = 1;
            treeMask[3][historyLen + 1] = 1;
            treeMask[4][historyLen + 0] = 1;
            treeMask[4][historyLen + 2] = 1;

            setKVCacheInfo(gBatchMeta, 0, stepLen);
            auto output = module->onForward({
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{stepQuery}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{stepKey}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{stepValue}),
                generatePackedIntMask(std::vector<std::vector<std::vector<int>>>{treeMask})
            })[0];
            syncBatchMeta(gBatchMeta);

            auto expectedKey = concatTensor(historyKey, stepKey);
            auto expectedValue = concatTensor(historyValue, stepValue);
            auto ref = computeSingleAttention(stepQuery, expectedKey, expectedValue, treeMask, stepLen, kvSeqLen);

            const float* outPtr = output->readMap<float>();
            bool pass = compareRequestOutput(outPtr, 0, ref, "FullTreeMaskStep");
            output->unMap();
            if (!pass) {
                return false;
            }
        }

        // Test 7: Reserve draft KV tokens before decode.
        // This covers CPUKVCacheManager::moveKV value-cache indexing used by Eagle.
        {
            const int historyLen = 160;
            const int draftLen = 7;
            const int decodeLen = 1;
            const int reservedCount = 2;
            int reserve[reservedCount * 2] = {
                2, 1,
                5, 1,
            };
            clearBatchMeta(gBatchMeta);

            auto module = _makePackedAttentionModule();

            auto allQuery = generateRandTensor(historyLen + draftLen + decodeLen, gPackedNumHead, gPackedHeadDim, precision);
            auto allKey = generateRandTensor(historyLen + draftLen + decodeLen, gPackedKvNumHead, gPackedHeadDim, precision);
            auto allValue = generateRandTensor(historyLen + draftLen + decodeLen, gPackedKvNumHead, gPackedHeadDim, precision);

            auto seedQuery = sliceTensor(allQuery, 0, historyLen + draftLen);
            auto seedKey = sliceTensor(allKey, 0, historyLen + draftLen);
            auto seedValue = sliceTensor(allValue, 0, historyLen + draftLen);
            auto decodeQuery = sliceTensor(allQuery, historyLen + draftLen, decodeLen);
            auto decodeKey = sliceTensor(allKey, historyLen + draftLen, decodeLen);
            auto decodeValue = sliceTensor(allValue, historyLen + draftLen, decodeLen);

            setKVCacheInfo(gBatchMeta, 0, historyLen + draftLen);
            auto seedOutput = module->onForward({
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{seedQuery}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{seedKey}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{seedValue}),
                generateBlockDiagonalMask(std::vector<int>{historyLen + draftLen})
            })[0];
            (void)seedOutput;
            syncBatchMeta(gBatchMeta);

            setKVCacheInfo(gBatchMeta, 0, decodeLen, draftLen, reserve, reservedCount);
            auto output = module->onForward({
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{decodeQuery}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{decodeKey}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{decodeValue}),
                generateBlockDiagonalMask(std::vector<int>{decodeLen})
            })[0];
            syncBatchMeta(gBatchMeta);

            auto expectedKey = sliceTensor(allKey, 0, historyLen);
            auto expectedValue = sliceTensor(allValue, 0, historyLen);
            for (int i = 0; i < reservedCount; ++i) {
                expectedKey = concatTensor(expectedKey, sliceTensor(allKey, historyLen + reserve[2 * i], reserve[2 * i + 1]));
                expectedValue = concatTensor(expectedValue, sliceTensor(allValue, historyLen + reserve[2 * i], reserve[2 * i + 1]));
            }
            expectedKey = concatTensor(expectedKey, decodeKey);
            expectedValue = concatTensor(expectedValue, decodeValue);
            auto ref = computeSingleAttention(decodeQuery, expectedKey, expectedValue,
                                              generateCausalMaskHost(decodeLen, (int)expectedKey.size()),
                                              decodeLen, (int)expectedKey.size());

            const float* outPtr = output->readMap<float>();
            bool pass = compareRequestOutput(outPtr, 0, ref, "ReserveDecode");
            output->unMap();
            if (!pass) {
                return false;
            }
            const size_t expectedPrevious = historyLen + reservedCount + decodeLen;
            if (gBatchMeta.mMetas[0] == nullptr || gBatchMeta.mMetas[0]->previous != expectedPrevious) {
                MNN_PRINT("Error: Test5 KV cache length mismatch, expected %zu, got %zu\n",
                          expectedPrevious, gBatchMeta.mMetas[0] == nullptr ? 0 : gBatchMeta.mMetas[0]->previous);
                return false;
            }
        }

        // Test 8: Reserve draft KV tokens before another multi-token tree step.
        // Eagle verifies a draft tree, compacts accepted draft tokens, then appends
        // the next draft tree with a square mask over only the newly appended rows.
        {
            const int historyLen = 96;
            const int draftLen = 7;
            const int nextTreeLen = 5;
            const int reservedCount = 3;
            int reserve[reservedCount * 2] = {
                0, 1,
                2, 1,
                5, 1,
            };
            clearBatchMeta(gBatchMeta);

            auto module = _makePackedAttentionModule();

            auto allQuery = generateRandTensor(historyLen + draftLen + nextTreeLen, gPackedNumHead, gPackedHeadDim, precision);
            auto allKey = generateRandTensor(historyLen + draftLen + nextTreeLen, gPackedKvNumHead, gPackedHeadDim, precision);
            auto allValue = generateRandTensor(historyLen + draftLen + nextTreeLen, gPackedKvNumHead, gPackedHeadDim, precision);

            auto seedQuery = sliceTensor(allQuery, 0, historyLen + draftLen);
            auto seedKey = sliceTensor(allKey, 0, historyLen + draftLen);
            auto seedValue = sliceTensor(allValue, 0, historyLen + draftLen);
            auto nextQuery = sliceTensor(allQuery, historyLen + draftLen, nextTreeLen);
            auto nextKey = sliceTensor(allKey, historyLen + draftLen, nextTreeLen);
            auto nextValue = sliceTensor(allValue, historyLen + draftLen, nextTreeLen);

            setKVCacheInfo(gBatchMeta, 0, historyLen + draftLen);
            auto seedOutput = module->onForward({
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{seedQuery}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{seedKey}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{seedValue}),
                generateBlockDiagonalMask(std::vector<int>{historyLen + draftLen})
            })[0];
            (void)seedOutput;
            syncBatchMeta(gBatchMeta);

            setKVCacheInfo(gBatchMeta, 0, nextTreeLen, draftLen, reserve, reservedCount);
            auto output = module->onForward({
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{nextQuery}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{nextKey}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{nextValue}),
                generateBlockDiagonalMask(std::vector<int>{nextTreeLen})
            })[0];
            syncBatchMeta(gBatchMeta);

            auto expectedKey = sliceTensor(allKey, 0, historyLen);
            auto expectedValue = sliceTensor(allValue, 0, historyLen);
            for (int i = 0; i < reservedCount; ++i) {
                expectedKey = concatTensor(expectedKey, sliceTensor(allKey, historyLen + reserve[2 * i], reserve[2 * i + 1]));
                expectedValue = concatTensor(expectedValue, sliceTensor(allValue, historyLen + reserve[2 * i], reserve[2 * i + 1]));
            }
            expectedKey = concatTensor(expectedKey, nextKey);
            expectedValue = concatTensor(expectedValue, nextValue);
            auto ref = computeSingleAttention(nextQuery, expectedKey, expectedValue,
                                              generateCausalMaskHost(nextTreeLen, (int)expectedKey.size()),
                                              nextTreeLen, (int)expectedKey.size());

            const float* outPtr = output->readMap<float>();
            bool pass = compareRequestOutput(outPtr, 0, ref, "ReserveTreeStep");
            output->unMap();
            if (!pass) {
                return false;
            }
            const size_t expectedPrevious = historyLen + reservedCount + nextTreeLen;
            if (gBatchMeta.mMetas[0] == nullptr || gBatchMeta.mMetas[0]->previous != expectedPrevious) {
                MNN_PRINT("Error: Test7 KV cache length mismatch, expected %zu, got %zu\n",
                          expectedPrevious, gBatchMeta.mMetas[0] == nullptr ? 0 : gBatchMeta.mMetas[0]->previous);
                return false;
            }
        }

        // Test 9: A single-token step with a full-KV mask must not use the
        // maskless decode fast path. Eagle can produce this shape for narrow trees.
        {
            const int historyLen = 6;
            const int stepLen = 1;
            const int kvSeqLen = historyLen + stepLen;
            clearBatchMeta(gBatchMeta);

            auto module = _makePackedAttentionModule();
            auto allQuery = generateSensitiveTensor(kvSeqLen, gPackedNumHead, gPackedHeadDim, 101);
            auto allKey = generateSensitiveTensor(kvSeqLen, gPackedKvNumHead, gPackedHeadDim, 103);
            auto allValue = generateSensitiveTensor(kvSeqLen, gPackedKvNumHead, gPackedHeadDim, 107);
            for (int i = 0; i < kvSeqLen; ++i) {
                for (int h = 0; h < gPackedKvNumHead; ++h) {
                    for (int d = 0; d < gPackedHeadDim; ++d) {
                        allKey[i][h][d] = 0.0f;
                        allValue[i][h][d] = i == 1 ? 1.0f : -1.0f;
                    }
                }
            }
            auto historyQuery = sliceTensor(allQuery, 0, historyLen);
            auto historyKey = sliceTensor(allKey, 0, historyLen);
            auto historyValue = sliceTensor(allValue, 0, historyLen);
            auto stepQuery = sliceTensor(allQuery, historyLen, stepLen);
            auto stepKey = sliceTensor(allKey, historyLen, stepLen);
            auto stepValue = sliceTensor(allValue, historyLen, stepLen);

            setKVCacheInfo(gBatchMeta, 0, historyLen);
            auto seedOutput = module->onForward({
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{historyQuery}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{historyKey}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{historyValue}),
                generateBlockDiagonalMask(std::vector<int>{historyLen})
            })[0];
            (void)seedOutput;
            syncBatchMeta(gBatchMeta);

            std::vector<std::vector<int>> fullMask(stepLen, std::vector<int>(kvSeqLen, 0));
            fullMask[0][1] = 1;
            setKVCacheInfo(gBatchMeta, 0, stepLen);
            auto output = module->onForward({
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{stepQuery}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{stepKey}),
                packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>{stepValue}),
                generatePackedFloatMask(std::vector<std::vector<std::vector<int>>>{fullMask})
            })[0];
            syncBatchMeta(gBatchMeta);

            auto expectedKey = concatTensor(historyKey, stepKey);
            auto expectedValue = concatTensor(historyValue, stepValue);
            auto ref = computeSingleAttention(stepQuery, expectedKey, expectedValue,
                                              fullMask, stepLen, kvSeqLen);
            const float* outPtr = output->readMap<float>();
            bool pass = compareRequestOutputStrict(outPtr, 0, ref, "SingleTokenFullMask", 0.002f);
            output->unMap();
            if (!pass) {
                return false;
            }
        }
        
        return true;
    }
};

MNNTestSuiteRegister(PackedAttentionTest, "op/packed_attention");

#endif
