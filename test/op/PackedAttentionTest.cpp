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
#include <MNN/AutoTime.hpp>

using namespace MNN::Express;

// Test parameters (use different names to avoid conflict with AttentionTest.cpp)
static int gPackedNumHead   = 16;
static int gPackedKvNumHead = 2;
static int gPackedHeadDim   = 128;
static const float gPackedDiffThreshold = 0.01;
static const float gPackedDiffPercentThreshold = 0.1;

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

// BatchKVMeta for packed attention testing
struct BatchKVMeta {
    std::map<int, KVMeta*> mMetas;
    std::vector<int> remove;
    std::vector<int> calId;
    
    void sync() {
        for (int id : calId) {
            if (mMetas.find(id) != mMetas.end()) {
                mMetas[id]->sync();
            }
        }
        calId.clear();
    }
    
    void setKVCacheInfo(int req_id, size_t add = 0, size_t remove = 0, int* reserve = nullptr, int n_reserve = 0) {
        if (mMetas.find(req_id) == mMetas.end()) {
            mMetas[req_id] = new KVMeta();
            mMetas[req_id]->previous = 0;
        }
        KVMeta* mMeta = mMetas[req_id];
        if (remove > mMeta->previous) {
            remove = mMeta->previous;
        }
        mMeta->remove = remove;
        mMeta->reserve = reserve;
        mMeta->n_reserve = n_reserve;
        mMeta->add = add;
        if (add > 0) {
            calId.push_back(req_id);
        }
    }

    void setKVMetaInfo(int req_id, int layer_nums, int layer_index, int seqlen_in_disk, const std::string& file_name, int file_flag) {
        if(mMetas.find(req_id) == mMetas.end()) {
            mMetas[req_id] = new KVMeta();
        }
        KVMeta* mMeta = mMetas[req_id];
        mMeta->layer_index = layer_index;
        mMeta->layer_nums = layer_nums;
        mMeta->seqlen_in_disk = seqlen_in_disk;
        mMeta->file_name = file_name;
        mMeta->file_flag = file_flag;
    }
    
    void clear() {
        for (auto& kv : mMetas) {
            if (kv.second) {
                delete kv.second;
            }
        }
        mMetas.clear();
        calId.clear();
    }
    
    ~BatchKVMeta() {
        clear();
    }
};

static BatchKVMeta gBatchMeta;

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

static VARP packed_vector_to_var(std::vector<std::vector<std::vector<float>>>& a) {
    int seqLen = a.size();
    int numHead = a[0].size();
    int headDim = a[0][0].size();
    VARP var = _Input({1, seqLen, numHead, headDim}, NCHW, halide_type_of<float>());
    float* ptr = var->writeMap<float>();
    for (int i = 0; i < seqLen; i++) {
        for (int j = 0; j < numHead; j++) {
            for (int k = 0; k < headDim; k++) {
                ptr[i * numHead * headDim + j * headDim + k] = a[i][j][k];
            }
        }
    }
    var->unMap();
    return var;
}

// Pack multiple requests into single tensor along seq dimension
VARP packRequests(std::vector<std::vector<std::vector<std::vector<float>>>>& requests) {
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
    std::vector<std::vector<int>>& mask,
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
VARP generateBlockDiagonalMask(std::vector<int>& seqLens) {
    int totalSeqLen = 0;
    for (int len : seqLens) {
        totalSeqLen += len;
    }
    
    std::vector<std::vector<int>> mask(totalSeqLen, std::vector<int>(totalSeqLen, 0));
    
    int offset = 0;
    for (int len : seqLens) {
        // Create lower triangular mask for each request block
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < len; j++) {
                if (j <= i + offset) {
                    mask[i + offset][j + offset] = 1;
                }
            }
        }
        offset += len;
    }
    
    // Convert to VARP with -inf/0 format
    VARP maskVar = _Input({totalSeqLen, totalSeqLen}, NCHW, halide_type_of<float>());
    float* ptr = maskVar->writeMap<float>();
    for (int i = 0; i < totalSeqLen; i++) {
        for (int j = 0; j < totalSeqLen; j++) {
            ptr[i * totalSeqLen + j] = (mask[i][j] == 1) ? 0.0f : std::numeric_limits<float>::lowest();
        }
    }
    maskVar->unMap();
    return maskVar;
}

class PackedAttentionTest : public MNNTestCase {
public:
    virtual bool run(int precision) {
        srand(2025);
        
        // Test 1: Two requests with different lengths
        {
            MNN_PRINT("=== PackedAttention Test 1: Two requests ===\n");
            
            int req0Len = 4;
            int req1Len = 3;
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
            VARP mask = generateBlockDiagonalMask(seqLens);
            
            // Setup BatchKVMeta
            gBatchMeta.clear();
            gBatchMeta.setKVCacheInfo(0, req0Len);
            gBatchMeta.setKVCacheInfo(1, req1Len);
            
            // Run packed attention
            auto module = _makePackedAttentionModule();
            auto output = module->onForward({packedQ, packedK, packedV, mask})[0];
            gBatchMeta.sync();
            
            // Verify output shape
            auto info = output->getInfo();
            if (info->dim[1] != totalLen) {
                MNN_PRINT("Error: Output seq_len mismatch. Expected %d, got %d\n", totalLen, (int)info->dim[1]);
                return false;
            }
            
            // Compute reference outputs for each request separately
            std::vector<std::vector<int>> mask0(req0Len, std::vector<int>(req0Len, 1));
            for (int i = 0; i < req0Len; i++) {
                for (int j = 0; j < req0Len; j++) {
                    mask0[i][j] = (j <= i) ? 1 : 0;
                }
            }
            auto ref0 = computeSingleAttention(query0, key0, value0, mask0, req0Len, req0Len);
            
            std::vector<std::vector<int>> mask1(req1Len, std::vector<int>(req1Len, 1));
            for (int i = 0; i < req1Len; i++) {
                for (int j = 0; j < req1Len; j++) {
                    mask1[i][j] = (j <= i) ? 1 : 0;
                }
            }
            auto ref1 = computeSingleAttention(query1, key1, value1, mask1, req1Len, req1Len);
            
            // Compare results
            const float* outPtr = output->readMap<float>();
            bool pass = true;
            
            // Check request 0 output
            for (int i = 0; i < req0Len && pass; i++) {
                for (int j = 0; j < gPackedNumHead && pass; j++) {
                    for (int k = 0; k < gPackedHeadDim && pass; k++) {
                        float diff = fabs(outPtr[i * gPackedNumHead * gPackedHeadDim + j * gPackedHeadDim + k] - ref0[i][j][k]);
                        float diffPercent = fabs(diff / (fabs(ref0[i][j][k]) + 1e-6));
                        if (diff > gPackedDiffThreshold && diffPercent > gPackedDiffPercentThreshold) {
                            MNN_PRINT("Req0 Mismatch at [%d][%d][%d]: expected %f, got %f, diff %f\n",
                                     i, j, k, ref0[i][j][k], outPtr[i * gPackedNumHead * gPackedHeadDim + j * gPackedHeadDim + k], diff);
                            pass = false;
                        }
                    }
                }
            }
            
            // Check request 1 output
            int offset = req0Len;
            for (int i = 0; i < req1Len && pass; i++) {
                for (int j = 0; j < gPackedNumHead && pass; j++) {
                    for (int k = 0; k < gPackedHeadDim && pass; k++) {
                        float diff = fabs(outPtr[(offset + i) * gPackedNumHead * gPackedHeadDim + j * gPackedHeadDim + k] - ref1[i][j][k]);
                        float diffPercent = fabs(diff / (fabs(ref1[i][j][k]) + 1e-6));
                        if (diff > gPackedDiffThreshold && diffPercent > gPackedDiffPercentThreshold) {
                            MNN_PRINT("Req1 Mismatch at [%d][%d][%d]: expected %f, got %f, diff %f\n",
                                     i, j, k, ref1[i][j][k], outPtr[(offset + i) * gPackedNumHead * gPackedHeadDim + j * gPackedHeadDim + k], diff);
                            pass = false;
                        }
                    }
                }
            }
            
            output->unMap();
            
            if (!pass) {
                MNN_PRINT("PackedAttention Test 1 FAILED!\n");
                return false;
            }
            MNN_PRINT("PackedAttention Test 1 PASSED!\n");
        }
        
        // Test 2: Three requests
        {
            MNN_PRINT("=== PackedAttention Test 2: Three requests ===\n");
            
            std::vector<int> seqLens = {2, 3, 2};
            int totalLen = 7;
            
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
            
            gBatchMeta.clear();
            for (int r = 0; r < 3; r++) {
                gBatchMeta.setKVCacheInfo(r, seqLens[r]);
            }
            
            auto module = _makePackedAttentionModule();
            auto output = module->onForward({packedQ, packedK, packedV, mask})[0];
            gBatchMeta.sync();
            
            // Verify shape
            auto info = output->getInfo();
            if (info->dim[1] != totalLen) {
                MNN_PRINT("Error: Output seq_len mismatch. Expected %d, got %d\n", totalLen, (int)info->dim[1]);
                return false;
            }
            
            output->unMap();
            MNN_PRINT("PackedAttention Test 2 PASSED!\n");
        }
        
        return true;
    }
};

MNNTestSuiteRegister(PackedAttentionTest, "op/packed_attention");

#endif
