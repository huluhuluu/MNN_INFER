//
//  llm_profiler.cpp
//  MNN
//
//  Created for LLM Op Profiling
//

#include "llm/llm_profiler.hpp"
#include <MNN/MNNDefine.h>
#include <MNN/expr/ExecutorScope.hpp>
#include "core/Backend.hpp"  // For Runtime class
#include "core/TensorUtils.hpp"  // For TensorUtils::getDescribeOrigin
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <regex>
#include <cmath>

namespace MNN {
namespace Transformer {

// ========== PhaseProfile Methods ==========

std::map<std::string, OpRecord> PhaseProfile::getOpTypeStats() const {
    std::map<std::string, OpRecord> stats;
    
    for (const auto& pair : opRecords) {
        const OpRecord& record = pair.second;
        
        // For special ops, use opName as key (treat as unique type)
        std::string key = record.isSpecial ? 
            (record.name + "@" + record.backend) : 
            (record.type + "@" + record.backend);
        
        if (stats.find(key) == stats.end()) {
            stats[key] = OpRecord();
            // For special ops, keep original name; for normal ops, use type as name
            stats[key].name = record.isSpecial ? record.name : record.type;
            stats[key].type = record.type;
            stats[key].backend = record.backend;
            stats[key].isSpecial = record.isSpecial;
        }
        
        OpRecord& s = stats[key];
        s.totalTime += record.totalTime;
        s.callCount += record.callCount;
    }
    
    return stats;
}

// ========== LLMOpProfiler Methods ==========

LLMOpProfiler::LLMOpProfiler() {
}

LLMOpProfiler::~LLMOpProfiler() {
}

void LLMOpProfiler::setConfig(const Config& config) {
    mConfig = config;
    setSpecialOps(config.specialOps);
}

void LLMOpProfiler::setSpecialOps(const std::vector<std::string>& specialOps) {
    mSpecialOpPatterns = specialOps;
}

bool LLMOpProfiler::isSpecialOp(const std::string& opName) const {
    for (const auto& pattern : mSpecialOpPatterns) {
        if (matchPattern(opName, pattern)) {
            return true;
        }
    }
    return false;
}

bool LLMOpProfiler::matchPattern(const std::string& opName, const std::string& pattern) const {
    // Simple wildcard matching
    // Supports: exact match, prefix*, *suffix, *contains*
    if (pattern == opName) {
        return true;
    }
    
    // Check if pattern contains wildcard
    if (pattern.find('*') != std::string::npos) {
        // Convert wildcard pattern to prefix/suffix matching
        bool prefixMatch = false;
        bool suffixMatch = false;
        std::string searchText = pattern;
        
        if (!searchText.empty() && searchText.back() == '*') {
            suffixMatch = true;
            searchText.pop_back();
        }
        if (!searchText.empty() && searchText.front() == '*') {
            prefixMatch = true;
            searchText = searchText.substr(1);
        }
        
        if (prefixMatch && suffixMatch) {
            // *text* - contains match
            return opName.find(searchText) != std::string::npos;
        } else if (prefixMatch) {
            // *text - suffix match
            return opName.size() >= searchText.size() && 
                   opName.compare(opName.size() - searchText.size(), searchText.size(), searchText) == 0;
        } else if (suffixMatch) {
            // text* - prefix match
            return opName.size() >= searchText.size() && 
                   opName.compare(0, searchText.size(), searchText) == 0;
        }
        // No wildcards left after processing, exact match
        return opName == searchText;
    }
    
    return false;
}

std::string LLMOpProfiler::forwardTypeToString(MNNForwardType type) {
    switch (type) {
        case MNN_FORWARD_CPU: return "CPU";
        case MNN_FORWARD_CPU_EXTENSION: return "CPU";
        case MNN_FORWARD_METAL: return "Metal";
        case MNN_FORWARD_CUDA: return "CUDA";
        case MNN_FORWARD_OPENCL: return "OpenCL";
        case MNN_FORWARD_OPENGL: return "OpenGL";
        case MNN_FORWARD_VULKAN: return "Vulkan";
        case MNN_FORWARD_NN: return "QNN";
        default: return "Other";
    }
}

std::string LLMOpProfiler::shapeToString(const Express::INTS& shape) {
    if (shape.empty()) return "[]";
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

// ========== Phase Control ==========

void LLMOpProfiler::onPrefillStart() {
    if (!mEnabled) return;
    mInPrefill = true;
    // mPrefillProfile.reset();
    mTimer.reset(); 
    // printf("[LLM Profiler] Prefill phase started\n");
}

void LLMOpProfiler::onPrefillEnd(int promptTokenCount) {
    if (!mEnabled) return;
    mInPrefill = false;
    // Record token time for prefill
    mPrefillProfile.tokenTotalTime += mTimer.durationInUs() / 1000.0f;  // us -> ms
    mPrefillProfile.tokenCount += promptTokenCount;
    // reset timer for next prefill
    mTimer.reset(); 
    // printf("[LLM Profiler] Prefill phase ended, %d tokens, time %.4f ms, avg time: %.4f ms/token\n", 
            //   promptTokenCount, 
            //   mPrefillProfile.tokenTotalTime ,
            //   promptTokenCount==0 ? 0 : mPrefillProfile.tokenTotalTime  / promptTokenCount);
}

void LLMOpProfiler::onDecodeTokenBegin() {
    if (!mEnabled) return;
    mTimer.reset();  // Use separate timer for token-level timing
}

void LLMOpProfiler::onDecodeTokenEnd(int count) {
    if (!mEnabled || count <= 0) return;
    float tokenTime = mTimer.durationInUs() / 1000.0f;  // us -> ms
    mDecodeTokenTimes.push_back(tokenTime);
    mDecodeProfile.tokenCount+=count;
}

void LLMOpProfiler::onDecodePhaseStart() {
    if (!mEnabled) return;
    mInPrefill = false;
    // mDecodeProfile.reset();
    // mDecodeTokenTimes.clear();
    // printf("[LLM Profiler] Decode phase started\n");
}

void LLMOpProfiler::onDecodePhaseEnd() {
    if (!mEnabled) return;
    if (!mDecodeTokenTimes.empty()) {
        float totalTime = 0.0f;
        for (float t : mDecodeTokenTimes) {
            totalTime += t;
        }
        mDecodeProfile.tokenTotalTime = totalTime;
    }
    // printf("[LLM Profiler] Decode phase ended, %d tokens, time %.4f ms, avg time: %.4f ms/token\n", 
            //   mDecodeProfile.tokenCount, 
            //   mDecodeTokenTimes.empty() ? 0 : mDecodeProfile.tokenTotalTime,
            //   mDecodeTokenTimes.empty() ? 0 : mDecodeProfile.tokenTotalTime / mDecodeTokenTimes.size());
}

// ========== CPU Backend Callbacks ==========

bool LLMOpProfiler::beforeOp(const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
    if (!mEnabled) return true;
    if (info != nullptr && info->type() == "Plugin") {
        return true;
    }
    
    mOpTimer.reset();
    
    // Get actual backend from tensor (set during _allocMemory based on execution->backend())
    // This correctly handles fallback scenarios where OpenCL ops fall back to CPU
    MNNForwardType actualBackend = MNN_FORWARD_CPU;
    if (!tensors.empty()) {
        auto describe = TensorUtils::getDescribeOrigin(tensors[0]);
        if (describe) {
            Backend* backend = describe->getBackend();
            if (backend) {
                actualBackend = backend->type();
            }
        }
    }
    std::string backendName = forwardTypeToString(actualBackend);
    
    // Get Runtime for the actual backend
    auto executor = MNN::Express::ExecutorScope::Current();
    if (!executor) return true;
    
    auto runtimeInfo = executor->getRuntime();
    auto it = runtimeInfo.first.find(actualBackend);
    std::shared_ptr<Runtime> actualRuntime = (it != runtimeInfo.first.end()) ? it->second : nullptr;
    
    // MNN_FORWARD_CPU_EXTENSION uses same runtime as MNN_FORWARD_CPU
    if (!actualRuntime && actualBackend == MNN_FORWARD_CPU_EXTENSION) {
        it = runtimeInfo.first.find(MNN_FORWARD_CPU);
        actualRuntime = (it != runtimeInfo.first.end()) ? it->second : nullptr;
    }

    // Backend-specific handling
    if (actualBackend == MNN_FORWARD_CPU || actualBackend == MNN_FORWARD_CPU_EXTENSION) {
        // CPU backend: timer already started, timing recorded in afterOp
    } else if (actualRuntime) {
        // GPU/NPU backends: mark op start for kernel tracking
        actualRuntime->profileBegin(tensors, info);
    }
    
    // Record input shapes to opRecords (only if not already recorded)
    PhaseProfile& profile = mInPrefill ? mPrefillProfile : mDecodeProfile;
    std::string opKey = info->name() + "@" + backendName;
    
    // Create op record if not exists
    if (profile.opRecords.find(opKey) == profile.opRecords.end()) {
        profile.opRecords[opKey] = OpRecord();
        profile.opRecords[opKey].name = info->name();
        profile.opRecords[opKey].type = info->type();
        profile.opRecords[opKey].backend = backendName;
        profile.opRecords[opKey].isSpecial = isSpecialOp(info->name());
        
        // Store input shapes (only once per phase)
        for (auto* tensor : tensors) {
            if (tensor) {
                profile.opRecords[opKey].inputShapes.push_back(tensor->shape());
            }
        }
    }
    
    return true;
}

void LLMOpProfiler::afterOp(const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
    if (!mEnabled) return;
    if (info != nullptr && info->type() == "Plugin") {
        return;
    }
    
    // Get actual backend from tensor
    MNNForwardType actualBackend = MNN_FORWARD_CPU;
    if (!tensors.empty()) {
        auto describe = TensorUtils::getDescribeOrigin(tensors[0]);
        if (describe) {
            Backend* backend = describe->getBackend();
            if (backend) {
                actualBackend = backend->type();
            }
        }
    }
    std::string backendName = forwardTypeToString(actualBackend);

    // Get Runtime for the actual backend
    auto executor = MNN::Express::ExecutorScope::Current();
    if (!executor) return;
    
    auto runtimeInfo = executor->getRuntime();
    auto it = runtimeInfo.first.find(actualBackend);
    std::shared_ptr<Runtime> actualRuntime = (it != runtimeInfo.first.end()) ? it->second : nullptr;
    
    // MNN_FORWARD_CPU_EXTENSION uses same runtime as MNN_FORWARD_CPU
    if (!actualRuntime && actualBackend == MNN_FORWARD_CPU_EXTENSION) {
        it = runtimeInfo.first.find(MNN_FORWARD_CPU);
        actualRuntime = (it != runtimeInfo.first.end()) ? it->second : nullptr;
    }
    
    // Backend-specific handling
    if (actualBackend == MNN_FORWARD_CPU || actualBackend == MNN_FORWARD_CPU_EXTENSION) {
        // CPU backend: record synchronous timing
        uint64_t opTimeUs = mOpTimer.durationInUs();
        if (actualRuntime) {
            actualRuntime->recordOpProfileTime(info->name(), info->type(), opTimeUs);
        }
    } else if (actualRuntime) {
        // GPU/NPU backends: mark op end for kernel tracking
        actualRuntime->profileEnd(tensors, info);
    }
    
    // Get op record (created in beforeOp) and record output shapes (only once)
    PhaseProfile& profile = mInPrefill ? mPrefillProfile : mDecodeProfile;
    std::string opKey = info->name() + "@" + backendName;
    OpRecord& record = profile.opRecords[opKey];
    
    // Store output shapes (only once per phase, if not already recorded)
    if (record.outputShapes.empty()) {
        for (auto* tensor : tensors) {
            if (tensor) {
                record.outputShapes.push_back(tensor->shape());
            }
        }
    }
}

// ========== Backend Profile Data Collection ==========

void LLMOpProfiler::collectBackendProfile(const BackendProfileData& data) {
    if (!mEnabled || !data.valid) return;
    
    PhaseProfile& profile = mInPrefill ? mPrefillProfile : mDecodeProfile;
    
    // Get backend name
    std::string backendName = data.backendName;
    if (backendName.empty()) {
        backendName = forwardTypeToString(data.backendType);
    }
    
    for (const auto& opInfoPair : data.opInfos) {
        const std::string& opName = opInfoPair.first;
        const BackendOpInfo& opInfo = opInfoPair.second;
        float timeMs = opInfo.timeMs;
        std::string opType = opInfo.type.empty() ? "Unknown" : opInfo.type;
        std::string recordBackendName = opInfo.backendName.empty() ? backendName : opInfo.backendName;
        
        // Key: opName@backend
        std::string opKey = opName + "@" + recordBackendName;
        
        // Create or update op record
        if (profile.opRecords.find(opKey) == profile.opRecords.end()) {
            profile.opRecords[opKey] = OpRecord();
            profile.opRecords[opKey].name = opName;
            profile.opRecords[opKey].type = opType;
            profile.opRecords[opKey].backend = recordBackendName;
            profile.opRecords[opKey].isSpecial = isSpecialOp(opName);
        }
        
        OpRecord& record = profile.opRecords[opKey];
        record.totalTime += timeMs;
        record.callCount += opInfo.callCount;
        // Note: shapes are not available from backend profile data
        
        // Track per-backend total times
        profile.backendTotalTimes[recordBackendName] += timeMs;
        profile.totalTime += timeMs;
    }
}

// ========== Results ==========

void LLMOpProfiler::printStats() const {
    // Check if profiler is enabled
    if (!mEnabled) {
        printf("\n[LLM Profiler] Profiler is not enabled.\n");
        printf("Hint: Call llm->enableProfiler(true) before running inference.\n\n");
        return;
    }
    
    printf("\n");
    printf("================================================================\n");
    printf("              LLM Operator Profiling Report                    \n");
    printf("================================================================\n");
    
    // Helper function to print phase stats by OpType
    auto printPhaseStats = [this](const PhaseProfile& profile, const std::string& phaseName, bool isDecode) {
        printf("\n=== %s Phase Ops Statistics ===\n", phaseName.c_str());
        
        // Show two different time measurements
        if (profile.tokenTotalTime > 0) {
            if (isDecode && profile.tokenCount > 0) {
                // Decode phase: show per-token stats
                float avgPerToken = profile.tokenTotalTime / profile.tokenCount;
                float tokenStd = 0.0f;
                if (mDecodeTokenTimes.size() > 1) {
                    float sum = 0.0f;
                    for (float t : mDecodeTokenTimes) {
                        sum += (t - avgPerToken) * (t - avgPerToken);
                    }
                    tokenStd = sqrtf(sum / mDecodeTokenTimes.size());
                }
                float tokensPerSec = profile.tokenTotalTime > 0 ? 
                    profile.tokenCount / (profile.tokenTotalTime / 1000.0f) : 0.0f;
                printf("Token-level time: %.2f ms (%d tokens, %.2f ± %.2f ms/token, %.2f tokens/s)\n", 
                          profile.tokenTotalTime, profile.tokenCount, avgPerToken, tokenStd, tokensPerSec);
                
                // Op-level timing (sum of all op execution times)
                printf("Op-level time (cumulative): %.2f ms\n", profile.totalTime);
            } else if (!isDecode && profile.tokenCount > 0) {
                // Prefill phase: show prompt tokens and tokens/s (aligned with decode format)
                float tokensPerSec = profile.tokenTotalTime > 0 ? 
                    profile.tokenCount / (profile.tokenTotalTime / 1000.0f) : 0.0f;
                printf("Token-level time: %.2f ms (%d prompt tokens, %.2f tokens/s)\n", 
                          profile.tokenTotalTime, profile.tokenCount, tokensPerSec);
                
                // Op-level timing (sum of all op execution times)
                printf("Op-level time (cumulative): %.2f ms\n", profile.totalTime);
            } else {
                // No token count info
                printf("Token-level time: %.2f ms\n", profile.tokenTotalTime);
                printf("Op-level time (cumulative): %.2f ms\n", profile.totalTime);
            }
        } else {
            printf("Total time: %.2f ms\n", profile.totalTime);
        }
        
        // Print per-backend breakdown (cumulative time)
        if (!profile.backendTotalTimes.empty()) {
            printf("Backend breakdown: ");
            bool first = true;
            for (const auto& backendTimePair : profile.backendTotalTimes) {
                if (!first) printf(", ");
                first = false;
                float percent = profile.totalTime > 0 ? (backendTimePair.second / profile.totalTime * 100.0f) : 0.0f;
                printf("%s: %.2f ms (%.1f%%)", backendTimePair.first.c_str(), backendTimePair.second, percent);
            }
            printf("\n");
        }
        printf("\n");
        
        // Get aggregated type stats
        auto typeStats = profile.getOpTypeStats();
        
        // Separate special ops and normal ops by type
        std::vector<std::pair<std::string, OpRecord>> normalTypes;
        std::vector<std::pair<std::string, OpRecord>> specialTypes;
        for (const auto& pair : typeStats) {
            if (pair.second.isSpecial) {
                specialTypes.push_back(pair);
            } else {
                normalTypes.push_back(pair);
            }
        }
        
        // Sort by time
        auto sortByTime = [](std::vector<std::pair<std::string, OpRecord>>& vec) {
            std::sort(vec.begin(), vec.end(), 
                [](const std::pair<std::string, OpRecord>& a, const std::pair<std::string, OpRecord>& b) { 
                    return a.second.totalTime > b.second.totalTime; 
                });
        };
        sortByTime(normalTypes);
        sortByTime(specialTypes);
        
        // Print normal OpTypes
        printf("--- OpType Statistics (by type) ---\n");
        printf("%-24s %8s %12s %12s %10s %8s\n", 
                  "OpType", "Backend", "Time(ms)", "Avg(ms)", "Calls", "Percent");
        printf("--------------------------------------------------------------------------\n");
        
        for (const auto& pair : normalTypes) {
            const OpRecord& stats = pair.second;
            float percent = profile.totalTime > 0 ? (stats.totalTime / profile.totalTime * 100.0f) : 0.0f;
            const char* backend = stats.backend.empty() ? "-" : stats.backend.c_str();
            float avg = stats.avgTime();
            printf("%-24s %8s %12.2f %12.2f %10d %7.1f%%\n", 
                      stats.type.c_str(), backend, stats.totalTime, avg, stats.callCount, percent);
        }
        
        // Print special OpTypes separately
        if (!specialTypes.empty()) {
            printf("\n--- Special Ops ---\n");
            printf("%-48s %12s %10s\n", "OpName [OpType]", "Time(ms)", "Calls");
            printf("----------------------------------------------------------------\n");
            
            for (const auto& pair : specialTypes) {
                const OpRecord& stats = pair.second;
                float percent = profile.totalTime > 0 ? (stats.totalTime / profile.totalTime * 100.0f) : 0.0f;
                float avg = stats.avgTime();
                // Format: opname [optype] (backend) time avg calls percent
                printf("%-48s %12.2f %10d %7.1f%%\n", 
                          (stats.name + " [" + stats.type + "]").c_str(), 
                          stats.totalTime, stats.callCount, percent);
            }
        }
    };
    
    // Print Prefill phase (not decode)
    printPhaseStats(mPrefillProfile, "Prefill", false);
    
    // Print Decode phase
    printPhaseStats(mDecodeProfile, "Decode", true);
    
    // Summary
    printf("\n=== Summary ===\n");
    // Prefill: show both token-level and op-level time
    if (mPrefillProfile.tokenTotalTime > 0) {
        if (mPrefillProfile.tokenCount > 0) {
            float prefillTokensPerSec = mPrefillProfile.tokenTotalTime > 0 ? 
                mPrefillProfile.tokenCount / (mPrefillProfile.tokenTotalTime / 1000.0f) : 0.0f;
            printf("Prefill time: %.2f ms (%d prompt tokens, %.2f tokens/s)\n", 
                      mPrefillProfile.tokenTotalTime, mPrefillProfile.tokenCount, prefillTokensPerSec);
        } else {
            printf("Prefill time: %.2f ms \n", mPrefillProfile.tokenTotalTime);
        }
        printf("            : %.2f ms (op-level cumulative)\n", mPrefillProfile.totalTime);
    } else {
        printf("Prefill time: %.2f ms\n", mPrefillProfile.totalTime);
    }
    // Decode: show both token-level and op-level time
    if (mDecodeProfile.tokenCount > 0) {
        float avgPerToken = mDecodeProfile.tokenTotalTime / mDecodeProfile.tokenCount;
        float tokenStd = 0.0f;
        float decodeTokensPerSec = mDecodeProfile.tokenTotalTime > 0 ? 
            mDecodeProfile.tokenCount / (mDecodeProfile.tokenTotalTime / 1000.0f) : 0.0f;
        if (mDecodeTokenTimes.size() > 1) {
            float sum = 0.0f;
            for (float t : mDecodeTokenTimes) {
                sum += (t - avgPerToken) * (t - avgPerToken);
            }
            tokenStd = sqrtf(sum / mDecodeTokenTimes.size());
        }
        printf("Decode time: %.2f ms (%d tokens, %.2f ± %.2f ms/token, %.2f tokens/s)\n", 
                  mDecodeProfile.tokenTotalTime, mDecodeProfile.tokenCount, avgPerToken, tokenStd, decodeTokensPerSec);
        printf("           : %.2f ms (op-level cumulative)\n", mDecodeProfile.totalTime);
    }
    printf("================================================================\n");
}

void LLMOpProfiler::printOpInfo() const { 
    // Check if profiler is enabled
    if (!mEnabled) {
        printf("\n[LLM Profiler] Profiler is not enabled.\n");
        printf("Hint: Call llm->enableProfiler(true) before running inference.\n\n");
        return;
    }
    
    printf("\n");
    printf("================================================================\n");
    printf("                        LLM Operator Info                       \n");
    printf("================================================================\n");
    
    // Collect all unique ops from prefill and decode
    std::map<std::string, OpRecord> allOps;
    
    for (const auto& pair : mPrefillProfile.opRecords) {
        if (allOps.find(pair.first) == allOps.end()) {
            allOps[pair.first] = pair.second;
        }
    }
    for (const auto& pair : mDecodeProfile.opRecords) {
        if (allOps.find(pair.first) == allOps.end()) {
            allOps[pair.first] = pair.second;
        }
    }
    
    // Print op name, type, and full shapes
    printf("\n=== Operator Info (%zu ops) ===\n", allOps.size());
    for (const auto& pair : allOps) {
        const OpRecord& record = pair.second;
        
        printf("%s [%s]\n", record.name.c_str(), record.type.c_str());
        
        // Format input shapes
        if (!record.inputShapes.empty()) {
            printf("  In:  ");
            for (size_t i = 0; i < record.inputShapes.size(); ++i) {
                if (i > 0) printf(", ");
                printf("%s", shapeToString(record.inputShapes[i]).c_str());
            }
            printf("\n");
        }
        
        // Format output shapes
        if (!record.outputShapes.empty()) {
            printf("  Out: ");
            for (size_t i = 0; i < record.outputShapes.size(); ++i) {
                if (i > 0) printf(", ");
                printf("%s", shapeToString(record.outputShapes[i]).c_str());
            }
            printf("\n");
        }
    }
    
    printf("\n================================================================\n");
}

void LLMOpProfiler::reset() {
    mPrefillProfile.reset();
    mDecodeProfile.reset();
    mDecodeTokenTimes.clear();
    mInPrefill = true;
}

} // namespace Transformer
} // namespace MNN
