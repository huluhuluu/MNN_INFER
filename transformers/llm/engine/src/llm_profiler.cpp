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
        std::string regexPattern;
        for (char c : pattern) {
            if (c == '*') {
                regexPattern += ".*";
            } else {
                regexPattern += "\\" + std::string(1, c);
            }
        }
        // Use simple wildcard matching instead of regex to avoid exceptions
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

// ========== Phase Control ==========

void LLMOpProfiler::onPrefillStart() {
    if (!mEnabled) return;
    mInPrefill = true;
    mPrefillProfile.reset();
    mTimer.reset(); 
    printf("[LLM Profiler] Prefill phase started\n");
}

void LLMOpProfiler::onPrefillEnd(int promptTokenCount) {
    if (!mEnabled) return;
    mInPrefill = false;
    // Record token time for prefill
    mPrefillProfile.tokenTotalTime = mTimer.durationInUs() / 1000.0f;  // us -> ms
    mPrefillProfile.tokenCount = promptTokenCount;
    
    printf("[LLM Profiler] Prefill phase ended, %d tokens, time %.4f ms, avg time: %.4f ms/token\n", 
              promptTokenCount, 
              mPrefillProfile.tokenTotalTime ,
              promptTokenCount==0 ? 0 : mPrefillProfile.tokenTotalTime  / promptTokenCount);
}

void LLMOpProfiler::onDecodeTokenStart(int tokenId) {
    if (!mEnabled) return;
    mCurrentDecodeToken = tokenId;
    mTimer.reset();  // Use separate timer for token-level timing
}

void LLMOpProfiler::onDecodeTokenEnd(int tokenId) {
    if (!mEnabled) return;
    float tokenTime = mTimer.durationInUs() / 1000.0f;  // us -> ms
    mDecodeTokenTimes.push_back(tokenTime);
    mDecodeProfile.tokenCount++;
}

void LLMOpProfiler::onDecodePhaseStart(){
    if (!mEnabled) return;
    mInPrefill = false;
    mDecodeProfile.reset();
    mDecodeTokenTimes.clear();
    printf("[LLM Profiler] Decode phase started\n");
}
void LLMOpProfiler::onDecodePhaseEnd() {
    if (!mEnabled) return;
    if (!mDecodeTokenTimes.empty()) {
        float avgTime = 0.0f;
        for (float t : mDecodeTokenTimes) {
            avgTime += t;
        }
        avgTime /= mDecodeTokenTimes.size();
        mDecodeProfile.tokenTotalTime = avgTime * mDecodeTokenTimes.size();
    }
    printf("[LLM Profiler] Decode phase ended, %d tokens, time %.4f ms, avg time: %.4f ms/token\n", 
              mDecodeProfile.tokenCount, 
              mDecodeTokenTimes.empty() ? 0 : mDecodeProfile.tokenTotalTime,
              mDecodeTokenTimes.empty() ? 0 : mDecodeProfile.tokenTotalTime / mDecodeTokenTimes.size());
}

// ========== CPU Backend Callbacks ==========

bool LLMOpProfiler::beforeOp(const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
    if (!mEnabled) return true;
    
    mCurrentOpName = info->name();
    mCurrentOpType = info->type();
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
    // Note: onMarkOpStart() is a virtual function in Runtime base class
    // - CPU Runtime: default implementation does nothing
    // - OpenCL Runtime: overridden to mark kernel entries
    if (actualBackend == MNN_FORWARD_CPU || actualBackend == MNN_FORWARD_CPU_EXTENSION) {
        // CPU backend: timer already started, timing recorded in afterOp
    } else if (actualRuntime) {
        // GPU/NPU backends: mark op start for kernel tracking
        actualRuntime->profileStart(tensors, info);
    }
    
    return true;
}

void LLMOpProfiler::afterOp(const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info) {
    if (!mEnabled) return;
    
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
    // Note: onMarkOpEnd() is a virtual function in Runtime base class
    // - CPU Runtime: default implementation does nothing
    // - OpenCL Runtime: overridden to set op name/type for kernel entries
    if (actualBackend == MNN_FORWARD_CPU || actualBackend == MNN_FORWARD_CPU_EXTENSION) {
        // CPU backend: record synchronous timing
        // recordOpProfileTime expects time in microseconds (us)
        uint64_t opTimeUs = mOpTimer.durationInUs();
        if (actualRuntime) {
            actualRuntime->recordOpProfileTime(info->name(), info->type(), opTimeUs);
        }
    } else if (actualRuntime) {
        // GPU/NPU backends: mark op end for kernel tracking
        actualRuntime->profileEnd(tensors, info);
    }
}

// ========== Backend Profile Data Collection ==========

void LLMOpProfiler::collectBackendProfile(const BackendProfileData& data) {
    if (!mEnabled || !data.valid) return;
    
    PhaseProfile& profile = mInPrefill ? mPrefillProfile : mDecodeProfile;
    
    // Get backend name (the actual backend this data comes from)
    std::string backendName = data.backendName;
    if (backendName.empty()) {
        switch (data.backendType) {
            case MNN_FORWARD_CPU: backendName = "CPU"; break;
            case MNN_FORWARD_CPU_EXTENSION: backendName = "CPU"; break;
            case MNN_FORWARD_OPENCL: backendName = "OpenCL"; break;
            case MNN_FORWARD_NN: backendName = "QNN"; break;
            default: backendName = "Other"; break;
        }
    }
    
    for (const auto& opInfoPair : data.opInfos) {
        const std::string& opName = opInfoPair.first;
        const BackendOpInfo& opInfo = opInfoPair.second;
        float timeMs = opInfo.timeMs;
        std::string opType = opInfo.type.empty() ? "Unknown" : opInfo.type;
        bool special = isSpecialOp(opName);
        
        // For special ops, use opName as type (treat as unique type)
        if (special) {
            opType = opName;  // Special ops become their own type
        }
        
        // Use "opType@backend" as key for type-based grouping
        std::string typeKey = opType + "@" + backendName;
        
        // Store in opTypeRecords (keyed by type@backend)
        if (profile.opTypeRecords.find(typeKey) == profile.opTypeRecords.end()) {
            profile.opTypeRecords[typeKey] = OpRecord();
        }
        auto& typeRecord = profile.opTypeRecords[typeKey];
        typeRecord.name = opType;
        typeRecord.type = opType;
        typeRecord.backend = backendName;
        typeRecord.totalTime += timeMs;
        typeRecord.callCount += opInfo.callCount;
        typeRecord.isSpecial = special;
        
        // Track per-backend total times
        profile.backendTotalTimes[backendName] += timeMs;
        
        profile.totalTime += timeMs;
    }
}

// ========== Results ==========

void LLMOpProfiler::printStats() const {
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
        
        // Separate special ops and normal ops by type
        std::vector<std::pair<std::string, OpRecord>> normalTypes;
        std::vector<std::pair<std::string, OpRecord>> specialTypes;
        for (const auto& keyRecordPair : profile.opTypeRecords) {
            if (keyRecordPair.second.isSpecial) {
                specialTypes.push_back(keyRecordPair);
            } else {
                normalTypes.push_back(keyRecordPair);
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
        
        for (const auto& keyRecordPair : normalTypes) {
            const OpRecord& record = keyRecordPair.second;
            float percent = profile.totalTime > 0 ? (record.totalTime / profile.totalTime * 100.0f) : 0.0f;
            const char* backend = record.backend.empty() ? "-" : record.backend.c_str();
            float avg = record.avgTime();
            printf("%-24s %8s %12.2f %12.2f %10d %7.1f%%\n", 
                      record.name.c_str(), backend, record.totalTime, avg, record.callCount, percent);
        }
        
        // Print special OpTypes separately
        if (!specialTypes.empty()) {
            printf("\n--- Special Ops (as unique OpType) ---\n");
            printf("%-24s %8s %12s %12s %10s %8s\n", 
                      "OpType", "Backend", "Time(ms)", "Avg(ms)", "Calls", "Percent");
            printf("--------------------------------------------------------------------------\n");
            
            for (const auto& keyRecordPair : specialTypes) {
                const OpRecord& record = keyRecordPair.second;
                float percent = profile.totalTime > 0 ? (record.totalTime / profile.totalTime * 100.0f) : 0.0f;
                const char* backend = record.backend.empty() ? "-" : record.backend.c_str();
                float avg = record.avgTime();
                printf("%-24s %8s %12.2f %12.2f %10d %7.1f%%\n", 
                          record.name.c_str(), backend, record.totalTime, avg, record.callCount, percent);
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

bool LLMOpProfiler::exportJSON(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        MNN_ERROR("Failed to open file for JSON export: %s\n", filepath.c_str());
        return false;
    }
    
    file << "{\n";
    
    // Backend breakdown - derived from stored data
    file << "  \"backends\": {\n";
    bool firstBackend = true;
    // Combine prefill and decode backend times
    std::map<std::string, float> allBackendTimes;
    for (const auto& pair : mPrefillProfile.backendTotalTimes) {
        allBackendTimes[pair.first] += pair.second;
    }
    for (const auto& pair : mDecodeProfile.backendTotalTimes) {
        allBackendTimes[pair.first] += pair.second;
    }
    for (const auto& pair : allBackendTimes) {
        if (!firstBackend) file << ",\n";
        firstBackend = false;
        file << "    \"" << pair.first << "\": " << pair.second;
    }
    file << "\n  },\n";
    
    // Prefill phase
    file << "  \"prefill\": {\n";
    file << "    \"total_time_ms\": " << mPrefillProfile.totalTime << ",\n";
    file << "    \"backend_times\": {\n";
    firstBackend = true;
    for (const auto& pair : mPrefillProfile.backendTotalTimes) {
        if (!firstBackend) file << ",\n";
        firstBackend = false;
        file << "      \"" << pair.first << "\": " << pair.second;
    }
    file << "\n    },\n";
    file << "    \"ops\": [\n";
    bool first = true;
    for (const auto& nameRecordPair : mPrefillProfile.opRecords) {
        if (!first) file << ",\n";
        first = false;
        const OpRecord& record = nameRecordPair.second;
        file << "      {\"name\": \"" << record.name << "\", "
             << "\"backend\": \"" << record.backend << "\", "
             << "\"time_ms\": " << record.totalTime << ", "
             << "\"avg_ms\": " << record.avgTime() << ", "
             << "\"calls\": " << record.callCount << ", "
             << "\"special\": " << (record.isSpecial ? "true" : "false") << "}";
    }
    file << "\n    ]\n";
    file << "  },\n";
    
    // Decode phase
    file << "  \"decode\": {\n";
    file << "    \"total_time_ms\": " << mDecodeProfile.totalTime << ",\n";
    file << "    \"token_count\": " << mDecodeProfile.tokenCount << ",\n";
    file << "    \"backend_times\": {\n";
    firstBackend = true;
    for (const auto& pair : mDecodeProfile.backendTotalTimes) {
        if (!firstBackend) file << ",\n";
        firstBackend = false;
        file << "      \"" << pair.first << "\": " << pair.second;
    }
    file << "\n    },\n";
    file << "    \"ops\": [\n";
    first = true;
    for (const auto& nameRecordPair : mDecodeProfile.opRecords) {
        if (!first) file << ",\n";
        first = false;
        const OpRecord& record = nameRecordPair.second;
        file << "      {\"name\": \"" << record.name << "\", "
             << "\"backend\": \"" << record.backend << "\", "
             << "\"time_ms\": " << record.totalTime << ", "
             << "\"avg_ms\": " << record.avgTime() << ", "
             << "\"calls\": " << record.callCount << ", "
             << "\"special\": " << (record.isSpecial ? "true" : "false") << "}";
    }
    file << "\n    ]\n";
    file << "  }\n";
    
    file << "}\n";
    file.close();
    
    printf("[LLM Profiler] Results exported to: %s\n", filepath.c_str());
    return true;
}

void LLMOpProfiler::reset() {
    mPrefillProfile.reset();
    mDecodeProfile.reset();
    mDecodeTokenTimes.clear();
    mInPrefill = true;
    mCurrentDecodeToken = -1;
}

} // namespace Transformer
} // namespace MNN
