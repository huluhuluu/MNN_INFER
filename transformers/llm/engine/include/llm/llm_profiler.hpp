//
//  llm_profiler.hpp
//  MNN
//
//  Created for LLM Op Profiling
//

#ifndef LLM_PROFILER_HPP
#define LLM_PROFILER_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <cmath>
#include <MNN/Interpreter.hpp>
#include <MNN/AutoTime.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/MNNForwardType.h>

namespace MNN {
namespace Transformer {

// Forward declarations
class Llm;

/**
 * Op Record for storing profiling data
 * Key format: "opName@backend" to support multi-backend execution
 */
struct OpRecord {
    std::string name;           // Op name
    std::string type;           // Op type
    std::string backend;        // Backend name (CPU, OpenCL, QNN)
    float totalTime = 0.0f;     // Total time in ms
    int callCount = 0;          // Number of calls
    float flops = 0.0f;         // Computation amount
    bool isSpecial = false;     // Special op: listed separately in output, not merged with same-name ops
    
    // Shape info (recorded once per phase)
    std::vector<Express::INTS> inputShapes;   // [input_tensor] -> shape
    std::vector<Express::INTS> outputShapes;  // [output_tensor] -> shape
    
    float avgTime() const {
        return callCount > 0 ? totalTime / callCount : 0.0f;
    }
};

/**
 * Profile data for a single phase (Prefill or Decode)
 */
struct PhaseProfile {
    // Key: opName@backend - stores per-op records
    std::map<std::string, OpRecord> opRecords;
    
    // Per-backend total times (from op-level profiling)
    std::map<std::string, float> backendTotalTimes;      // backend -> total time
    
    float totalTime = 0.0f;              // Total phase time in ms (from op-level profiling)
    float tokenTotalTime = 0.0f;         // Total time from token-level measurement (decode only)
    int tokenCount = 0;                  // Number of tokens (for decode)
    
    void reset() {
        opRecords.clear();
        backendTotalTimes.clear();
        totalTime = 0.0f;
        tokenTotalTime = 0.0f;
        tokenCount = 0;
    }
    
    // Aggregate op records by type, returns type@backend -> OpRecord
    std::map<std::string, OpRecord> getOpTypeStats() const;
};

/**
 * Profile info for a single operator (from backend)
 */
struct BackendOpInfo {
    std::string name;       // Op name
    std::string type;       // Op type
    float timeMs = 0.0f;    // Time in milliseconds
    int callCount = 0;      // Number of calls
};

/**
 * Backend profile data interface
 */
struct BackendProfileData {
    MNNForwardType backendType;
    std::string backendName;                               // Backend name string
    std::map<std::string, BackendOpInfo> opInfos;          // op_name -> info (with type and time)
    float totalTime = 0.0f;
    bool valid = false;
};

/**
 * LLM Op Profiler Class
 * 
 * Provides operator-level profiling for LLM inference,
 * supporting CPU, OpenCL, and QNN backends.
 */
class LLMOpProfiler {
public:
    /**
     * Profiler configuration
     */
    struct Config {
        std::vector<std::string> specialOps;   // Special ops for separate timing
        bool separatePrefillDecode = true;      // Separate prefill/decode timing
        int decodeIterationsPerReport = 1;      // Report interval for decode
    };
    
    LLMOpProfiler();
    ~LLMOpProfiler();
    
    /**
     * Set configuration
     */
    void setConfig(const Config& config);
    
    /**
     * Set special ops for separate timing
     */
    void setSpecialOps(const std::vector<std::string>& specialOps);
    
    /**
     * Check if an op name is special
     */
    bool isSpecialOp(const std::string& opName) const;
    
    // ========== Phase Control ==========
    
    /**
     * Called when prefill phase starts
     */
    void onPrefillStart();
    
    /**
     * Called when prefill phase ends
     * @param promptTokenCount Number of input tokens in prompt
     */
    void onPrefillEnd(int promptTokenCount = 0);
    
    /**
     * Called when decode phase starts (per token)
     */
    void onDecodeTokenBegin();
    
    /**
     * Called when decode phase ends (per token)
     */
    void onDecodeTokenEnd(int count = 1);

    
    /**
     * Called when entire decode phase start
     */
    void onDecodePhaseStart();
    
    /**
     * Called when entire decode phase ends
     */
    void onDecodePhaseEnd();
    
    // ========== CPU Backend Callbacks ==========
    
    /**
     * Before op execution with OperatorInfo
     */
    bool beforeOp(const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info);
    
    /**
     * After op execution with OperatorInfo
     */
    void afterOp(const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo* info);
    
    // ========== Backend Profile Data Collection =///
    
    /**
     * Collect profile data from backend (OpenCL/QNN)
     * Called after each phase (prefill or decode batch)
     */
    void collectBackendProfile(const BackendProfileData& data);
    
    // ========== Results ==========
    
    /**
     * Print statistics to stdout
     */
    void printStats() const;

    /**
     * Print op info to stdout (name, type, shapes)
     */
    void printOpInfo() const;
    
    /**
     * Get prefill phase profile
     */
    const PhaseProfile& getPrefillProfile() const { return mPrefillProfile; }
    
    /**
     * Get decode phase profile
     */
    const PhaseProfile& getDecodeProfile() const { return mDecodeProfile; }
    
    /**
     * Reset all profiling data
     */
    void reset();
    
    /**
     * Enable/disable profiling
     */
    void setEnabled(bool enabled) { mEnabled = enabled; }
    bool isEnabled() const { return mEnabled; }
    
private:
    // Check if op name matches pattern (supports wildcards)
    bool matchPattern(const std::string& opName, const std::string& pattern) const;
    
    // Helper: MNNForwardType to string
    static std::string forwardTypeToString(MNNForwardType type);
    
    // Helper to get tensor shape as string
    static std::string shapeToString(const Express::INTS& shape);
    
private:
    Config mConfig;
    bool mEnabled = false;
    bool mInPrefill = true;
    
    // Profile data for each phase
    PhaseProfile mPrefillProfile;
    PhaseProfile mDecodeProfile;
    
    // Timing for CPU callback (op-level)
    MNN::Timer mOpTimer;
    
    // Timing for token-level(separate from op-level)
    MNN::Timer mTimer;
    
    // Special ops patterns
    std::vector<std::string> mSpecialOpPatterns;
    
    // Decode token timing history
    std::vector<float> mDecodeTokenTimes;
};

} // namespace Transformer
} // namespace MNN

#endif // LLM_PROFILER_HPP