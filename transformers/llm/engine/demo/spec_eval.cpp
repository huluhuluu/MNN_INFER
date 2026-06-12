//
//  spec_eval.cpp
//  MNN
//
//  Speculative Decoding Evaluation Tool
//  - Evaluate accept rate and accept length from test data
//  - Use config.json settings by default (backend, precision)
//  - Command line args can override config.json
//

#include "llm/llm.hpp"
#include "speculative_decoding/generate.hpp"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <iterator>

using namespace MNN::Transformer;

// ==================== Configuration ====================

struct EvalConfig {
    std::string configPath;      // Model config.json path (required)
    std::string dataFile;        // Test data file path (required)
    std::string outputFile;      // Output results file (optional)
    std::string backend;         // Override backend (optional)
    std::string precision;       // Override precision (optional)
    int maxNewTokens = 512;       // Max tokens per sample
    bool verbose = false;
};

// ==================== Evaluator ====================

class Evaluator {
public:
    Evaluator(const EvalConfig& config) : mConfig(config) {}
    
    bool run() {
        std::cout << "\n================================================\n";
        std::cout << "     Speculative Decoding Evaluation\n";
        std::cout << "================================================\n\n";
        
        std::cout << "Config: " << mConfig.configPath << "\n";
        std::cout << "Data: " << mConfig.dataFile << "\n";
        
        // Load test data
        if (!loadTestData()) return false;
        
        // Load model (uses config.json settings)
        if (!loadModel()) return false;
        
        // Check speculative decoding
        if (!mLlm->isInSpec()) {
            std::cerr << "Error: Spec decoding not enabled in config\n";
            return false;
        }
        
        // Run evaluation (accumulates in LLM's SpecContext)
        int totalSamples = runEvaluation();
        
        // Print results (read from LLM's SpecContext)
        printResults(totalSamples);
        
        // Save if specified
        if (!mConfig.outputFile.empty()) {
            saveResults(totalSamples);
        }
        
        return true;
    }

private:
    EvalConfig mConfig;
    std::unique_ptr<Llm> mLlm;
    std::vector<std::string> mTestPrompts;
    int64_t mTotalPromptLen = 0;
    
    bool loadTestData() {
        std::ifstream file(mConfig.dataFile);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open data file: " << mConfig.dataFile << "\n";
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            // Trim
            size_t start = line.find_first_not_of(" \t\r\n");
            size_t end = line.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos) {
                line = line.substr(start, end - start + 1);
            }
            if (!line.empty()) {
                mTestPrompts.push_back(line);
            }
        }
        
        std::cout << "Loaded " << mTestPrompts.size() << " test samples\n\n";
        return !mTestPrompts.empty();
    }
    
    bool loadModel() {
        std::cout << "Loading model...\n";
        
        mLlm.reset(Llm::createLLM(mConfig.configPath));
        if (!mLlm) {
            std::cerr << "Error: Failed to create LLM\n";
            return false;
        }
        
        // Override config settings if specified via command line
        if (!mConfig.backend.empty() || !mConfig.precision.empty()) {
            std::string configJson = "{";
            if (!mConfig.backend.empty()) {
                configJson += "\"backend_type\": \"" + mConfig.backend + "\"";
            }
            if (!mConfig.precision.empty()) {
                if (!mConfig.backend.empty()) configJson += ", ";
                configJson += "\"precision\": \"" + mConfig.precision + "\"";
            }
            configJson += "}";
            mLlm->set_config(configJson.c_str());
        }
        
        MNN::Timer timer;
        timer.reset();
        if (!mLlm->load()) {
            std::cerr << "Error: Failed to load model\n";
            return false;
        }
        
        std::cout << "Model loaded in " << timer.durationInUs() / 1000.0f << " ms\n";
        std::cout << "Draft length: " << mLlm->getDraftLength() << "\n\n";
        
        return true;
    }
    
    int runEvaluation() {
        std::cout << "=== Running Evaluation ===\n\n";
        
        // Reset once at the beginning - LLM's SpecContext will accumulate all samples
        mLlm->resetSpecContext();
        mTotalPromptLen = 0;
        
        int sampleCount = 0;
        for (const auto& prompt : mTestPrompts) {
            sampleCount++;
            
            // Reset history but keep SpecContext accumulating
            mLlm->reset();
            
            MNN::Timer timer;
            timer.reset();
            
            mLlm->response(prompt, nullptr, nullptr, mConfig.maxNewTokens);
            // show output
            // mLlm->response(prompt,  &std::cout, nullptr, mConfig.maxNewTokens);
            
            float timeMs = timer.durationInUs() / 1000.0f;
            
            // Get current stats (accumulated so far)
            auto context = mLlm->getContext();
            const SpecContext* specCtx = mLlm->getSpecContext();
            mTotalPromptLen += context->prompt_len;
            
            {
                // Print progress and current stats
                int steps = specCtx ? specCtx->steps : 0;
                float avgAccept = specCtx ? specCtx->avgAcceptLen() : 0;
                double percent = 100.0 * sampleCount / mTestPrompts.size();
                std::ostringstream oss;
                oss << "\r[" << std::setw(4) << sampleCount << "/" << mTestPrompts.size() << "] "
                    << "(" << std::setw(3) << static_cast<int>(percent) << "%) "
                    << "| tokens=" << std::setw(4) << context->gen_seq_len
                    << ", steps=" << std::setw(3) << steps
                    << ", avg_accept=" << std::fixed << std::setprecision(2) << avgAccept
                    << ", time=" << std::setprecision(1) << timeMs << "ms";

                std::cout << oss.str() << "\033[K" << std::flush;
            }
        }
        
        return sampleCount;
    }
    
    void printResults(int totalSamples) {
        const SpecContext* ctx = mLlm->getSpecContext();
        
        std::cout << "\n================================================\n";
        std::cout << "              Evaluation Results\n";
        std::cout << "================================================\n\n";
        
        std::cout << "--- Statistics ---\n";
        std::cout << "Total samples:          " << totalSamples << "\n";
        std::cout << "Total prompt len:       " << mTotalPromptLen << "\n";
        std::cout << "Total decoding steps:   " << ctx->steps << "\n";
        std::cout << "Total draft tokens:     " << ctx->draft << "\n";
        std::cout << "Total accepted tokens:  " << ctx->accepted << "\n";
        std::cout << "Total draft time:       " << std::fixed << std::setprecision(2) 
                  << ctx->draft_time_us / 1000.0f << " ms\n";
        std::cout << "Total target time:      " << ctx->target_time_us / 1000.0f << " ms\n\n";
        
        std::cout << "--- Key Metrics ---\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Average Accept Length:  " << ctx->avgAcceptLen() << " tokens/step\n";
        std::cout << "Accept Rate:            " << ctx->acceptRate() * 100.0f << "%\n";
        std::cout << "Compression Ratio:      " << ctx->compressionRatio() << " tokens/step\n";
        std::cout << "Avg Draft Time:         " << ctx->avgDraftTimeMs() << " ms/step\n";
        std::cout << "Avg Target Time:        " << ctx->avgTargetTimeMs() << " ms/step\n";
        std::cout << "Theoretical Speedup:    " << ctx->theoreticalSpeedup() << "x\n";

        std::cout << "\n--- Accept Length Frequency ---\n";
        for (const auto& item : ctx->accept_len_freq) {
            std::cout << "Accept Length " << std::setw(3) << item.first << ": "
                      << item.second << "\n";
        }
        
        std::cout << "\n================================================\n";
    }
    
    void saveResults(int totalSamples) {
        const SpecContext* ctx = mLlm->getSpecContext();
        
        std::ofstream file(mConfig.outputFile);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open output file: " << mConfig.outputFile << "\n";
            return;
        }
        
        file << "{\n";
        file << "  \"samples\": " << totalSamples << ",\n";
        file << "  \"total_prompt_len\": " << mTotalPromptLen << ",\n";
        file << "  \"total_steps\": " << ctx->steps << ",\n";
        file << "  \"total_draft_tokens\": " << ctx->draft << ",\n";
        file << "  \"total_accepted_tokens\": " << ctx->accepted << ",\n";
        file << "  \"draft_time_ms\": " << ctx->draft_time_us / 1000.0 << ",\n";
        file << "  \"target_time_ms\": " << ctx->target_time_us / 1000.0 << ",\n";
        file << std::fixed << std::setprecision(4);
        file << "  \"avg_accept_length\": " << ctx->avgAcceptLen() << ",\n";
        file << "  \"accept_rate\": " << ctx->acceptRate() * 100.0f << ",\n";
        file << "  \"compression_ratio\": " << ctx->compressionRatio() << ",\n";
        file << "  \"theoretical_speedup\": " << ctx->theoreticalSpeedup() << ",\n";
        file << "  \"accept_length_frequency\": {\n";
        for (auto iter = ctx->accept_len_freq.begin(); iter != ctx->accept_len_freq.end(); ++iter) {
            file << "    \"" << iter->first << "\": " << iter->second;
            if (std::next(iter) != ctx->accept_len_freq.end()) {
                file << ",";
            }
            file << "\n";
        }
        file << "  }\n";
        file << "}\n";
        
        std::cout << "Results saved to: " << mConfig.outputFile << "\n";
    }
};

// ==================== CLI ====================

void printUsage(const char* progName) {
    std::cout << "Spec Decoding Evaluation\n\n";
    std::cout << "Usage: " << progName << " config.json data.txt [options]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  config.json    Model config (uses backend/precision from this file)\n";
    std::cout << "  data.txt       Test data (one prompt per line)\n\n";
    std::cout << "Options:\n";
    std::cout << "  --backend=TYPE    Override backend (cpu/opencl)\n";
    std::cout << "  --precision=MODE  Override precision (normal/high/low)\n";
    std::cout << "  --max-tokens=N    Max new tokens (default: 64)\n";
    std::cout << "  --output=FILE     Save results to JSON\n";
    std::cout << "  --verbose         Print per-sample details\n";
    std::cout << "  --help            Show this help\n";
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 0;
    }
    
    EvalConfig config;
    config.configPath = argv[1];
    config.dataFile = argv[2];
    
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg.find("--backend=") == 0) {
            config.backend = arg.substr(10);
        } else if (arg.find("--precision=") == 0) {
            config.precision = arg.substr(12);
        } else if (arg.find("--max-tokens=") == 0) {
            config.maxNewTokens = std::stoi(arg.substr(13));
        } else if (arg.find("--output=") == 0) {
            config.outputFile = arg.substr(9);
        } else if (arg == "--verbose") {
            config.verbose = true;
        }
    }
    
    // Create executor (determine type from config or override)
    MNNForwardType forwardType = MNN_FORWARD_CPU;
    if (config.backend == "opencl") {
        forwardType = MNN_FORWARD_OPENCL;
    }
    
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(forwardType, backendConfig, 4);
    MNN::Express::ExecutorScope scope(executor);
    
    Evaluator evaluator(config);
    return evaluator.run() ? 0 : 1;
}
