//
//  llm_op_profiler.cpp
//  MNN-LLM
//
//  Created for LLM Op Profiling
//

#include "llm/llm.hpp"
#include "llm/llm_profiler.hpp"
#include "core/MNNFileUtils.h"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

using namespace MNN::Transformer;

// Print usage
static void printUsage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\n");
    printf("LLM Operator-level Profiler Tool\n");
    printf("Profiles individual operator execution times for LLM inference.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help              Show this help message\n");
    printf("  -m, --model <path>      Model config path (required)\n");
    printf("  -b, --backend <type>    Backend type: cpu, opencl, qnn (default: cpu)\n");
    printf("  -t, --threads <n>       Number of threads (default: 4)\n");
    printf("  -p, --prompt <text>     Input prompt text\n");
    printf("  -n, --max-tokens <n>    Max tokens to generate (default: 16)\n");
    printf("  -o, --output <file>     Output JSON file for profile data\n");
    printf("  -v, --verbose           Print detailed operator timing info\n");
    printf("  --special-ops <ops>     Comma-separated list of special ops to track\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s -m ./Qwen2.5-1.5B-Instruct/config.json -b opencl -p \"Hello\" -n 32\n", prog);
    printf("  %s -m ./model/config.json -b qnn -o profile.json -v\n", prog);
}

// Parse backend type
static int parseBackendType(const std::string& type) {
    if (type == "metal") return 1;
    if (type == "opencl") return 3;
    if (type == "qnn") return 4;  // QNN backend
    return 0; // cpu
}

// Build LLM with specified configuration
static Llm* buildLLM(const std::string& config_path, int backend, int threads, bool use_mmap = false) {
    auto llmPtr = Llm::createLLM(config_path);
    llmPtr->set_config(R"({
        "async": false
    })");
    
    std::map<int, std::string> backend_type = {
        {0, "cpu"}, {1, "metal"}, {3, "opencl"}, {4, "qnn"}
    };
    
    std::map<bool, std::string> mmap = {{true, "true"}, {false, "false"}};
    
    if (llmPtr->set_config("{\"backend_type\":\"" + backend_type[backend] + "\"}")) {
        std::cout << "Backend: " << backend_type[backend] << std::endl;
    }
    
    if (llmPtr->set_config("{\"thread_num\":" + std::to_string(threads) + "}")) {
        std::cout << "Threads: " << threads << std::endl;
    }
    
    llmPtr->set_config("{\"use_mmap\":" + mmap[use_mmap] + "}");
    llmPtr->set_config("{\"tmp_path\":\"tmp\"}");
    
    return llmPtr;
}

int main(int argc, char** argv) {
    // Default parameters
    std::string modelPath;
    std::string backendType = "cpu";
    std::string promptText = "Hello, how are you?";
    std::string outputFile;
    std::string specialOps;
    int threads = 4;
    int maxTokens = 16;
    bool verbose = false;
    bool showHelp = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            showHelp = true;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            modelPath = argv[++i];
        } else if ((arg == "-b" || arg == "--backend") && i + 1 < argc) {
            backendType = argv[++i];
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            promptText = argv[++i];
        } else if ((arg == "-n" || arg == "--max-tokens") && i + 1 < argc) {
            maxTokens = std::stoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputFile = argv[++i];
        } else if ((arg == "-v" || arg == "--verbose")) {
            verbose = true;
        } else if (arg == "--special-ops" && i + 1 < argc) {
            specialOps = argv[++i];
        }
    }
    
    if (showHelp) {
        printUsage(argv[0]);
        return 0;
    }
    
    if (modelPath.empty()) {
        std::cerr << "Error: Model path is required. Use -m <path>" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    int backend = parseBackendType(backendType);
    
    std::cout << "========================================" << std::endl;
    std::cout << "LLM Operator Profiler" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Backend: " << backendType << std::endl;
    std::cout << "Threads: " << threads << std::endl;
    std::cout << "Max Tokens: " << maxTokens << std::endl;
    std::cout << "Prompt: " << promptText << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Create executor
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope scope(executor);
    
    // Build and load LLM
    auto llmPtr = buildLLM(modelPath, backend, threads);
    std::unique_ptr<Llm> llm(llmPtr);
    
    // Enable profiler
    llm->enableProfiler(true);
    
    // Set special ops if specified
    if (!specialOps.empty()) {
        std::vector<std::string> ops;
        std::istringstream ss(specialOps);
        std::string op;
        while (std::getline(ss, op, ',')) {
            ops.push_back(op);
        }
        llm->setProfilerSpecialOps(ops);
        std::cout << "Special ops to track: " << specialOps << std::endl;
    }
    
    std::cout << "\nLoading model..." << std::endl;
    Timer loadTimer;
    llm->load();
    std::cout << "Model loaded in " << loadTimer.durationInUs() / 1000.0 << " ms" << std::endl;
    
    // Set max tokens
    llm->set_config("{\"max_new_tokens\":" + std::to_string(maxTokens) + "}");
    
    // Run inference
    std::cout << "\nRunning inference..." << std::endl;
    
    Timer inferTimer;
    llm->response(promptText, nullptr, nullptr, maxTokens);
    double inferTimeMs = inferTimer.durationInUs() / 1000.0;
    
    std::cout << "Inference completed in " << inferTimeMs << " ms" << std::endl;
    
    // Print profiler statistics
    std::cout << "\n========================================" << std::endl;
    std::cout << "Profiler Statistics" << std::endl;
    std::cout << "========================================" << std::endl;
    
    llm->printProfilerStats();
    
    if (verbose) {
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Detailed Operator Timings" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        auto profiler = llm->getProfiler();
        if (profiler) {
            // Print prefill phase details
            const auto& prefillProfile = profiler->getPrefillProfile();
            std::cout << "\n[Prefill Phase]" << std::endl;
            std::cout << "  Total time: " << prefillProfile.totalTime << " ms" << std::endl;
            std::cout << "  Operators:" << std::endl;
            
            std::vector<std::pair<std::string, OpRecord>> sortedOps(
                prefillProfile.opRecords.begin(), prefillProfile.opRecords.end());
            std::sort(sortedOps.begin(), sortedOps.end(),
                [](const std::pair<std::string, OpRecord>& a, const std::pair<std::string, OpRecord>& b) {
                    return a.second.totalTime > b.second.totalTime;
                });
            
            for (const auto& nameRecordPair : sortedOps) {
                const OpRecord& record = nameRecordPair.second;
                std::cout << "    " << record.name << " [" << record.backend << "]: " 
                          << record.totalTime << " ms "
                          << "(calls: " << record.callCount << ")" << std::endl;
            }
            
            // Print decode phase details
            const auto& decodeProfile = profiler->getDecodeProfile();
            std::cout << "\n[Decode Phase]" << std::endl;
            std::cout << "  Total tokens: " << decodeProfile.tokenCount << std::endl;
            std::cout << "  Total time: " << decodeProfile.totalTime << " ms" << std::endl;
            std::cout << "  Operators:" << std::endl;
            
            sortedOps.clear();
            sortedOps.assign(decodeProfile.opRecords.begin(), decodeProfile.opRecords.end());
            std::sort(sortedOps.begin(), sortedOps.end(),
                [](const std::pair<std::string, OpRecord>& a, const std::pair<std::string, OpRecord>& b) {
                    return a.second.totalTime > b.second.totalTime;
                });
            
            for (const auto& nameRecordPair : sortedOps) {
                const OpRecord& record = nameRecordPair.second;
                std::cout << "    " << record.name << " [" << record.backend << "]: " 
                          << record.totalTime << " ms "
                          << "(calls: " << record.callCount << ")" << std::endl;
            }
        }
    }
    
    // Export to JSON if specified
    if (!outputFile.empty()) {
        if (llm->exportProfilerJSON(outputFile)) {
            std::cout << "\nProfile data exported to: " << outputFile << std::endl;
        } else {
            std::cerr << "Error: Could not export profile data to: " << outputFile << std::endl;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Profiling Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
