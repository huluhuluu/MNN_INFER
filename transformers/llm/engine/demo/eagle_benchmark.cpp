//
//  eagle_benchmark.cpp
//  MNN
//
//  Benchmark for Eagle speculative decoding.
//

#include "llm/llm.hpp"
#include "speculative_decoding/generate.hpp"
#include <MNN/expr/ExecutorScope.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace MNN::Transformer;

namespace {

struct BenchmarkConfig {
    int warmupIterations = 2;
    int repeatIterations = 3;
    int maxNewTokens = 5;
    int threadNum = 4;
    std::vector<int> promptLengths = {8, 16, 32, 64, 128, 256, 512, 1024};
    std::string backendType;
    std::string precision;
};

struct Stats {
    double mean = 0.0;
    double stddev = 0.0;

    static Stats calculate(const std::vector<double>& samples) {
        Stats stats;
        if (samples.empty()) {
            return stats;
        }

        double sum = 0.0;
        for (double sample : samples) {
            sum += sample;
        }
        stats.mean = sum / static_cast<double>(samples.size());

        if (samples.size() < 2) {
            return stats;
        }

        double variance = 0.0;
        for (double sample : samples) {
            const double diff = sample - stats.mean;
            variance += diff * diff;
        }
        stats.stddev = std::sqrt(variance / static_cast<double>(samples.size() - 1));
        return stats;
    }
};

struct RunMetrics {
    double targetPrefillMs = 0.0;
    double decodeTotalMs = 0.0;
    double targetVerifyMs = 0.0;
    double draftPrefillMs = 0.0;
    double draftDecodeMs = 0.0;
    int outputTokens = 0;
    int acceptedTokens = 0;
    int draftTokens = 0;
    int steps = 0;
    bool valid = false;
};

struct PromptResult {
    int promptLength = 0;
    Stats targetPrefill;
    Stats decodeTotal;
    Stats targetVerify;
    Stats draftPrefill;
    Stats draftDecode;
    Stats outputTokens;
    Stats acceptedTokens;
    Stats draftTokens;
    Stats steps;
};

std::vector<int> parseLengths(const std::string& arg) {
    std::vector<int> lengths;
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            lengths.push_back(std::stoi(item));
        }
    }
    return lengths;
}

std::string parseStringValueFromConfig(const std::string& configPath, const std::string& key) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string content = buffer.str();

    const std::string quotedKey = "\"" + key + "\"";
    size_t keyPos = content.find(quotedKey);
    if (keyPos == std::string::npos) {
        return "";
    }
    size_t colonPos = content.find(':', keyPos + quotedKey.size());
    if (colonPos == std::string::npos) {
        return "";
    }
    size_t firstQuote = content.find('"', colonPos + 1);
    if (firstQuote == std::string::npos) {
        return "";
    }
    size_t secondQuote = content.find('"', firstQuote + 1);
    if (secondQuote == std::string::npos) {
        return "";
    }
    return content.substr(firstQuote + 1, secondQuote - firstQuote - 1);
}

MNNForwardType executorTypeFromBackend(const std::string& backend) {
    if (backend == "opencl") {
        return MNN_FORWARD_OPENCL;
    }
    if (backend == "vulkan") {
        return MNN_FORWARD_VULKAN;
    }
    if (backend == "metal") {
        return MNN_FORWARD_METAL;
    }
    if (backend == "cuda") {
        return MNN_FORWARD_CUDA;
    }
    if (backend == "opengl") {
        return MNN_FORWARD_OPENGL;
    }
    return MNN_FORWARD_CPU;
}

void printUsage(const char* progName) {
    std::cout << "Eagle Benchmark\n\n";
    std::cout << "Usage: " << progName << " config.json [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --warmup=N              Warmup runs per prompt length (default: 2)\n";
    std::cout << "  --repeat=N              Measured runs per prompt length (default: 3)\n";
    std::cout << "  --max-new-tokens=N      Max new tokens per run (default: 5)\n";
    std::cout << "  --prompt-len=N,N,N      Prompt lengths (default: 8,16,32,64,128,256,512,1024)\n";
    std::cout << "  --backend=TYPE          Optional backend override (cpu/opencl/vulkan/metal/cuda)\n";
    std::cout << "  --precision=MODE        Optional precision override\n";
    std::cout << "  --thread=N              Executor thread count (default: 4)\n";
    std::cout << "  --help                  Show this help\n";
}

class BenchmarkRunner {
public:
    BenchmarkRunner(const std::string& configPath, const BenchmarkConfig& config)
        : mConfigPath(configPath), mConfig(config) {}

    bool run() {
        if (!loadLlm()) {
            return false;
        }

        printHeader();
        for (int promptLength : mConfig.promptLengths) {
            PromptResult result;
            if (!runPromptLength(promptLength, result)) {
                return false;
            }
            mResults.push_back(result);
        }
        printSummary();
        return true;
    }

private:
    std::string mConfigPath;
    BenchmarkConfig mConfig;
    std::unique_ptr<Llm> mLlm;
    std::vector<PromptResult> mResults;

    bool loadLlm() {
        mLlm.reset(Llm::createLLM(mConfigPath));
        if (!mLlm) {
            std::cerr << "Error: failed to create LLM from " << mConfigPath << "\n";
            return false;
        }

        mLlm->set_config("{\"tmp_path\":\"tmp\"}");
        if (!mConfig.backendType.empty() || !mConfig.precision.empty()) {
            std::ostringstream configJson;
            configJson << "{";
            bool hasItem = false;
            if (!mConfig.backendType.empty()) {
                configJson << "\"backend_type\":\"" << mConfig.backendType << "\"";
                hasItem = true;
            }
            if (!mConfig.precision.empty()) {
                if (hasItem) {
                    configJson << ",";
                }
                configJson << "\"precision\":\"" << mConfig.precision << "\"";
            }
            configJson << "}";
            mLlm->set_config(configJson.str());
        }

        if (!mLlm->load()) {
            std::cerr << "Error: failed to load model\n";
            return false;
        }
        if (!mLlm->isInSpeculative() || mLlm->getEagleContext() == nullptr) {
            std::cerr << "Error: current config is not Eagle speculative decoding\n";
            return false;
        }
        return true;
    }

    void printHeader() const {
        std::cout << "\n================================================\n";
        std::cout << "              Eagle Benchmark\n";
        std::cout << "================================================\n";
        std::cout << "Config: " << mConfigPath << "\n";
        std::cout << "Warmup: " << mConfig.warmupIterations
                  << ", Repeat: " << mConfig.repeatIterations
                  << ", Max new tokens: " << mConfig.maxNewTokens << "\n";
        std::cout << "Prompt lengths: ";
        for (int promptLength : mConfig.promptLengths) {
            std::cout << promptLength << " ";
        }
        std::cout << "\n";
        if (!mConfig.backendType.empty()) {
            std::cout << "Backend override: " << mConfig.backendType << "\n";
        }
        if (!mConfig.precision.empty()) {
            std::cout << "Precision override: " << mConfig.precision << "\n";
        }
        std::cout << "Metrics: target_prefill=base model prompt prefill, "
                  << "decode_total=whole speculative decode loop,\n"
                  << "         target_verify=target treeDecoding, "
                  << "draft_prefill=first draft topkGenerate, "
                  << "draft_decode=subsequent draft updateDraft\n";
        std::cout << "\n";
    }

    RunMetrics runOnce(const std::vector<int>& promptTokens) {
        RunMetrics metrics;
        mLlm->reset();
        mLlm->generate_init(nullptr, nullptr);
        mLlm->resetEagleContext();

        const auto outputTokens = mLlm->generate(promptTokens, mConfig.maxNewTokens);
        const auto* context = mLlm->getContext();
        const auto* eagleContext = mLlm->getEagleContext();
        if (context == nullptr || eagleContext == nullptr || outputTokens.empty()) {
            return metrics;
        }

        metrics.targetPrefillMs = context->prefill_us / 1000.0;
        metrics.decodeTotalMs = context->decode_us / 1000.0;
        metrics.outputTokens = context->gen_seq_len;
        metrics.targetVerifyMs = eagleContext->target_time_us / 1000.0;
        metrics.draftPrefillMs = eagleContext->draft_prefill_time_us / 1000.0;
        metrics.draftDecodeMs = eagleContext->draft_decode_time_us / 1000.0;
        metrics.acceptedTokens = static_cast<int>(eagleContext->accepted);
        metrics.draftTokens = static_cast<int>(eagleContext->draft);
        metrics.steps = static_cast<int>(eagleContext->steps);
        metrics.valid = metrics.outputTokens > 0;
        return metrics;
    }

    bool runPromptLength(int promptLength, PromptResult& result) {
        std::vector<int> promptTokens(promptLength, 16);
        std::cout << "### prompt_len=" << promptLength << "\n";

        for (int i = 0; i < mConfig.warmupIterations; ++i) {
            RunMetrics metrics = runOnce(promptTokens);
            if (!metrics.valid) {
                std::cerr << "Error: warmup failed for prompt_len=" << promptLength << "\n";
                return false;
            }
            std::cout << "  warmup " << (i + 1) << "/" << mConfig.warmupIterations
                      << ": target_prefill=" << formatMs(metrics.targetPrefillMs)
                      << ", decode_total=" << formatMs(metrics.decodeTotalMs) << "\n";
        }

        std::vector<double> targetPrefillSamples;
        std::vector<double> decodeTotalSamples;
        std::vector<double> targetVerifySamples;
        std::vector<double> draftPrefillSamples;
        std::vector<double> draftDecodeSamples;
        std::vector<double> outputTokenSamples;
        std::vector<double> acceptedSamples;
        std::vector<double> draftTokenSamples;
        std::vector<double> stepSamples;
        targetPrefillSamples.reserve(mConfig.repeatIterations);
        decodeTotalSamples.reserve(mConfig.repeatIterations);
        targetVerifySamples.reserve(mConfig.repeatIterations);
        draftPrefillSamples.reserve(mConfig.repeatIterations);
        draftDecodeSamples.reserve(mConfig.repeatIterations);
        outputTokenSamples.reserve(mConfig.repeatIterations);
        acceptedSamples.reserve(mConfig.repeatIterations);
        draftTokenSamples.reserve(mConfig.repeatIterations);
        stepSamples.reserve(mConfig.repeatIterations);

        for (int i = 0; i < mConfig.repeatIterations; ++i) {
            RunMetrics metrics = runOnce(promptTokens);
            if (!metrics.valid) {
                std::cerr << "Error: benchmark failed for prompt_len=" << promptLength << "\n";
                return false;
            }
            targetPrefillSamples.push_back(metrics.targetPrefillMs);
            decodeTotalSamples.push_back(metrics.decodeTotalMs);
            targetVerifySamples.push_back(metrics.targetVerifyMs);
            draftPrefillSamples.push_back(metrics.draftPrefillMs);
            draftDecodeSamples.push_back(metrics.draftDecodeMs);
            outputTokenSamples.push_back(metrics.outputTokens);
            acceptedSamples.push_back(metrics.acceptedTokens);
            draftTokenSamples.push_back(metrics.draftTokens);
            stepSamples.push_back(metrics.steps);

            std::cout << "  repeat " << (i + 1) << "/" << mConfig.repeatIterations
                      << ": target_prefill=" << formatMs(metrics.targetPrefillMs)
                      << ", decode_total=" << formatMs(metrics.decodeTotalMs)
                      << ", target_verify=" << formatMs(metrics.targetVerifyMs)
                      << ", draft_prefill=" << formatMs(metrics.draftPrefillMs)
                      << ", draft_decode=" << formatMs(metrics.draftDecodeMs)
                      << ", output_tokens=" << metrics.outputTokens
                      << ", accepted_tokens=" << metrics.acceptedTokens
                      << ", draft_tokens=" << metrics.draftTokens
                      << ", steps=" << metrics.steps << "\n";
        }

        result.promptLength = promptLength;
        result.targetPrefill = Stats::calculate(targetPrefillSamples);
        result.decodeTotal = Stats::calculate(decodeTotalSamples);
        result.targetVerify = Stats::calculate(targetVerifySamples);
        result.draftPrefill = Stats::calculate(draftPrefillSamples);
        result.draftDecode = Stats::calculate(draftDecodeSamples);
        result.outputTokens = Stats::calculate(outputTokenSamples);
        result.acceptedTokens = Stats::calculate(acceptedSamples);
        result.draftTokens = Stats::calculate(draftTokenSamples);
        result.steps = Stats::calculate(stepSamples);

        std::cout << "  result: target_prefill=" << formatStats(result.targetPrefill)
                  << ", decode_total=" << formatStats(result.decodeTotal)
                  << ", target_verify=" << formatStats(result.targetVerify)
                  << ", draft_prefill=" << formatStats(result.draftPrefill)
                  << ", draft_decode=" << formatStats(result.draftDecode) << "\n\n";
        return true;
    }

    static std::string formatMs(double value) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(3) << value << " ms";
        return os.str();
    }

    static std::string formatStats(const Stats& stats) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(3)
           << stats.mean << " ± " << stats.stddev << " ms";
        return os.str();
    }

    static void printStatsCell(const Stats& stats) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(3)
           << stats.mean << "±" << stats.stddev;
        std::cout << std::setw(20) << os.str();
    }

    static void printNumberStatsCell(const Stats& stats) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(1)
           << stats.mean << "±" << stats.stddev;
        std::cout << std::setw(18) << os.str();
    }

    void printSummary() const {
        std::cout << "================================================\n";
        std::cout << "Summary (mean ± std, ms)\n";
        std::cout << "================================================\n";
        std::cout << std::setw(10) << "Prompt"
                  << std::setw(20) << "TargetPrefill"
                  << std::setw(20) << "DecodeTotal"
                  << std::setw(20) << "TargetVerify"
                  << std::setw(20) << "DraftPrefill"
                  << std::setw(20) << "DraftDecode"
                  << std::setw(18) << "OutputTokens"
                  << std::setw(18) << "Accepted"
                  << std::setw(18) << "DraftTokens"
                  << std::setw(18) << "Steps"
                  << "\n";
        std::cout << std::string(182, '-') << "\n";
        for (const auto& result : mResults) {
            std::cout << std::setw(10) << result.promptLength;
            printStatsCell(result.targetPrefill);
            printStatsCell(result.decodeTotal);
            printStatsCell(result.targetVerify);
            printStatsCell(result.draftPrefill);
            printStatsCell(result.draftDecode);
            printNumberStatsCell(result.outputTokens);
            printNumberStatsCell(result.acceptedTokens);
            printNumberStatsCell(result.draftTokens);
            printNumberStatsCell(result.steps);
            std::cout << "\n";
        }
    }
};

} // namespace

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    const std::string configPath = argv[1];
    BenchmarkConfig config;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--warmup=") == 0) {
            config.warmupIterations = std::stoi(arg.substr(9));
        } else if (arg.find("--repeat=") == 0) {
            config.repeatIterations = std::stoi(arg.substr(9));
        } else if (arg.find("--iterations=") == 0) {
            config.repeatIterations = std::stoi(arg.substr(13));
        } else if (arg.find("--max-new-tokens=") == 0) {
            config.maxNewTokens = std::stoi(arg.substr(17));
        } else if (arg.find("--prompt-len=") == 0) {
            config.promptLengths = parseLengths(arg.substr(13));
        } else if (arg.find("--input-len=") == 0) {
            config.promptLengths = parseLengths(arg.substr(12));
        } else if (arg.find("--backend=") == 0) {
            config.backendType = arg.substr(10);
        } else if (arg.find("--precision=") == 0) {
            config.precision = arg.substr(12);
        } else if (arg.find("--thread=") == 0) {
            config.threadNum = std::stoi(arg.substr(9));
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    std::sort(config.promptLengths.begin(), config.promptLengths.end());
    config.promptLengths.erase(std::unique(config.promptLengths.begin(), config.promptLengths.end()),
                               config.promptLengths.end());
    if (config.promptLengths.empty()) {
        std::cerr << "Error: prompt length list is empty\n";
        return 1;
    }

    const std::string executorBackend = config.backendType.empty()
        ? parseStringValueFromConfig(configPath, "backend_type")
        : config.backendType;
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(executorTypeFromBackend(executorBackend),
                                                        backendConfig,
                                                        config.threadNum);
    MNN::Express::ExecutorScope scope(executor);

    BenchmarkRunner runner(configPath, config);
    return runner.run() ? 0 : 1;
}
