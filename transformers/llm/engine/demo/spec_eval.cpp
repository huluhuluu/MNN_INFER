// Speculative decoding acceptance and timing evaluation.

#include "llm/llm.hpp"
#include "speculative_decoding/generate.hpp"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <rapidjson/document.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MNN::Transformer;

struct EvalConfig {
    std::string configPath;
    std::string dataFile;
    std::string outputFile;
    std::string backend;
    std::string precision;
    std::string mode = "single";
    std::string templateFile;
    std::string templateName;
    int maxNewTokens = 64;
    int limit = 0;
    int batchSize = 0;
    double minAvgAccept = 0.0;
    bool noThinking = false;
};

struct PromptMessageTemplate {
    std::string role;
    std::string content;
};

class Evaluator {
public:
    explicit Evaluator(EvalConfig config) : mConfig(std::move(config)) {}

    bool run() {
        if (!loadTestData() || !loadPromptTemplate() || !loadModel()) {
            return false;
        }
        if (isSpeculativeMode()) {
            mLlm->resetSpecContext();
        }
        const int samples = (mConfig.mode == "single" || mConfig.mode == "ar")
            ? runSingle() : runBatch();
        const auto* spec = mLlm->getSpecContext();
        if (samples <= 0 || (isSpeculativeMode() && spec == nullptr)) {
            return false;
        }
        printResults(samples);
        if (!mConfig.outputFile.empty() && !saveResults(samples)) {
            return false;
        }
        if (spec != nullptr && mConfig.minAvgAccept > 0.0 && spec->avgAcceptLen() <= mConfig.minAvgAccept) {
            std::cerr << "Average accept length " << spec->avgAcceptLen()
                      << " did not exceed required threshold " << mConfig.minAvgAccept << "\n";
            return false;
        }
        return true;
    }

private:
    EvalConfig mConfig;
    std::unique_ptr<Llm> mLlm;
    std::vector<std::string> mPrompts;
    std::vector<PromptMessageTemplate> mPromptTemplate;
    int64_t mPromptTokens = 0;
    int64_t mGeneratedTokens = 0;
    uint64_t mWallTimeUs = 0;
    uint64_t mPrefillTimeUs = 0;
    uint64_t mDecodeTimeUs = 0;

    bool isSpeculativeMode() const {
        return mConfig.mode != "ar";
    }

    static void replaceAll(std::string& text, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = text.find(from, pos)) != std::string::npos) {
            text.replace(pos, from.size(), to);
            pos += to.size();
        }
    }

    bool loadTestData() {
        std::ifstream file(mConfig.dataFile);
        if (!file.is_open()) {
            std::cerr << "Cannot open data file: " << mConfig.dataFile << "\n";
            return false;
        }
        std::string line;
        while (std::getline(file, line)) {
            const size_t begin = line.find_first_not_of(" \t\r\n");
            const size_t end = line.find_last_not_of(" \t\r\n");
            if (begin == std::string::npos || line[begin] == '#') {
                continue;
            }
            mPrompts.push_back(line.substr(begin, end - begin + 1));
            if (mConfig.limit > 0 && static_cast<int>(mPrompts.size()) >= mConfig.limit) {
                break;
            }
        }
        return !mPrompts.empty();
    }

    bool loadPromptTemplate() {
        if (mConfig.templateFile.empty() && mConfig.templateName.empty()) {
            return true;
        }
        if (mConfig.templateFile.empty() || mConfig.templateName.empty()) {
            std::cerr << "--template-file and --template-name must be used together\n";
            return false;
        }
        std::ifstream file(mConfig.templateFile);
        std::ostringstream json;
        json << file.rdbuf();
        rapidjson::Document document;
        document.Parse(json.str().c_str());
        if (!file || document.HasParseError() || !document.IsObject() ||
            !document.HasMember(mConfig.templateName.c_str())) {
            std::cerr << "Invalid prompt template: " << mConfig.templateName << "\n";
            return false;
        }
        const auto& item = document[mConfig.templateName.c_str()];
        if (!item.IsObject() || !item.HasMember("messages") || !item["messages"].IsArray()) {
            return false;
        }
        for (const auto& message : item["messages"].GetArray()) {
            if (!message.IsObject() || !message.HasMember("role") || !message["role"].IsString() ||
                !message.HasMember("content") || !message["content"].IsString()) {
                return false;
            }
            mPromptTemplate.push_back({message["role"].GetString(), message["content"].GetString()});
        }
        return true;
    }

    ChatMessages messagesFor(const std::string& prompt) const {
        ChatMessages messages;
        for (const auto& item : mPromptTemplate) {
            std::string content = item.content;
            replaceAll(content, "{{question}}", prompt);
            replaceAll(content, "{question}", prompt);
            messages.emplace_back(item.role, content);
        }
        return messages;
    }

    std::vector<int> encodePrompt(const std::string& prompt) const {
        std::string rendered = mPromptTemplate.empty()
            ? mLlm->apply_chat_template(prompt)
            : mLlm->apply_chat_template(messagesFor(prompt));
        return mLlm->tokenizer_encode(rendered);
    }

    bool loadModel() {
        mLlm.reset(Llm::createLLM(mConfig.configPath));
        if (!mLlm) {
            return false;
        }
        std::ostringstream overrides;
        overrides << "{";
        bool comma = false;
        auto addString = [&](const char* key, const std::string& value) {
            if (value.empty()) return;
            if (comma) overrides << ",";
            overrides << "\"" << key << "\":\"" << value << "\"";
            comma = true;
        };
        addString("backend_type", mConfig.backend);
        addString("precision", mConfig.precision);
        if (comma) overrides << ",";
        if (mConfig.mode == "dual") {
            overrides << "\"packed_attention_mode\":true,\"dual_pipeline_mode\":true";
        } else if (mConfig.mode == "batch") {
            overrides << "\"packed_attention_mode\":true,\"dual_pipeline_mode\":false";
        } else if (mConfig.mode == "ar") {
            overrides << "\"speculative_type\":\"\",\"hidden_states\":false,"
                         "\"packed_attention_mode\":false,\"dual_pipeline_mode\":false";
        } else {
            overrides << "\"packed_attention_mode\":false,\"dual_pipeline_mode\":false";
        }
        overrides << "}";
        if (!mLlm->set_config(overrides.str())) {
            return false;
        }
        if (mConfig.noThinking) {
            mLlm->set_config(R"({"jinja":{"context":{"enable_thinking":false}}})");
        }
        MNN::Timer loadTimer;
        if (!mLlm->load()) {
            return false;
        }
        if ((isSpeculativeMode() && (!mLlm->isInSpec() || mLlm->getSpecContext() == nullptr)) ||
            (!isSpeculativeMode() && mLlm->isInSpec())) {
            return false;
        }
        std::cout << "mode=" << mConfig.mode << ", samples=" << mPrompts.size()
                  << ", draft_length=" << mLlm->getDraftLength()
                  << ", load_ms=" << loadTimer.durationInUs() / 1000.0 << "\n";
        return true;
    }

    int runSingle() {
        int completed = 0;
        for (const auto& prompt : mPrompts) {
            mLlm->reset();
            MNN::Timer timer;
            if (mPromptTemplate.empty()) {
                mLlm->response(prompt, nullptr, nullptr, mConfig.maxNewTokens);
            } else {
                mLlm->response(messagesFor(prompt), nullptr, nullptr, mConfig.maxNewTokens);
            }
            mWallTimeUs += timer.durationInUs();
            const auto* context = mLlm->getContext();
            mPromptTokens += context->prompt_len;
            mGeneratedTokens += context->gen_seq_len;
            mPrefillTimeUs += context->prefill_us;
            mDecodeTimeUs += context->decode_us;
            ++completed;
            printProgress(completed);
            if (context->status == LlmStatus::INTERNAL_ERROR) {
                return 0;
            }
        }
        return completed;
    }

    int runBatch() {
        std::vector<std::vector<int>> encoded;
        encoded.reserve(mPrompts.size());
        for (const auto& prompt : mPrompts) {
            encoded.push_back(encodePrompt(prompt));
            mPromptTokens += encoded.back().size();
        }
        const int batchSize = mConfig.batchSize > 0 ? mConfig.batchSize : static_cast<int>(encoded.size());
        int completed = 0;
        for (int begin = 0; begin < static_cast<int>(encoded.size()); begin += batchSize) {
            const int end = std::min(begin + batchSize, static_cast<int>(encoded.size()));
            std::vector<std::vector<int>> group(encoded.begin() + begin, encoded.begin() + end);
            mLlm->reset();
            MNN::Timer timer;
            auto outputs = mLlm->generate(group, nullptr, mConfig.maxNewTokens);
            mWallTimeUs += timer.durationInUs();
            const auto* context = mLlm->getContext();
            mPrefillTimeUs += context->prefill_us;
            mDecodeTimeUs += context->decode_us;
            if (outputs.size() != group.size() || mLlm->getContext()->status == LlmStatus::INTERNAL_ERROR) {
                std::cerr << "spec_eval batch generation failed: outputs=" << outputs.size()
                          << ", expected=" << group.size()
                          << ", status=" << static_cast<int>(mLlm->getContext()->status) << "\n";
                return 0;
            }
            for (const auto& output : outputs) {
                mGeneratedTokens += output.size();
            }
            completed += outputs.size();
            printProgress(completed);
        }
        return completed;
    }

    void printProgress(int completed) const {
        const auto* spec = mLlm->getSpecContext();
        std::cout << "\r[" << completed << "/" << mPrompts.size() << "] generated="
                  << mGeneratedTokens;
        if (spec != nullptr) {
            std::cout << ", steps=" << spec->steps << ", avg_accept="
                      << std::fixed << std::setprecision(3) << spec->avgAcceptLen();
        }
        std::cout << std::flush;
    }

    void printResults(int samples) const {
        const auto* spec = mLlm->getSpecContext();
        const double wallSeconds = mWallTimeUs / 1000000.0;
        std::cout << "\n\nmode: " << mConfig.mode
                  << "\nsamples: " << samples
                  << "\nprompt_tokens: " << mPromptTokens
                  << "\ngenerated_tokens: " << mGeneratedTokens;
        if (spec != nullptr) {
            std::cout << "\nsteps: " << spec->steps
                      << "\ndraft_tokens: " << spec->draft
                      << "\naccepted_tokens: " << spec->accepted
                      << "\navg_accept_length: " << spec->avgAcceptLen()
                      << "\naccept_rate: " << spec->acceptRate() * 100.0 << "%"
                      << "\ndraft_time_ms: " << spec->draft_time_us / 1000.0
                      << "\ntarget_time_ms: " << spec->target_time_us / 1000.0;
        }
        std::cout
                  << "\nwall_time_ms: " << mWallTimeUs / 1000.0
                  << "\nthroughput_tokens_per_s: " << (wallSeconds > 0.0 ? mGeneratedTokens / wallSeconds : 0.0);
        if (mPrefillTimeUs > 0 || mDecodeTimeUs > 0) {
            const double decodeSeconds = mDecodeTimeUs / 1000000.0;
            std::cout << "\nprefill_time_ms: " << mPrefillTimeUs / 1000.0
                      << "\ndecode_time_ms: " << mDecodeTimeUs / 1000.0
                      << "\ndecode_tokens_per_s: "
                      << (decodeSeconds > 0.0 ? mGeneratedTokens / decodeSeconds : 0.0);
        }
        if (spec != nullptr) {
            std::cout << "\ntheoretical_speedup: " << spec->theoreticalSpeedup() << "x";
            for (const auto& item : spec->accept_len_freq) {
                std::cout << "\naccept_len[" << item.first << "]=" << item.second;
            }
        }
        std::cout << "\n";
    }

    bool saveResults(int samples) const {
        const auto* spec = mLlm->getSpecContext();
        std::ofstream file(mConfig.outputFile);
        if (!file.is_open()) {
            return false;
        }
        file << std::fixed << std::setprecision(6);
        file << "{\n"
             << "  \"mode\": \"" << mConfig.mode << "\",\n"
             << "  \"samples\": " << samples << ",\n"
             << "  \"prompt_tokens\": " << mPromptTokens << ",\n"
             << "  \"generated_tokens\": " << mGeneratedTokens << ",\n"
             << "  \"wall_time_ms\": " << mWallTimeUs / 1000.0 << ",\n"
             << "  \"throughput_tokens_per_s\": "
             << (mWallTimeUs > 0 ? mGeneratedTokens * 1000000.0 / mWallTimeUs : 0.0);
        if (mPrefillTimeUs > 0 || mDecodeTimeUs > 0) {
            file << ",\n"
                 << "  \"prefill_time_ms\": " << mPrefillTimeUs / 1000.0 << ",\n"
                 << "  \"decode_time_ms\": " << mDecodeTimeUs / 1000.0 << ",\n"
                 << "  \"decode_tokens_per_s\": "
                 << (mDecodeTimeUs > 0 ? mGeneratedTokens * 1000000.0 / mDecodeTimeUs : 0.0);
        }
        if (spec != nullptr) {
            file << ",\n"
                 << "  \"steps\": " << spec->steps << ",\n"
                 << "  \"draft_tokens\": " << spec->draft << ",\n"
                 << "  \"accepted_tokens\": " << spec->accepted << ",\n"
                 << "  \"avg_accept_length\": " << spec->avgAcceptLen() << ",\n"
                 << "  \"accept_rate_percent\": " << spec->acceptRate() * 100.0 << ",\n"
                 << "  \"draft_time_ms\": " << spec->draft_time_us / 1000.0 << ",\n"
                 << "  \"target_time_ms\": " << spec->target_time_us / 1000.0 << ",\n"
                 << "  \"accept_length_frequency\": {\n";
            for (auto it = spec->accept_len_freq.begin(); it != spec->accept_len_freq.end(); ++it) {
                file << "    \"" << it->first << "\": " << it->second;
                if (std::next(it) != spec->accept_len_freq.end()) file << ",";
                file << "\n";
            }
            file << "  }";
        }
        file << "\n}\n";
        return true;
    }
};

static void printUsage(const char* program) {
    std::cout << "Usage: " << program << " config.json data.txt [options]\n"
              << "  --mode=single|batch|dual|ar\n"
              << "  --backend=cpu|opencl\n"
              << "  --precision=low|high\n"
              << "  --max-tokens=N --limit=N --batch-size=N\n"
              << "  --template-file=FILE --template-name=NAME --no-thinking\n"
              << "  --output=FILE --min-avg-accept=N\n";
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }
    EvalConfig config;
    config.configPath = argv[1];
    config.dataFile = argv[2];
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg.find("--mode=") == 0) config.mode = arg.substr(7);
        else if (arg.find("--backend=") == 0) config.backend = arg.substr(10);
        else if (arg.find("--precision=") == 0) config.precision = arg.substr(12);
        else if (arg.find("--max-tokens=") == 0) config.maxNewTokens = std::stoi(arg.substr(13));
        else if (arg.find("--limit=") == 0) config.limit = std::stoi(arg.substr(8));
        else if (arg.find("--batch-size=") == 0) config.batchSize = std::stoi(arg.substr(13));
        else if (arg.find("--output=") == 0) config.outputFile = arg.substr(9);
        else if (arg.find("--template-file=") == 0) config.templateFile = arg.substr(16);
        else if (arg.find("--template-name=") == 0) config.templateName = arg.substr(16);
        else if (arg.find("--min-avg-accept=") == 0) config.minAvgAccept = std::stod(arg.substr(17));
        else if (arg == "--no-thinking") config.noThinking = true;
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        }
    }
    if (config.mode != "single" && config.mode != "batch" && config.mode != "dual" &&
        config.mode != "ar") {
        std::cerr << "Invalid mode: " << config.mode << "\n";
        return 1;
    }
    MNNForwardType type = config.backend == "opencl" ? MNN_FORWARD_OPENCL : MNN_FORWARD_CPU;
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(type, backendConfig, 4);
    MNN::Express::ExecutorScope scope(executor);
    return Evaluator(std::move(config)).run() ? 0 : 2;
}
