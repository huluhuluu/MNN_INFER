//
//  profiler.cpp
//

#include "llm/llm.hpp"
#include "llm/llm_profiler.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <sstream>
#include <unistd.h>
#include <vector>

using namespace MNN::Transformer;

static std::vector<int> parseList(const char* text) {
    std::vector<int> values;
    std::stringstream ss(text ? text : "");
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::stoi(item));
        }
    }
    return values;
}

static void runProfileTest(Llm* llm, const std::vector<int>& promptLens, const std::vector<int>& decodeLens,
                           int warmup, int repeat) {
    llm->enableProfiler(true);
    llm->setProfilerSpecialOps({"/lm/lm_head/Linear", "/lm_head/MatMul_output_0__matmul_converted", "*matmul_converted"});
    for (int promptLen : promptLens) {
        for (int decodeLen : decodeLens) {
            printf("\n\n### test response with prompt %d + prefill + tree decode %d tokens ###\n", promptLen, decodeLen);
            std::vector<int> promptTokens(promptLen, 16);
            sleep(3);
            for (int iter = 0; iter < warmup + repeat; ++iter) {
                llm->reset();
                llm->response(promptTokens, nullptr, nullptr, 0);
                auto profiler = llm->getProfiler();
                if (profiler) {
                    profiler->onDecodePhaseStart();
                }

                std::vector<int> decodeTokens(decodeLen, 16);
                auto inputEmbeds = llm->embedding(decodeTokens);
                auto attentionMask = llm->gen_attention_mask(decodeLen);
                auto positionIds = llm->gen_position_ids(decodeLen);
                llm->setKVCacheInfo(decodeLen, 0);

                if (profiler) {
                    profiler->onDecodeTokenBegin();
                }
                llm->forwardRaw(inputEmbeds, attentionMask, positionIds);
                if (profiler) {
                    llm->setActiveProfiler(profiler);
                    profiler->onDecodeTokenEnd(decodeLen);
                }
                if (iter < warmup) {
                    llm->clearProfilerInfo();
                }
            }
            auto profiler = llm->getProfiler();
            if (profiler) {
                profiler->onDecodePhaseEnd();
            }
            llm->printProfilerStats();
            llm->clearProfilerInfo();
        }
    }
}

static void printUsage(const char* prog) {
    printf("Usage: %s config.json [--prompt-lens=128] [--decode-lens=1,6] [--warmup=2] [--repeat=3]\n", prog);
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 0;
    }

    std::string configPath = argv[1];
    std::vector<int> promptLens{128};
    std::vector<int> decodeLens{1, 6};
    int warmup = 2;
    int repeat = 3;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--prompt-lens=") == 0) {
            promptLens = parseList(arg.c_str() + 14);
        } else if (arg.find("--decode-lens=") == 0) {
            decodeLens = parseList(arg.c_str() + 14);
        } else if (arg.find("--warmup=") == 0) {
            warmup = std::stoi(arg.substr(9));
        } else if (arg.find("--repeat=") == 0) {
            repeat = std::stoi(arg.substr(9));
        }
    }

    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);

    printf("config path is %s\n", configPath.c_str());
    std::unique_ptr<Llm> llm(Llm::createLLM(configPath));
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    {
        AUTOTIME;
        if (!llm->load()) {
            MNN_ERROR("LLM init error\n");
            return 0;
        }
    }

    runProfileTest(llm.get(), promptLens, decodeLens, warmup, repeat);
    return 0;
}
