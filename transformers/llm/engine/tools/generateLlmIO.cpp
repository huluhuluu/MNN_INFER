#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <ctime>

#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "core/MNNFileUtils.h"

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include <limits>
#include <set>

static void saveInputOutputs(const MNN::Express::Module::Info* info, std::vector<MNN::Express::VARP> inputs, std::vector<MNN::Express::VARP> outputs, const std::string & outputDir, int index) {
    MNN_ASSERT(info->inputNames.size() == inputs.size());
    MNN_ASSERT(info->outputNames.size() == outputs.size());
    for (int i=0; i<info->inputNames.size(); ++i) {
        inputs[i].fix(MNN::Express::VARP::CONSTANT);
        inputs[i]->setName(info->inputNames[i]);
    }
    for (int i=0; i<info->outputNames.size(); ++i) {
        outputs[i]->setName(info->outputNames[i]);
    }
    auto subDir = MNNFilePathConcat(outputDir, std::to_string(index));
    if (!(MNNCreateDir(subDir.c_str()))) {
        MNN_PRINT("Failed to create dir %s.\n", outputDir.c_str());
    }

    std::string inputPath = MNNFilePathConcat(subDir, "input.mnn");
    std::string outputPath = MNNFilePathConcat(subDir, "output.mnn");
    MNN::Express::Variable::save(inputs, inputPath.c_str());
    MNN::Express::Variable::save(outputs, outputPath.c_str());
    MNN_PRINT("Successfully generate %s and %s.\n", inputPath.c_str(), outputPath.c_str());
}

// logitsTokens is the number of trailing sequence positions for which logits
// are required.  Zero keeps the historical all-logits behavior.  The model
// interprets logits_index as the start position of the slice, so the value is
// baked into each shape-specific QNN graph during offline conversion.
static void createInputsForLLM(int seqLen, int hiddenSize, const std::string& attentionMaskType, int logitsTokens, std::vector<MNN::Express::VARP>& inputs) {
    if (attentionMaskType != "float") {
        MNN_ERROR("Don't support Attention Mask Type other than 'float', currently.\n");
        return;
    }

    MNN::Express::VARP inputIdx = MNN::Express::_Input({seqLen, 1, hiddenSize}, MNN::Express::NCHW, halide_type_of<float>());
    float * inputIdxData = inputIdx->writeMap<float>();
    for (int i = 0; i < seqLen * hiddenSize; ++i) {
        inputIdxData[i] = (float)(rand()) / RAND_MAX;
    }
    inputs.push_back(inputIdx);

    MNN::Express::VARP attentionMask =  MNN::Express::_Input({1, 1, seqLen, seqLen}, MNN::Express::NCHW, halide_type_of<float>());
    float * attentionMaskData = attentionMask->writeMap<float>();
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < seqLen; ++j) {
            attentionMaskData[i * seqLen + j] = (j > i) * std::numeric_limits<float>::lowest();
        }
    }
    inputs.push_back(attentionMask);

    MNN::Express::VARP positionIds = MNN::Express::_Input({seqLen}, MNN::Express::NCHW, halide_type_of<int>());
    int * positionIdsData = positionIds->writeMap<int>();
    for (int i = 0; i < seqLen; i++) {
        positionIdsData[i] = i;
    }
    inputs.push_back(positionIds);

    if (logitsTokens < 0) {
        MNN_ERROR("logits_tokens must be >= 0.\n");
        return;
    }
    int logitsIndexValue = 0;
    if (logitsTokens > 0) {
        logitsIndexValue = std::max(0, seqLen - logitsTokens);
    }
    MNN::Express::VARP logitsIndex = MNN::Express::_Const((const void *) &logitsIndexValue, {1}, MNN::Express::NHWC, halide_type_of<int>());
    inputs.push_back(logitsIndex);

    return;
}

static void createInputsForEagle(int seqLen, int hiddenSize, const std::string& attentionMaskType, std::vector<MNN::Express::VARP>& inputs) {
    if (attentionMaskType != "float") {
        MNN_ERROR("Don't support Attention Mask Type other than 'float', currently.\n");
        return;
    }
    inputs.push_back(MNN::Express::_Input({seqLen, 1, hiddenSize}, MNN::Express::NCHW, halide_type_of<float>()));
    inputs.push_back(MNN::Express::_Input({seqLen, 1, hiddenSize}, MNN::Express::NCHW, halide_type_of<float>()));
    auto attentionMask = MNN::Express::_Input({1, 1, seqLen, seqLen}, MNN::Express::NCHW, halide_type_of<float>());
    auto mask = attentionMask->writeMap<float>();
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < seqLen; ++j) {
            mask[i * seqLen + j] = (j > i) * std::numeric_limits<float>::lowest();
        }
    }
    inputs.push_back(attentionMask);
    auto positionIds = MNN::Express::_Input({seqLen}, MNN::Express::NCHW, halide_type_of<int>());
    auto positions = positionIds->writeMap<int>();
    for (int i = 0; i < seqLen; ++i) {
        positions[i] = i;
    }
    inputs.push_back(positionIds);
    int logitsIndexValue = 0;
    inputs.push_back(MNN::Express::_Const(&logitsIndexValue, {1}, MNN::Express::NHWC, halide_type_of<int>()));
}

static bool generateComponent(const std::string& modelPath,
                              const std::string& outputDir,
                              const std::string& jsonPath,
                              const std::string& component,
                              const std::vector<int>& buckets) {
    std::shared_ptr<MNN::Express::Module> net;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    if (component == "target") {
        inputNames = {"input_ids", "attention_mask", "position_ids", "logits_index"};
        outputNames = {"logits", "hidden_states"};
    } else if (component == "eagle") {
        inputNames = {"input_embed", "hidden_states", "attention_mask", "position_ids", "logits_index"};
        outputNames = {"logits", "out_hidden_states"};
    } else if (component == "eagle_fc") {
        inputNames = {"fc_hidden"};
        outputNames = {"hidden_states"};
    } else {
        MNN_ERROR("Unsupported component: %s.\n", component.c_str());
        return false;
    }

    int hiddenSize;
    std::string attentionMaskType;
    {
        std::ifstream ifs(jsonPath);
        if (!ifs.is_open()) {
            MNN_ERROR("Failed to open JSON config file: %s.\n", jsonPath.c_str());
            return false;
        }
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::Document doc;
        doc.ParseStream(isw);

        if (doc.HasParseError() || !doc.IsObject()) {
            MNN_ERROR("Failed to parse JSON config file: %s.\n", jsonPath.c_str());
            return false;
        }

        if (!doc.HasMember("hidden_size") || !doc["hidden_size"].IsInt()) {
            MNN_ERROR("'hidden_size' not found or not an integer in %s\n", jsonPath.c_str());
            return false;
        }
        hiddenSize = doc["hidden_size"].GetInt();

        if (!doc.HasMember("attention_mask") || !doc["attention_mask"].IsString()) {
            MNN_ERROR("'attention_mask' not found or not a string in %s\n", jsonPath.c_str());
            return false;
        }
        attentionMaskType = doc["attention_mask"].GetString();
    }

    // Load Model.
    MNN::ScheduleConfig config;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(config));
    rtmgr->setExternalFile((modelPath + ".weight").c_str());
    net.reset(MNN::Express::Module::load(inputNames, outputNames, modelPath.c_str(), rtmgr), MNN::Express::Module::destroy);
    rtmgr->setExternalFile("");
    if (!net) {
        MNN_ERROR("Failed to load %s component model: %s.\n", component.c_str(), modelPath.c_str());
        return false;
    }

    for (int bucket : buckets) {
        if (bucket <= 0) {
            MNN_ERROR("Invalid bucket size: %d.\n", bucket);
            return false;
        }
        std::vector<MNN::Express::VARP> inputs;
        std::vector<MNN::Express::VARP> outputs;
        if (component == "target") {
            createInputsForLLM(bucket, hiddenSize, attentionMaskType, bucket == 1, inputs);
        } else if (component == "eagle") {
            createInputsForEagle(bucket, hiddenSize, attentionMaskType, inputs);
        } else {
            inputs.push_back(MNN::Express::_Input({bucket, 1, hiddenSize * 3}, MNN::Express::NCHW, halide_type_of<float>()));
        }
        outputs = net->onForward(inputs);
        if (outputs.size() != outputNames.size()) {
            MNN_ERROR("%s forward returned %zu outputs, expected %zu.\n",
                      component.c_str(), outputs.size(), outputNames.size());
            return false;
        }
        saveInputOutputs(net->getInfo(), inputs, outputs, outputDir, bucket);
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./generateLlmIO model_dir output_dir [target|eagle|eagle_fc] [bucket ...]\n");
        return 1;
    }

    srand(time(NULL));
    std::string component = "target";
    std::vector<int> buckets;
    int firstBucketArg = 3;
    if (argc >= 4) {
        std::string arg = argv[3];
        if (arg == "target" || arg == "eagle" || arg == "eagle_fc") {
            component = arg;
            firstBucketArg = 4;
        } else {
            buckets.push_back(1);
        }
    }
    for (int i = firstBucketArg; i < argc; ++i) {
        buckets.push_back(atoi(argv[i]));
    }
    if (buckets.empty()) {
        buckets = {1, 128};
    }
    std::set<int> uniqueBuckets(buckets.begin(), buckets.end());
    buckets.assign(uniqueBuckets.begin(), uniqueBuckets.end());

    const std::string fileName = component == "target" ? "llm.mnn" : component + ".mnn";
    std::string modelPath = MNNFilePathConcat(argv[1], fileName);
    std::string llmConfigPath = std::string(argv[1]) + "/llm_config.json";
    FUNC_PRINT_ALL(modelPath.c_str(), s);
    FUNC_PRINT_ALL(llmConfigPath.c_str(), s);
    std::string outputDir = argv[2];

    if (!(MNNCreateDir(outputDir.c_str()))) {
        MNN_PRINT("Failed to create dir %s.\n", outputDir.c_str());
    }

    return generateComponent(modelPath, outputDir, llmConfigPath, component, buckets) ? 0 : 2;
}
