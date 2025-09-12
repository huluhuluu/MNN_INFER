#include "llm/llm.hpp"
#include "core/MNNFileUtils.h"
#include "train/source/optimizer/SGD.hpp"
#include "train/source/grad/OpGrad.hpp"
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <regex>
#include <stdlib.h>
#include <initializer_list>
#include <rapidjson/document.h>
#include <thread>
#include <algorithm>
#include <numeric>

#define DYNAMICRUN
// #define STATICRUN
#define MNN_OPEN_TIME_TRACE


using namespace MNN::Transformer;

struct RuntimeParameters {
    std::vector<std::string>         model;
    std::vector<int>                 backends;
    std::vector<int>                 threads;
    bool                             useMmap;
    std::vector<int>                 power;
    std::vector<int>                 precision;
    std::vector<int>                 memory;
    std::vector<int>                 dynamicOption;
};

struct TestParameters {
    std::vector<int>                 nPrompt;
    std::vector<int>                 nGenerate;
    std::vector<std::pair<int, int>> nPrompGen;
    std::vector<int>                 nRepeat;
    std::string                      kvCache;
    std::string                      loadTime;
    int                              begin;
    int                              end;
};

struct CommandParameters {
    std::string         model;
    int                 backend;
    int                 threads;
    bool                useMmap;
    int                 power;
    int                 precision;
    int                 memory;
    int                 dynamicOption;

    int                 nPrompt;
    int                 nGenerate;
    std::pair<int, int> nPrompGen;
    int                 nRepeat;
    int                 begin;
    int                 end;
    std::string         kvCache;
    std::string         loadingTime;

};


static const RuntimeParameters runtimeParamsDefaults = {
    /* model                */ { "./Qwen2.5-1.5B-Instruct" },
    /* backends             */ { 0 },
    /* threads            */ { 4 },
    /* useMmap             */ false,
    /* power                */ { 0 },
    /* precision            */ { 2 },
    /* memory               */ { 2 },
    /* dynamicOption       */ { 0 }
};


static const TestParameters testParamsDefaults = {
    /* nPrompt             */ { 512 },
    /* nGenerate           */ { 128 },
    /* nPrompGen           */ {std::make_pair(512, 128)},
    /* nRepeat             */ { 100 },
    /* kvCache             */ { "false" },
    /* loadingTime         */ {"false"},
    /* begin               */ 20,
    /* end                 */ 80,
};


struct commandParametersInstance {

    CommandParameters mCmdParam;

    commandParametersInstance(CommandParameters cmdParam) {
        mCmdParam.model          = cmdParam.model;
        mCmdParam.backend        = cmdParam.backend;
        mCmdParam.threads        = cmdParam.threads;
        mCmdParam.useMmap        = cmdParam.useMmap;
        mCmdParam.power          = cmdParam.power;
        mCmdParam.precision      = cmdParam.precision;
        mCmdParam.memory         = cmdParam.memory;
        mCmdParam.dynamicOption  = cmdParam.dynamicOption;

        mCmdParam.nPrompt        = cmdParam.nPrompt;
        mCmdParam.nGenerate      = cmdParam.nGenerate;
        mCmdParam.nPrompGen      = cmdParam.nPrompGen;
        mCmdParam.nRepeat        = cmdParam.nRepeat;
        mCmdParam.kvCache        = cmdParam.kvCache;
        mCmdParam.loadingTime    = cmdParam.loadingTime;
        mCmdParam.begin          = cmdParam.begin;
        mCmdParam.end            = cmdParam.end;
    }

    CommandParameters get_cmd_parameters() const {
        return mCmdParam;
    }

    bool equal_runtime_params(const commandParametersInstance & other) const {
        return mCmdParam.model == other.mCmdParam.model &&
        mCmdParam.useMmap == other.mCmdParam.useMmap &&
        mCmdParam.power == other.mCmdParam.power &&
        mCmdParam.precision == other.mCmdParam.precision &&
        mCmdParam.memory == other.mCmdParam.memory &&
        mCmdParam.dynamicOption == other.mCmdParam.dynamicOption;
    }
};

template <typename T> static T avg(const std::vector<T> & v) {
    if (v.empty()) {
        return 0;
    }
    T sum = std::accumulate(v.begin(), v.end(), T(0));
    return sum / (T) v.size();
}

template <typename T> static T stdev(const std::vector<T> & v) {
    if (v.size() <= 1) {
        return 0;
    }
    T mean   = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    T stdev  = std::sqrt(sq_sum / (T) (v.size() - 1) - mean * mean * (T) v.size() / (T) (v.size() - 1));
    return stdev;
}

template <class T> static std::string join(const std::vector<T> & values, const std::string & delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) {
            str << delim;
        }
    }
    return str.str();
}

struct TestInstance {
//    static const std::string build_commit;
    std::string              modelConfigFile;
    std::string              modelType;
    uint64_t                 modelSize;
    int                      threads;
    bool                     useMmap;
    int                      nPrompt;
    int                      nGenerate;
    std::vector<int64_t>     prefillUs;
    std::vector<int64_t>     decodeUs;
    std::vector<int64_t>     samplesUs;
    std::vector<int64_t>     fwdUs;
    std::vector<int64_t>     bwdUs;
    std::vector<int64_t>     fbwdUs;
    std::vector<double>      loadingS;
    int                      backend;
    int                      precision;
    int                      power;
    int                      memory;
    int                      dynamicOption;

    TestInstance(const commandParametersInstance & instance) {

        modelConfigFile = instance.mCmdParam.model;
        threads         = instance.mCmdParam.threads;
        useMmap          = instance.mCmdParam.useMmap;
        nPrompt          = instance.mCmdParam.nPrompt;
        nGenerate             = instance.mCmdParam.nGenerate;
        backend           = instance.mCmdParam.backend;
        precision         = instance.mCmdParam.precision;
        memory            = instance.mCmdParam.memory;
        power             = instance.mCmdParam.power;
        dynamicOption     = instance.mCmdParam.dynamicOption;
    }

    std::vector<double> getTokensPerSecond(int n_tokens, std::vector<int64_t> cost_us) const {
        std::vector<double> ts;
        std::transform(cost_us.begin(), cost_us.end(), std::back_inserter(ts), [n_tokens](int64_t t) { return 1e6 * n_tokens / t; });
        return ts;
    }

    double getAvgUs(std::vector<double> v) const { return ::avg(v); }
    double getStdevUs(std::vector<double> v) const { return ::stdev(v); }
    enum fieldType { STRING, BOOL, INT, FLOAT };

    static fieldType getFieldType(const std::string & field) {
        if (field == "threads") {
            return INT;
        }
        if (field == "useMmap") {
            return BOOL;
        }
        if (field == "t/s" || field == "modelSize" || field == "prefill&decode speed (tok/s)" || field == "Fwd Time(s)" || field == "Bwd Time(s)" || field == "FBwd Time(s)") {
            return FLOAT;
        }
        return STRING;
    }
};

static std::string pairString(const std::pair<int, int> & p) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d,%d", p.first, p.second);
    return buf;
}

template <typename T, typename F> static std::vector<std::string> transform2String(const std::vector<T> & values, F f) {
    std::vector<std::string> str_values;
    std::transform(values.begin(), values.end(), std::back_inserter(str_values), f);
    return str_values;
}

template<class T>
static std::vector<T> splitString(const std::string & str, char delim) {
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream tokenStream(token);
        tokenStream >> value;
        values.push_back(value);
    }
    return values;
}

struct Printer {
    virtual ~Printer() {}

    FILE * fout;

    virtual void printHeader(const RuntimeParameters & rp, const TestParameters & tp) { (void) rp; (void) tp; }

    virtual void printPerformance(const TestInstance & t) = 0;

//    virtual void print_footer() {}
};

struct markdownPrinter : public Printer {
    std::vector<std::string> fields;

    static int getFieldWidth(const std::string & field) {
        if (field == "model") {
            return -30;
        }
        if (field == "prefill&decode speed (tok/s)") {
            return 20;
        }
        if (field == "threads") {
            return 7;
        }
        if (field == "useMmap") {
            return 4;
        }
        if (field == "test") {
            return -13;
        }

        int width = std::max((int) field.length(), 10);

        if (TestInstance::getFieldType(field) == TestInstance::STRING) {
            return -width;
        }
        return width;
    }

    static std::string getFieldDisplayName(const std::string & field) {
        if (field == "useMmap") {
            return "mmap";
        }
        return field;
    }

    void printHeader(const RuntimeParameters & rp, const TestParameters & tp) override {
        // select fields to print
        fields.emplace_back("model");
        // fields.emplace_back("modelSize");
        fields.emplace_back("backend");
        fields.emplace_back("threads");
        fields.emplace_back("memory");
        fields.emplace_back("Fwd Time(s)");
        // fields.emplace_back("Bwd Time(s)");
        fields.emplace_back("FBwd Time(s)");
        fields.emplace_back("loadingTime(s)");
        fields.emplace_back("test");

        if (rp.precision.size() > 1) {
            fields.emplace_back("precision");
        }
        if (rp.dynamicOption.size() > 1) {
            fields.emplace_back("dynamicOption");
        }
        if (rp.useMmap) {
            fields.emplace_back("useMmap");
        }


        fprintf(fout, "|");
        for (const auto & field : fields) {
            fprintf(fout, " %*s |", getFieldWidth(field), getFieldDisplayName(field).c_str());
        }
        fprintf(fout, "\n");
        fprintf(fout, "|");
        for (const auto & field : fields) {
            int width = getFieldWidth(field);
            fprintf(fout, " %s%s |", std::string(std::abs(width) - 1, '-').c_str(), width > 0 ? ":" : "-");
        }
        fprintf(fout, "\n");
    }

    void printPerformance(const TestInstance & t) override {
        fprintf(fout, "|");
        for (const auto & field : fields) {
            std::string value;
            char        buf[128];
            if (field == "model") {
                value = t.modelType;
            } else if (field == "modelSize") {
                if (t.modelSize < 1024 * 1024 * 1024) {
                    snprintf(buf, sizeof(buf), "%.2f MiB", t.modelSize / 1024.0 / 1024.0);
                } else {
                    snprintf(buf, sizeof(buf), "%.2f GiB", t.modelSize / 1024.0 / 1024.0 / 1024.0);
                }
                value = buf;
            }  else if (field == "backend") {
                if (t.backend == 1) value = "METAL";
                else if (t.backend == 3) value = "OPENCL";
                else value = "CPU";
            } else if (field == "test") {
                if (t.nPrompt > 0 && t.nGenerate == 0) {
                    snprintf(buf, sizeof(buf), "pp%d", t.nPrompt);
                } else if (t.nGenerate > 0 && t.nPrompt == 0) {
                    snprintf(buf, sizeof(buf), "tg%d", t.nGenerate);
                } else {
                    snprintf(buf, sizeof(buf), "pp%d+tg%d", t.nPrompt, t.nGenerate);
                }
                value = buf;
            } else if (field == "llm_demo") {
                snprintf(buf, sizeof(buf), "prompt=%d<br>decode=%d", t.nPrompt, t.nGenerate);
                value = buf;
            } else if (field == "t/s") {
                auto spd = t.getTokensPerSecond(t.nPrompt + t.nGenerate, t.samplesUs);
                snprintf(buf, sizeof(buf), "%.2f ± %.2f", t.getAvgUs(spd), t.getStdevUs(spd));
                value = buf;
            } else if (field == "Fwd Time(s)") {
                std::vector<double> profilerUs;
                std::transform(t.fwdUs.begin(), t.fwdUs.end(), std::back_inserter( profilerUs), [](int64_t t) { return (t * 1.0) / 1e6; });
                if (profilerUs.empty()) {
                    snprintf(buf, sizeof(buf), "0.00 ± 0.00");
                } else {
                    snprintf(buf, sizeof(buf), "%.2f ± %.2f", t.getAvgUs(profilerUs), t.getStdevUs(profilerUs));
                }
                value = buf;
            }
            else if (field == "Bwd Time(s)") {
                std::vector<double> profilerUs;
                std::transform(t.bwdUs.begin(), t.bwdUs.end(), std::back_inserter( profilerUs), [](int64_t t) { return (t * 1.0) / 1e6; });
                if (profilerUs.empty()) {
                    snprintf(buf, sizeof(buf), "0.00 ± 0.00");
                } else {
                    snprintf(buf, sizeof(buf), "%.2f ± %.2f", t.getAvgUs(profilerUs), t.getStdevUs(profilerUs));
                }
                value = buf;
            }
            else if (field == "FBwd Time(s)") {
                std::vector<double> profilerUs;
                std::transform(t.fbwdUs.begin(), t.fbwdUs.end(), std::back_inserter( profilerUs), [](int64_t t) { return (t * 1.0) / 1e6; });
                if (profilerUs.empty()) {
                    snprintf(buf, sizeof(buf), "0.00 ± 0.00");
                } else {
                    snprintf(buf, sizeof(buf), "%.2f ± %.2f", t.getAvgUs(profilerUs), t.getStdevUs(profilerUs));
                }
                value = buf;
            }
            else if (field == "speed(tok/s)") {
                auto decode_speed = t.getTokensPerSecond(t.nGenerate, t.decodeUs);
                auto prefill_speed = t.getTokensPerSecond(t.nPrompt, t.prefillUs);
                snprintf(buf, sizeof(buf), "%.2f ± %.2f<br>%.2f ± %.2f", t.getAvgUs(prefill_speed), t.getStdevUs(prefill_speed), t.getAvgUs(decode_speed), t.getStdevUs(decode_speed));
                value = buf;
            } else if (field == "precision") {
                if (t.precision == 2) value = "Low";
                else if (t.precision == 0) value = "Normal";
                else value = "High";
            } else if (field == "memory") {
                if (t.memory == 2) value = "Low";
                else if (t.memory == 0) value = "Normal";
                else value = "High";
            } else if (field == "power") {
                if (t.power == 2) value = "Low";
                else if (t.power == 0) value = "Normal";
                else value = "High";
            } else if (field == "threads") {
                snprintf(buf, sizeof(buf), "%d", t.threads);
                value = buf;
            } else if (field == "loadingTime(s)") {
                snprintf(buf, sizeof(buf), "%.2f ± %.2f", t.getAvgUs(t.loadingS), t.getStdevUs(t.loadingS));
                value = buf;
            } else if (field == "useMmap") {
                if (t.useMmap) value = "true";
                else value = "false";
            }
            else {
                assert(false);
                MNN_ERROR("llm bench print fields error\n");
                return;
            }

            int width = getFieldWidth(field);
            if (field == "prefill&decode speed (tok/s)" || field == "t/s") {
                // HACK: the utf-8 character is 2 bytes
                width += 1;
            }
            fprintf(fout, " %*s |", width, value.c_str());
        }
        fprintf(fout, "\n");
    }

//    void print_footer() override {
//        fprintf(fout, "\nbuild: %s \n", test::build_commit.c_str());
//    }
};

static FILE* openFile(const char* file, bool read) {
#if defined(_MSC_VER)
    wchar_t wFilename[1024];
    if (0 == MultiByteToWideChar(CP_ACP, 0, file, -1, wFilename, sizeof(wFilename))) {
        return nullptr;
    }
#if _MSC_VER >= 1400
    FILE* mFile = nullptr;
    if (read) {
        if (0 != _wfopen_s(&mFile, wFilename, L"r")) {
            return nullptr;
        }
    } else {
        if (0 != _wfopen_s(&mFile, wFilename, L"a")) {
            return nullptr;
        }
    }
    return mFile;
#else
    if (read) {
        return _wfopen(wFilename, L"r");
    } else {
        return _wfopen(wFilename, L"a");
    }
#endif
#else
    if (read) {
        return fopen(file, "r");
    } else {
        return fopen(file, "a");
    }
#endif
    return nullptr;
}

static std::vector<commandParametersInstance> get_cmd_params_instances(const RuntimeParameters & rp, const TestParameters& tp) {
    std::vector<commandParametersInstance> instances;

    // this ordering minimizes the number of times that each model needs to be reloaded
    // clang-format off
    for (const auto & m : rp.model)
    for (const auto & backend : rp.backends)
    for (const auto & precision : rp.precision)
    for (const auto & memory : rp.memory)
    for (const auto & power : rp.power)
    for (const auto & nt : rp.threads)
    for (const auto & dyop : rp.dynamicOption)
        // llama.cpp llama-bench's test standard
        for (const auto & nPrompt : tp.nPrompt) {
            if (nPrompt == 0) {
                continue;
            }
            CommandParameters tmpParam;
            tmpParam.model = m;
            tmpParam.nPrompt = nPrompt;
            tmpParam.nGenerate = 0;
            tmpParam.threads = nt;
            tmpParam.useMmap = rp.useMmap;
            tmpParam.backend = backend;
            tmpParam.power = power;
            tmpParam.precision = precision;
            tmpParam.memory = memory;
            tmpParam.dynamicOption = dyop;
            tmpParam.nRepeat = tp.nRepeat[0];
            tmpParam.begin = tp.begin;
            tmpParam.end = tp.end;
            tmpParam.kvCache = "false";
            tmpParam.loadingTime = tp.loadTime;
            auto instance = commandParametersInstance(tmpParam);
            instances.push_back(instance);
        }

    return instances;
}

std::string getDirectoryOf(const std::string& file_path, std::string& modelname) {
    // weight filename
    std::string weight_name = "llm.mnn.weight";
    std::ifstream file(file_path.c_str());
    std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    rapidjson::Document doc;
    doc.Parse(json_str.c_str());

    if (doc.HasMember("llm_weight") && doc["llm_weight"].IsString()) {
        weight_name = doc["llm_weight"].GetString();
    }

    size_t pos = file_path.find_last_of("/\\");
    if (pos == std::string::npos) {
        MNN_ERROR("Invalid model config path\n");
        return "";
    }
    auto dir = file_path.substr(0, pos);
    pos = dir.find_last_of("/\\");
    modelname = dir.substr(pos + 1, -1);
    return MNNFilePathConcat(dir, weight_name);
}

static void printUsage(int /* argc */, char ** argv) {
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help\n");
    printf("  -m, --model <filename>                    (default: %s)\n", join(runtimeParamsDefaults.model, ",").c_str());
    printf("  -a, --backends <cpu,opencl,metal>         (default: %s)\n", "cpu");
    printf("  -c, --precision <n>                       (default: %s) | Note: (0:Normal(for cpu bakend, 'Nornal' is 'High'),1:High,2:Low)\n", join(runtimeParamsDefaults.precision, ",").c_str());
    printf("  -t, --threads <n>                         (default: %s)\n", join(runtimeParamsDefaults.threads, ",").c_str());
    printf("  -p, --n-prompt <n>                        (default: %s)\n", join(testParamsDefaults.nPrompt, ",").c_str());
    printf("  -pg <pp,tg>                               (default: %s)\n", join(transform2String(testParamsDefaults.nPrompGen, pairString), ",").c_str());
    printf("  -mmp, --mmap <0|1>                        (default: %s)\n", "0");
    printf("  -rep, --n-repeat <n>                      (default: %s)\n", join(testParamsDefaults.nRepeat, ",").c_str());
    printf("  -fp, --file-print <stdout|filename>       (default: %s)\n", "stdout");
    printf("  -b, --begin <n>                           (default: %d)\n", testParamsDefaults.begin);
    printf("  -e, --end <n>                             (default: %d)\n", testParamsDefaults.end);
}


static bool parseCmdParams(int argc, char ** argv, RuntimeParameters & runtimeParams, TestParameters & testParams, FILE** outfile, bool& helpInfo) {
    std::string       arg;
    bool              invalidParam = false;
    const std::string argPrefix    = "--";
    const char        splitDelim   = ',';

    runtimeParams.useMmap = runtimeParamsDefaults.useMmap;
    testParams.loadTime = testParamsDefaults.loadTime;

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, argPrefix.size(), argPrefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-h" || arg == "--help") {
            printUsage(argc, argv);
            helpInfo = true;
            return true;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<std::string>(argv[i], splitDelim);
            runtimeParams.model.insert(runtimeParams.model.end(), p.begin(), p.end());
        } else if (arg == "-p" || arg == "--n-prompt") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            testParams.nPrompt.insert(testParams.nPrompt.end(), p.begin(), p.end());
        } else if (arg == "-n" || arg == "--n-gen") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            testParams.nGenerate.insert(testParams.nGenerate.end(), p.begin(), p.end());
        } else if (arg == "-pg") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<std::string>(argv[i], ',');
            if (p.size() != 2) {
                invalidParam = true;
                break;
            }
            testParams.nPrompGen.push_back({ std::stoi(p[0]), std::stoi(p[1]) });
        } else if (arg == "-a" || arg == "--backends") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto ba = splitString<std::string>(argv[i], splitDelim);
            std::vector<int> p;
            for (auto& type: ba) {
                if (type == "metal") {
                    p.emplace_back(1);
                } else if (type == "opencl") {
                    p.emplace_back(3);
                } else {
                    p.emplace_back(0);
                }
            }
            runtimeParams.backends.insert(runtimeParams.backends.end(), p.begin(), p.end());
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            runtimeParams.threads.insert(runtimeParams.threads.end(), p.begin(), p.end());
        } else if (arg == "-mmp" || arg == "--mmap") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<bool>(argv[i], splitDelim);
            runtimeParams.useMmap = p[0];
        } else if (arg == "-c" || arg == "--precision") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            runtimeParams.precision.insert(runtimeParams.precision.end(), p.begin(), p.end());
        } else if (arg == "--memory") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            runtimeParams.memory.insert(runtimeParams.memory.end(), p.begin(), p.end());
        } else if (arg == "--power") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            runtimeParams.power.insert(runtimeParams.power.end(), p.begin(), p.end());
        } else if (arg == "-dyo" || arg == "--dynamicOption") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            runtimeParams.dynamicOption.insert(runtimeParams.dynamicOption.end(), p.begin(), p.end());
        } else if (arg == "-rep" || arg == "--n-repeat") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            testParams.nRepeat.insert(testParams.nRepeat.end(), p.begin(), p.end());
        } else if (arg == "-b" || arg == "--begin") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            testParams.begin = p[0];
        } else if (arg == "-e" || arg == "--end") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<int>(argv[i], splitDelim);
            testParams.end = p[0];
        } else if (arg == "-fp" || arg == "--file-print") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<std::string>(argv[i], splitDelim);
            if (!MNNFileExist(p[0].c_str())) {
                MNNCreateFile(p[0].c_str());
            }
            *outfile = openFile(p[0].c_str(), false);
        } else if (arg == "-load" || arg == "--loading-time") {
            if (++i >= argc) {
                invalidParam = true;
                break;
            }
            auto p = splitString<std::string>(argv[i], splitDelim);
            testParams.loadTime = p[0];
        }
        else {
            invalidParam = true;
            break;
        }
    } // parse end


    if (invalidParam) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        printUsage(argc, argv);
        return false;
    }

    // set defaults
    if (runtimeParams.model.empty()) {
        runtimeParams.model = runtimeParamsDefaults.model;
    }
    if (testParams.nPrompt.empty()) {
        testParams.nPrompt = testParamsDefaults.nPrompt;
    }
    if (testParams.nGenerate.empty()) {
        testParams.nGenerate = testParamsDefaults.nGenerate;
    }
    if (runtimeParams.backends.empty()) {
        runtimeParams.backends = runtimeParamsDefaults.backends;
    }
    if (runtimeParams.memory.empty()) {
        runtimeParams.memory = runtimeParamsDefaults.memory;
    }
    if (runtimeParams.precision.empty()) {
        runtimeParams.precision = runtimeParamsDefaults.precision;
    }
    if (runtimeParams.power.empty()) {
        runtimeParams.power = runtimeParamsDefaults.power;
    }
    if (runtimeParams.threads.empty()) {
        runtimeParams.threads = runtimeParamsDefaults.threads;
    }
    if (runtimeParams.dynamicOption.empty()) {
        runtimeParams.dynamicOption = runtimeParamsDefaults.dynamicOption;
    }
    if (testParams.nRepeat.empty()) {
        testParams.nRepeat = testParamsDefaults.nRepeat;
    }
    if (testParams.begin < 0) {
        testParams.begin = testParamsDefaults.begin;
    }
    if (testParams.end < 0) {
        testParams.end = testParamsDefaults.end;
    }

    return true;
}


static Llm* buildLLM(const std::string& config_path, int backend, int memory, int precision, int threads, int power, int dynamic_option, bool use_mmap) {
    auto llmPtr = Llm::createLLM(config_path);
    std::map<int, std::string> lever = {{0,"normal"}, {1, "high"}, {2, "low"}};
    std::map<int, std::string> backend_type = {{0, "cpu"}, {1, "metal"}, {3, "opencl"}};
    std::map<bool, std::string> mmap = {{true,"true"}, {false, "false"}};

    bool setSuccess = true;
    // setSuccess &= llmPtr->set_config("{\"precision\":\"" + lever[precision] + "\"}");
    // if (!setSuccess) {
    //     MNN_ERROR("precison for LLM config set error\n");
    //     return nullptr;
    // }
    // setSuccess &= llmPtr->set_config("{\"memory\":\"" + lever[memory] + "\"}");
    // if (!setSuccess) {
    //     MNN_ERROR("memory for LLM config set error\n");
    //     return nullptr;
    // }
    // setSuccess &= llmPtr->set_config("{\"power\":\"" + lever[power] + "\"}");
    // if (!setSuccess) {
    //     MNN_ERROR("power for LLM config set error\n");
    //     return nullptr;
    // }
    // setSuccess &= llmPtr->set_config("{\"backend_type\":\"" + backend_type[backend] + "\"}");
    // if (!setSuccess) {
    //     MNN_ERROR("backend_type for LLM config set error\n");
    //     return nullptr;
    // }
    // setSuccess &= llmPtr->set_config("{\"thread_num\":" + std::to_string(threads) + "}");
    // if (!setSuccess) {
    //     MNN_ERROR("thread_num for LLM config set error\n");
    //     return nullptr;
    // }
    // setSuccess &= llmPtr->set_config("{\"dynamic_option\":" + std::to_string(dynamic_option) + "}");
    // if (!setSuccess) {
    //     MNN_ERROR("dynamic_option for LLM config set error\n");
    //     return nullptr;
    // }
    // setSuccess &= llmPtr->set_config("{\"use_mmap\":" + mmap[use_mmap] + "}");
    // if (!setSuccess) {
    //     MNN_ERROR("use_mmap for LLM config set error\n");
    //     return nullptr;
    // }
    setSuccess &= llmPtr->set_config("{\"tmp_path\":\"tmp\"}");
    return llmPtr;
}

static void tuning_prepare(Llm* llm) {
    llm->tuning(OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
}

int main(int argc, char ** argv) {
    RuntimeParameters runtimeParams;
    TestParameters testParams;
    FILE* outfile = stdout;
    bool helpInfo = false;
    bool parseSuccess = parseCmdParams(argc, argv, runtimeParams, testParams, &outfile, helpInfo);
    if (!parseSuccess) {
        MNN_ERROR("Parse arguments error\n");
        return -1;
    }
    if (parseSuccess && helpInfo) {
        return 0;
    }
    std::vector<commandParametersInstance> paramsInstances = get_cmd_params_instances(runtimeParams, testParams);
    std::unique_ptr<Printer> printer_(new markdownPrinter());
    bool printHeader = true;

    for (auto instance: paramsInstances){
        TestInstance t(instance);

        MNN::BackendConfig backendConfig;
        auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
        MNN::Express::ExecutorScope scope(executor);

        auto llmPtr = buildLLM(instance.mCmdParam.model, instance.mCmdParam.backend, instance.mCmdParam.memory, instance.mCmdParam.precision, instance.mCmdParam.threads, instance.mCmdParam.power, instance.mCmdParam.dynamicOption, instance.mCmdParam.useMmap);
        std::unique_ptr<Llm> llm(llmPtr);

        for (int k = 0; k < 1; ++k) {
            Timer loadingCost;
            llm->load();
            t.loadingS.push_back((double)loadingCost.durationInUs() / 1e6);
        }
        break;
        auto context = llm->getContext();
        if (instance.mCmdParam.nGenerate > 0) {
            llm->set_config("{\"max_new_tokens\":1}");
        }
        
        auto prompt_tokens = instance.mCmdParam.nPrompt;
        // TODO: forward time too long
        
        // TODO: too long prompt casue heap overflow
        /*
            =================================================================
            ==2180924==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x62d000008450 at pc 0x7f521fc1a0af bp 0x7f521cafdf30 sp 0x7f521cafdf28
            WRITE of size 4 at 0x62d000008450 thread T1
                #0 0x7f521fc1a0ae in void MNN::KVCacheManager::pack_key<float>(MNN::Tensor const*, int, int) /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/KVCacheManager.cpp:622:94
                #1 0x7f521fc177d2 in MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_0::operator()(int) const /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/KVCacheManager.cpp:676:17
                #2 0x7f521fc1742d in void std::__invoke_impl<void, MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_0&, int>(std::__invoke_other, MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_0&, int&&) /usr/include/c++/11/bits/invoke.h:61:14
                #3 0x7f521fc17371 in std::enable_if<__and_<std::is_void<void>, std::__is_invocable<MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_0&, int> >::value, void>::type std::__invoke_r<void, MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_0&, int>(MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_0&, int&&) /usr/include/c++/11/bits/invoke.h:154:7
                #4 0x7f521fc17211 in std::_Function_handler<void (int), MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_0>::_M_invoke(std::_Any_data const&, int&&) /usr/include/c++/11/bits/std_function.h:290:9
                #5 0x7f52211eb521 in std::function<void (int)>::operator()(int) const /usr/include/c++/11/bits/std_function.h:590:9
                #6 0x7f521fc1b82f in MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_1::operator()(int) const /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/KVCacheManager.cpp:682:9
                #7 0x7f521fc1b7cd in void std::__invoke_impl<void, MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_1&, int>(std::__invoke_other, MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_1&, int&&) /usr/include/c++/11/bits/invoke.h:61:14
                #8 0x7f521fc1b721 in std::enable_if<__and_<std::is_void<void>, std::__is_invocable<MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_1&, int> >::value, void>::type std::__invoke_r<void, MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_1&, int>(MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_1&, int&&) /usr/include/c++/11/bits/invoke.h:154:7
                #9 0x7f521fc1b611 in std::_Function_handler<void (int), MNN::KVCacheManager::onPushBack(MNN::Tensor const*, MNN::Tensor const*)::$_1>::_M_invoke(std::_Any_data const&, int&&) /usr/include/c++/11/bits/std_function.h:290:9
                #10 0x7f52211eb521 in std::function<void (int)>::operator()(int) const /usr/include/c++/11/bits/std_function.h:590:9
                #11 0x7f521fc28512 in MNN::ThreadPool::ThreadPool(int)::$_1::operator()() const /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/ThreadPool.cpp:63:29
                #12 0x7f521fc2825c in void std::__invoke_impl<void, MNN::ThreadPool::ThreadPool(int)::$_1>(std::__invoke_other, MNN::ThreadPool::ThreadPool(int)::$_1&&) /usr/include/c++/11/bits/invoke.h:61:14
                #13 0x7f521fc281ec in std::__invoke_result<MNN::ThreadPool::ThreadPool(int)::$_1>::type std::__invoke<MNN::ThreadPool::ThreadPool(int)::$_1>(MNN::ThreadPool::ThreadPool(int)::$_1&&) /usr/include/c++/11/bits/invoke.h:96:14
                #14 0x7f521fc281c4 in void std::thread::_Invoker<std::tuple<MNN::ThreadPool::ThreadPool(int)::$_1> >::_M_invoke<0ul>(std::_Index_tuple<0ul>) /usr/include/c++/11/bits/std_thread.h:259:13
                #15 0x7f521fc28194 in std::thread::_Invoker<std::tuple<MNN::ThreadPool::ThreadPool(int)::$_1> >::operator()() /usr/include/c++/11/bits/std_thread.h:266:11
                #16 0x7f521fc280b8 in std::thread::_State_impl<std::thread::_Invoker<std::tuple<MNN::ThreadPool::ThreadPool(int)::$_1> > >::_M_run() /usr/include/c++/11/bits/std_thread.h:211:13
                #17 0x7f521f144252  (/lib/x86_64-linux-gnu/libstdc++.so.6+0xdc252) (BuildId: e37fe1a879783838de78cbc8c80621fa685d58a2)
                #18 0x7f521edccac2 in start_thread nptl/./nptl/pthread_create.c:442:8
                #19 0x7f521ee5da03 in __clone misc/../sysdeps/unix/sysv/linux/x86_64/clone.S:100

            0x62d000008450 is located 8 bytes to the right of 32840-byte region [0x62d000000400,0x62d000008448)
            allocated by thread T0 here:
                #0 0x56018f04708e in malloc (/root/code/mnn_base/MNN_LLM_TUNEING/build/llm_backward+0xc508e) (BuildId: a4a0d3901e26bac8b097d9b695985986f683f48d)
                #1 0x7f521f6236ec in MNNMemoryAllocAlign /root/code/mnn_base/MNN_LLM_TUNEING/source/core/MNNMemoryUtils.cpp:24:30
                #2 0x7f521f5cd819 in MNN::DefaultAllocator::onAlloc(unsigned long, unsigned long) /root/code/mnn_base/MNN_LLM_TUNEING/source/core/BufferAllocator.cpp:59:25
                #3 0x7f521f5c040b in MNN::EagerBufferAllocator::alloc(unsigned long, bool, unsigned long) /root/code/mnn_base/MNN_LLM_TUNEING/source/core/BufferAllocator.cpp:223:30
                #4 0x7f521f9e98f4 in MNN::CPUBackend::allocBuffer(unsigned long, MNN::Tensor*, MNN::Backend::StorageType) /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/CPUBackend.cpp:600:49
                #5 0x7f521fe20df9 in MNN::AVX2Backend::onAcquire(MNN::Tensor const*, MNN::Backend::StorageType) /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/x86_x64/AVX2Backend.cpp:344:16
                #6 0x7f521f5b8e80 in MNN::Backend::onAcquireBuffer(MNN::Tensor const*, MNN::Backend::StorageType) /root/code/mnn_base/MNN_LLM_TUNEING/source/core/Backend.cpp:128:22
                #7 0x7f521fc0ace6 in MNN::KVCacheManager::onAlloc(int) /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/KVCacheManager.cpp:390:19
                #8 0x7f521f9c7f5b in MNN::CPUAttention::onExecute(std::vector<MNN::Tensor*, std::allocator<MNN::Tensor*> > const&, std::vector<MNN::Tensor*, std::allocator<MNN::Tensor*> > const&) /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/CPUAttention.cpp:271:30
                #9 0x7f521f6b71c7 in MNN::Pipeline::execute() /root/code/mnn_base/MNN_LLM_TUNEING/source/core/Pipeline.cpp:1214:40
                #10 0x7f521f71d351 in MNN::Session::run() const /root/code/mnn_base/MNN_LLM_TUNEING/source/core/Session.cpp:243:28
                #11 0x7f52205e6833 in MNN::Express::Executor::ComputeCache::compute() /root/code/mnn_base/MNN_LLM_TUNEING/express/Utils.cpp:273:41
                #12 0x7f522044efd6 in MNN::Express::Variable::readInternal(bool) /root/code/mnn_base/MNN_LLM_TUNEING/express/Expr.cpp:838:28
                #13 0x56018f092bf6 in void const* MNN::Express::Variable::readMap<void>() /root/code/mnn_base/MNN_LLM_TUNEING/include/MNN/expr/Expr.hpp:130:26
                #14 0x56018f08832f in main /root/code/mnn_base/MNN_LLM_TUNEING/transformers/llm/engine/demo/llm_backward.cpp:886:21
                #15 0x7f521ed61d8f in __libc_start_call_main csu/../sysdeps/nptl/libc_start_call_main.h:58:16

            Thread T1 created by T0 here:
                #0 0x56018f03050c in __interceptor_pthread_create (/root/code/mnn_base/MNN_LLM_TUNEING/build/llm_backward+0xae50c) (BuildId: a4a0d3901e26bac8b097d9b695985986f683f48d)
                #1 0x7f521f144328 in std::thread::_M_start_thread(std::unique_ptr<std::thread::_State, std::default_delete<std::thread::_State> >, void (*)()) (/lib/x86_64-linux-gnu/libstdc++.so.6+0xdc328) (BuildId: e37fe1a879783838de78cbc8c80621fa685d58a2)
                #2 0x7f521fc27e10 in void __gnu_cxx::new_allocator<std::thread>::construct<std::thread, MNN::ThreadPool::ThreadPool(int)::$_1>(std::thread*, MNN::ThreadPool::ThreadPool(int)::$_1&&) /usr/include/c++/11/ext/new_allocator.h:162:23
                #3 0x7f521fc2781c in void std::allocator_traits<std::allocator<std::thread> >::construct<std::thread, MNN::ThreadPool::ThreadPool(int)::$_1>(std::allocator<std::thread>&, std::thread*, MNN::ThreadPool::ThreadPool(int)::$_1&&) /usr/include/c++/11/bits/alloc_traits.h:516:8
                #4 0x7f521fc27b1d in void std::vector<std::thread, std::allocator<std::thread> >::_M_realloc_insert<MNN::ThreadPool::ThreadPool(int)::$_1>(__gnu_cxx::__normal_iterator<std::thread*, std::vector<std::thread, std::allocator<std::thread> > >, MNN::ThreadPool::ThreadPool(int)::$_1&&) /usr/include/c++/11/bits/vector.tcc:449:4
                #5 0x7f521fc1f0fe in void std::vector<std::thread, std::allocator<std::thread> >::emplace_back<MNN::ThreadPool::ThreadPool(int)::$_1>(MNN::ThreadPool::ThreadPool(int)::$_1&&) /usr/include/c++/11/bits/vector.tcc:121:4
                #6 0x7f521fc1e843 in MNN::ThreadPool::ThreadPool(int) /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/ThreadPool.cpp:58:18
                #7 0x7f521fc1d3ff in MNN::ThreadPool::init(int, unsigned long, MNN::ThreadPool*&) /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/ThreadPool.cpp:26:35
                #8 0x7f521f9ded57 in MNN::CPURuntime::_resetThreadPool() const /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/CPUBackend.cpp:122:25
                #9 0x7f521f9e1222 in MNN::CPURuntime::onReset(int, MNN::BackendConfig const*, bool) /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/CPUBackend.cpp:221:5
                #10 0x7f52203d8f86 in MNN::Express::Executor::_getOrCreateRuntime(MNNForwardType, MNN::BackendConfig const*, int, bool) /root/code/mnn_base/MNN_LLM_TUNEING/express/Executor.cpp:57:23
                #11 0x7f52203de865 in MNN::Express::Executor::RuntimeManager::createRuntimeManager(MNN::ScheduleConfig const&) /root/code/mnn_base/MNN_LLM_TUNEING/express/Executor.cpp:309:20
                #12 0x7f5220f74bc3 in MNN::Transformer::Llm::initRuntime() /root/code/mnn_base/MNN_LLM_TUNEING/transformers/llm/engine/src/llm.cpp:171:27
                #13 0x7f5220f760f7 in MNN::Transformer::Llm::load() /root/code/mnn_base/MNN_LLM_TUNEING/transformers/llm/engine/src/llm.cpp:219:5
                #14 0x56018f087431 in main /root/code/mnn_base/MNN_LLM_TUNEING/transformers/llm/engine/demo/llm_backward.cpp:846:18
                #15 0x7f521ed61d8f in __libc_start_call_main csu/../sysdeps/nptl/libc_start_call_main.h:58:16

            SUMMARY: AddressSanitizer: heap-buffer-overflow /root/code/mnn_base/MNN_LLM_TUNEING/source/backend/cpu/KVCacheManager.cpp:622:94 in void MNN::KVCacheManager::pack_key<float>(MNN::Tensor const*, int, int)
            Shadow bytes around the buggy address:
            0x0c5a7fff9030: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0x0c5a7fff9040: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0x0c5a7fff9050: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0x0c5a7fff9060: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0x0c5a7fff9070: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            =>0x0c5a7fff9080: 00 00 00 00 00 00 00 00 00 fa[fa]fa fa fa fa fa
            0x0c5a7fff9090: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
            0x0c5a7fff90a0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
            0x0c5a7fff90b0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
            0x0c5a7fff90c0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
            0x0c5a7fff90d0: fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa fa
            Shadow byte legend (one shadow byte represents 8 application bytes):
            Addressable:           00
            Partially addressable: 01 02 03 04 05 06 07 
            Heap left redzone:       fa
            Freed heap region:       fd
            Stack left redzone:      f1
            Stack mid redzone:       f2
            Stack right redzone:     f3
            Stack after return:      f5
            Stack use after scope:   f8
            Global redzone:          f9
            Global init order:       f6
            Poisoned by user:        f7
            Container overflow:      fc
            Array cookie:            ac
            Intra object redzone:    bb
            ASan internal:           fe
            Left alloca redzone:     ca
            Right alloca redzone:    cb
            ==2180924==ABORTING
        */
        // TODO: backward only to "logits__matmul_converted"
        /*
        graph:
            logits
            -> ConvertTensor6467 Concat6482 
            -> logits__matmul_converted --- here...
            -> Reshape6465___tr4logits__matmul_converted
            -> Reshape6465
            -> hidden_states Concat14 
            -> /norm/Cast_output_0 
            -> /Add_119_output_0 logits_index /Constant_19_output_0 Unsqueeze27 Unsqueeze27 
            -> /post_attention_layernorm_23/Cast_output_0 /mlp/down_proj_23/FakeLinear_output_0 
        */
        // TODO: get single block
        
        // profiler forward
        int tok = 16;
        std::vector<int> tokens(prompt_tokens, tok);
        // input_embeds
        auto hidden_states = llm->embedding(tokens);
        int seq_len         = hidden_states->getInfo()->dim[llm->getSeqLenIndex()];
        auto attention_mask = llm->gen_attention_mask(seq_len);
        auto position_ids = llm->gen_position_ids(seq_len);
        Express::VARP logits_index = new_var<int>({-1}, {1});
        hidden_states->readMap<void>();
        attention_mask->readMap<void>();
        position_ids->readMap<void>();
        logits_index->readMap<void>();

        int begin = instance.mCmdParam.begin, end = instance.mCmdParam.end;
        Timer _t;
        // repeat: test forward
        // for (int i = 0; i < instance.mCmdParam.nRepeat + 1; ++i) {
        //     if (i == begin){
        //         _t.reset();
        //     }
        //     if (i <= end && i > begin) {
        //         t.fwdUs.push_back(_t.durationInUs());
        //         _t.reset();
        //     }
        //     if (i == instance.mCmdParam.nRepeat) {
        //         break;
        //     }
        //     auto res = llm->forwardDyn(hidden_states, attention_mask, position_ids, logits_index);
        //     res[0]->readMap<void>();
        // }
        // Cool
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        
        // fwd + bwd
        llm->getModules()[0]->setIsTraining(true);
        // Timer tDiff, tComp;
        for (int i = 0; i < instance.mCmdParam.nRepeat + 1; ++i) {
            if (i == begin){
                _t.reset();
            }
            if (i <= end && i > begin) {
                t.fbwdUs.push_back(_t.durationInUs());
                _t.reset();
            }
            if (i == instance.mCmdParam.nRepeat) {
                break;
            }
            auto res = llm->forwardDyn(hidden_states, attention_mask, position_ids, logits_index);
            res[0]->readMap<void>();
            std::vector<Express::VARP> diff;
            for (auto r:res){
                auto shape_ = r->getInfo()->dim; // Express::_Shape(r); 1 1 151936
                std::vector<int> shape(shape_.begin(), shape_.end());
                diff.emplace_back(Express::_Const(1.0f, shape));
            }
            // backward
            auto grad = MNN::OpGrad::gradCommon(res, diff, {}); 
        }
        
        // Print
        if (printHeader) {
            printer_->fout = outfile;
            printer_->printHeader(runtimeParams, testParams);
            printHeader = false;
        }
        printer_->printPerformance(t);
        
        // Cool
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    fprintf(printer_->fout, "\n");
    if (printer_->fout != stdout) {
        fclose(printer_->fout);
    }
    return 0;
}
