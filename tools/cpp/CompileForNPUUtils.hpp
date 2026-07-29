//
//  CompileForNPUUtils.hpp
//  MNN
//

#ifndef MNN_COMPILE_FOR_NPU_UTILS_HPP
#define MNN_COMPILE_FOR_NPU_UTILS_HPP

#include "MNN_generated.h"

#include <cstdio>
#include <set>
#include <string>
#include <utility>

namespace MNN {
namespace Tools {

inline std::set<int> collectExplicitSkipOutputs(const Net* net,
                                                const std::set<std::string>& skipNames) {
    std::set<int> outputs;
    if (net == nullptr || net->oplists() == nullptr) {
        return outputs;
    }
    for (int i = 0; i < net->oplists()->size(); ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (op->name() == nullptr || skipNames.find(op->name()->str()) == skipNames.end()) {
            continue;
        }
        std::printf("Skip %s op\n", op->name()->c_str());
        if (op->outputIndexes() == nullptr) {
            continue;
        }
        for (int index : *op->outputIndexes()) {
            outputs.insert(index);
        }
    }
    return outputs;
}

inline bool reorderOps(NetT* net) {
    if (net == nullptr) {
        return false;
    }

    auto oplist = std::move(net->oplists);
    std::set<int> validInputs;
    size_t remaining = 0;
    for (const auto& op : oplist) {
        remaining += op != nullptr;
    }

    while (remaining > 0) {
        size_t progressed = 0;
        for (auto& op : oplist) {
            if (op == nullptr) {
                continue;
            }
            bool valid = true;
            for (int index : op->inputIndexes) {
                if (index >= 0 && validInputs.find(index) == validInputs.end()) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                continue;
            }
            for (int index : op->outputIndexes) {
                validInputs.insert(index);
            }
            net->oplists.emplace_back(std::move(op));
            --remaining;
            ++progressed;
        }
        if (progressed > 0) {
            continue;
        }

        std::fprintf(stderr, "compilefornpu: cannot order %zu ops with unresolved inputs\n", remaining);
        for (const auto& op : oplist) {
            if (op == nullptr) {
                continue;
            }
            std::fprintf(stderr, "  op %s:", op->name.c_str());
            for (int index : op->inputIndexes) {
                if (index >= 0 && validInputs.find(index) == validInputs.end()) {
                    std::fprintf(stderr, " %d", index);
                }
            }
            std::fprintf(stderr, "\n");
        }
        return false;
    }
    return true;
}

} // namespace Tools
} // namespace MNN

#endif // MNN_COMPILE_FOR_NPU_UTILS_HPP
