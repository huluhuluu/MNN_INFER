#include "MNN_generated.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace MNN;

static bool rewritePath(std::string& path, const std::string& sourcePrefix, const std::string& targetPrefix) {
    if (path.compare(0, sourcePrefix.size(), sourcePrefix) != 0) {
        return false;
    }
    path = targetPrefix + path.substr(sourcePrefix.size());
    return true;
}

static int rewriteOps(std::vector<std::unique_ptr<OpT>>& ops, const std::string& sourcePrefix,
                      const std::string& targetPrefix) {
    int rewritten = 0;
    for (auto& op : ops) {
        if (op == nullptr) {
            continue;
        }
        rewritten += rewritePath(op->externalPath, sourcePrefix, targetPrefix);
        rewritten += rewritePath(op->name, sourcePrefix, targetPrefix);
        auto plugin = op->main.AsPlugin();
        if (plugin == nullptr) {
            continue;
        }
        for (auto& attribute : plugin->attr) {
            if (attribute != nullptr && attribute->key == "path") {
                rewritten += rewritePath(attribute->s, sourcePrefix, targetPrefix);
            }
        }
    }
    return rewritten;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: rewriteMNNExternalPath input.mnn output.mnn source_prefix target_prefix\n";
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];
    const std::string sourcePrefix = argv[3];
    const std::string targetPrefix = argv[4];
    if (inputPath == outputPath || sourcePrefix.empty()) {
        std::cerr << "Input/output paths must differ and source_prefix must not be empty.\n";
        return 1;
    }

    std::ifstream input(inputPath, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "Cannot open input model: " << inputPath << "\n";
        return 1;
    }
    const auto size = input.tellg();
    if (size <= 0) {
        std::cerr << "Input model is empty: " << inputPath << "\n";
        return 1;
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (!input.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Cannot read input model: " << inputPath << "\n";
        return 1;
    }

    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!VerifyNetBuffer(verifier)) {
        std::cerr << "Invalid MNN FlatBuffer: " << inputPath << "\n";
        return 1;
    }

    std::unique_ptr<NetT> net(GetNet(buffer.data())->UnPack());
    int rewritten = rewriteOps(net->oplists, sourcePrefix, targetPrefix);
    for (auto& subgraph : net->subgraphs) {
        if (subgraph != nullptr) {
            rewritten += rewriteOps(subgraph->nodes, sourcePrefix, targetPrefix);
        }
    }
    if (rewritten == 0) {
        std::cerr << "No model path starts with source_prefix.\n";
        return 2;
    }

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Net::Pack(builder, net.get()));
    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize())) {
        std::cerr << "Cannot write output model: " << outputPath << "\n";
        return 1;
    }

    std::cout << "Rewrote " << rewritten << " external paths.\n";
    return 0;
}
