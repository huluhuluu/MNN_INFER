//
//  TokenTreeTest.cpp
//

#include "../MNNTestSuite.h"
#include "../../transformers/llm/engine/src/speculative_decoding/tokentree.hpp"
#include <MNN/MNNDefine.h>

using namespace MNN::Transformer;

class TokenTreeLeafPruningTest : public MNNTestCase {
public:
    bool run(int) override {
        TokenTree tree(2);
        const int roots[] = {1, 2};
        const float rootScores[] = {-0.1f, -10.0f};
        tree.init(roots, rootScores);

        const int children[] = {3, 4, 5, 6};
        const float childScores[] = {-0.1f, -0.2f, -10.0f, -11.0f};
        tree.grow(children, childScores);

        const auto output = tree.finalize(0, 2);
        MNNTEST_ASSERT(output.draftTokens == std::vector<int>({0, 1, 3}));
        MNNTEST_ASSERT(output.positionIds == std::vector<int>({0, 1, 2}));
        MNNTEST_ASSERT(output.retrieveIndices.size() == 1);
        MNNTEST_ASSERT(output.retrieveIndices[0] == std::vector<int>({0, 1, 2}));
        MNNTEST_ASSERT(output.attentionMask.size() == 3);
        MNNTEST_ASSERT(output.attentionMask[2][1]);
        return true;
    }
};

MNNTestSuiteRegister(TokenTreeLeafPruningTest, "llm/token_tree_leaf_pruning");
