//
//  dflash.cpp
//

#include "generate.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

DFlashGeneration::DFlashGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) : Generation(llm, context) {
    mBlockSize = config->dflash_block_size();
    mMaskTokenId = config->dflash_mask_token_id();
    mDDTreeTopK = std::max(0, config->dflash_ddtree_topk());
    mDDTreeBudget = std::max(0, config->dflash_ddtree_budget());
}

void DFlashGeneration::load(Module::Config module_config) {
    mDFlashMeta.reset(new KVMeta);
    auto targetLayerIds = mLlm->mConfig->dflash_target_layer_ids();
    mDFlashMeta->layer_nums = static_cast<int>(targetLayerIds.size());
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mDFlashMeta.get());

    std::vector<std::string> inputNames{"noise_embedding", "target_hidden", "attention_mask", "position_ids"};
    std::vector<std::string> outputNames{"logits"};
    mDFlashModule.reset(Module::load(inputNames, outputNames, mLlm->mConfig->dflash_model().c_str(), mLlm->mRuntimeManager, &module_config));
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mLlm->mMeta.get());
    mHiddenStateIndex = mLlm->getOutputIndex("hidden_states");
    if (mHiddenStateIndex < 0) {
        MNN_ERROR("DFlash requires hidden_states output from target model.\n");
    }
    if (mMaskTokenId < 0) {
        MNN_ERROR("DFlash requires dflash_mask_token_id in config.\n");
    }
}

VARP DFlashGeneration::buildAttentionMask(int draftLen, int targetLen) {
    int queryLen = draftLen;
    int kvLen = targetLen + draftLen;
    auto attentionMask = _Input({1, 1, queryLen, kvLen}, NCHW, halide_type_of<float>());
    auto ptr = attentionMask->writeMap<float>();
    for (int i = 0; i < queryLen * kvLen; i++) {
        ptr[i] = 0.0f;
    }
    return attentionMask;
}

VARP DFlashGeneration::buildPositionIds(int start, int len) {
    auto positionIds = _Input({1, len}, NCHW, halide_type_of<int>());
    auto ptr = positionIds->writeMap<int>();
    for (int i = 0; i < len; i++) {
        ptr[i] = start + i;
    }
    return positionIds;
}

DFlashGeneration::DFlashDraftInfo DFlashGeneration::sampleDraftWithLogits(VARP targetHidden, const std::vector<int>& blockTokens) {
    if (mDFlashModule == nullptr || blockTokens.size() <= 1) {
        return {};
    }
    std::vector<int> draftTokensInput(blockTokens.begin(), blockTokens.end());
    int draftLen = static_cast<int>(draftTokensInput.size());
    int targetLen = targetHidden->getInfo()->dim[0];
    auto noiseEmbedding = mLlm->embedding(draftTokensInput);
    auto attentionMask = buildAttentionMask(draftLen, targetLen);
    auto positionIds = buildPositionIds(0, targetLen + draftLen);
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mDFlashMeta.get());
    auto outputs = mDFlashModule->onForward({noiseEmbedding, targetHidden, attentionMask, positionIds});
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mLlm->mMeta.get());
    if (outputs.empty()) {
        return {};
    }
    auto logits = outputs[0];
    int sampleSize = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
    std::vector<int> draftTokens(blockTokens.size() - 1);
    for (int i = 1; i < static_cast<int>(blockTokens.size()); i++) {
        draftTokens[i - 1] = mLlm->sample(logits, i * sampleSize, sampleSize);
    }
    return {std::move(draftTokens), logits};
}

std::vector<int> DFlashGeneration::sampleDraft(VARP targetHidden, const std::vector<int>& blockTokens) {
    return sampleDraftWithLogits(targetHidden, blockTokens).draftTokens;
}

VARP DFlashGeneration::gatherHidden(VARP hidden, const std::vector<int>& indices) {
    int len = static_cast<int>(indices.size());
    auto dims = hidden->getInfo()->dim;
    int hiddenSize = dims[dims.size() - 1];
    auto result = _Input({len, 1, hiddenSize}, NCHW, halide_type_of<float>());
    auto dst = result->writeMap<float>();
    auto src = hidden->readMap<float>();
    for (int i = 0; i < len; i++) {
        ::memcpy(dst + i * hiddenSize, src + indices[i] * hiddenSize, hiddenSize * sizeof(float));
    }
    return result;
}

DFlashGeneration::TreeInfo DFlashGeneration::buildDDTree(int anchorToken, VARP logits, int startPosition, int pastLength) {
    TreeInfo info;
    if (logits == nullptr || mDDTreeBudget <= 0) {
        return info;
    }

    int vocabSize = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
    int rowCount = static_cast<int>(logits->getInfo()->size / vocabSize);
    int maxDepth = std::min(mBlockSize - 1, rowCount - 1);
    if (maxDepth <= 0) {
        return info;
    }
    int topkLimit = mDDTreeTopK > 0 ? std::min(mDDTreeTopK, mDDTreeBudget) : mDDTreeBudget;
    int topk = std::min(topkLimit, vocabSize);
    if (topk <= 0) {
        return info;
    }

    struct TopToken {
        int token = 0;
        float score = 0.0f;
        TopToken() = default;
        TopToken(int token, float score) : token(token), score(score) {}
    };
    std::vector<std::vector<TopToken>> topByDepth(maxDepth + 1);
    auto logitsPtr = logits->readMap<float>();
    for (int depth = 1; depth <= maxDepth; depth++) {
        std::vector<TopToken> row;
        row.reserve(vocabSize);
        auto rowPtr = logitsPtr + depth * vocabSize;
        float maxLogit = rowPtr[0];
        for (int token = 0; token < vocabSize; token++) {
            row.emplace_back(token, rowPtr[token]);
            maxLogit = std::max(maxLogit, rowPtr[token]);
        }
        float logZ = 0.0f;
        for (int token = 0; token < vocabSize; token++) {
            logZ += std::exp(rowPtr[token] - maxLogit);
        }
        logZ = maxLogit + std::log(logZ);
        std::partial_sort(row.begin(), row.begin() + topk, row.end(), [](const TopToken& a, const TopToken& b) {
            return a.score > b.score;
        });
        row.resize(topk);
        for (auto& item : row) {
            item.score -= logZ;
        }
        topByDepth[depth] = std::move(row);
    }

    struct Node {
        int token = 0;
        int parent = -1;
        int depth = 0;
        float score = 0.0f;
        Node(int token, int parent, int depth, float score) : token(token), parent(parent), depth(depth), score(score) {}
    };
    struct Candidate {
        int parent = 0;
        int token = 0;
        int depth = 0;
        int rank = 0;
        float score = 0.0f;
        Candidate(int parent, int token, int depth, int rank, float score) : parent(parent), token(token), depth(depth), rank(rank), score(score) {}
    };
    struct CandidateLess {
        bool operator()(const Candidate& a, const Candidate& b) const {
            return a.score < b.score;
        }
    };

    std::vector<Node> nodes;
    nodes.emplace_back(anchorToken, -1, 0, 0.0f);
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateLess> heap;
    heap.emplace(0, topByDepth[1][0].token, 1, 0, topByDepth[1][0].score);
    int budget = mDDTreeBudget + 1;
    while (!heap.empty() && static_cast<int>(nodes.size()) < budget) {
        auto cand = heap.top();
        heap.pop();
        int nodeIndex = static_cast<int>(nodes.size());
        nodes.emplace_back(cand.token, cand.parent, cand.depth, cand.score);
        if (cand.rank + 1 < static_cast<int>(topByDepth[cand.depth].size())) {
            const auto& sibling = topByDepth[cand.depth][cand.rank + 1];
            float siblingScore = cand.score - topByDepth[cand.depth][cand.rank].score + sibling.score;
            heap.emplace(cand.parent, sibling.token, cand.depth, cand.rank + 1, siblingScore);
        }
        int nextDepth = cand.depth + 1;
        if (nextDepth <= maxDepth) {
            const auto& child = topByDepth[nextDepth][0];
            heap.emplace(nodeIndex, child.token, nextDepth, 0, cand.score + child.score);
        }
    }

    if (nodes.size() <= 1) {
        return info;
    }

    int nodeCount = static_cast<int>(nodes.size());
    info.draftTokens.resize(nodeCount);
    info.childMaps.resize(nodeCount);
    std::vector<int> positionOffsets(nodeCount);
    for (int i = 0; i < nodeCount; i++) {
        info.draftTokens[i] = nodes[i].token;
        positionOffsets[i] = nodes[i].depth;
        if (i > 0) {
            info.childMaps[nodes[i].parent][nodes[i].token] = i;
        }
    }

    std::vector<std::vector<bool>> visible(nodeCount, std::vector<bool>(nodeCount, false));
    for (int i = 0; i < nodeCount; i++) {
        int current = i;
        while (current >= 0) {
            visible[i][current] = true;
            current = nodes[current].parent;
        }
    }
    int kvLen = pastLength + nodeCount;
    info.attentionMask = _Input({1, 1, nodeCount, kvLen}, NCHW, halide_type_of<float>());
    auto maskPtr = info.attentionMask->writeMap<float>();
    for (int i = 0; i < nodeCount; i++) {
        for (int j = 0; j < pastLength; j++) {
            maskPtr[i * kvLen + j] = 0.0f;
        }
        for (int j = 0; j < nodeCount; j++) {
            maskPtr[i * kvLen + pastLength + j] = visible[i][j] ? 0.0f : std::numeric_limits<float>::lowest();
        }
    }

    info.positionIds = _Input({1, nodeCount}, NCHW, halide_type_of<int>());
    auto posPtr = info.positionIds->writeMap<int>();
    for (int i = 0; i < nodeCount; i++) {
        posPtr[i] = startPosition + positionOffsets[i];
    }

    std::vector<std::vector<int>> children(nodeCount);
    for (int i = 1; i < nodeCount; i++) {
        children[nodes[i].parent].push_back(i);
    }
    for (int i = 1; i < nodeCount; i++) {
        if (!children[i].empty()) {
            continue;
        }
        std::vector<int> path;
        int current = i;
        while (current >= 0) {
            path.push_back(current);
            current = nodes[current].parent;
        }
        std::reverse(path.begin(), path.end());
        info.retrieveIndices.push_back(std::move(path));
    }
    return info;
}

DFlashGeneration::AcceptInfo DFlashGeneration::evaluateTreePosterior(const TreeInfo& treeInfo, VARP logits) {
    auto sampleTokens = _ArgMax(logits, -1);
    std::vector<int> samples(treeInfo.draftTokens.size());
    ::memcpy(samples.data(), sampleTokens->readMap<int>(), samples.size() * sizeof(int));
    std::vector<int> bestCandidate;
    if (!treeInfo.childMaps.empty()) {
        bestCandidate.push_back(0);
        int currentIndex = 0;
        while (currentIndex < static_cast<int>(samples.size())) {
            int nextToken = samples[currentIndex];
            auto childIt = treeInfo.childMaps[currentIndex].find(nextToken);
            if (childIt == treeInfo.childMaps[currentIndex].end()) {
                break;
            }
            currentIndex = childIt->second;
            bestCandidate.push_back(currentIndex);
        }
    } else {
        for (const auto& indices : treeInfo.retrieveIndices) {
            std::vector<int> candidate;
            candidate.push_back(indices[0]);
            for (int i = 0; i < static_cast<int>(indices.size()) - 1; i++) {
                int sampleIdx = indices[i];
                int draftIdx = indices[i + 1];
                if (samples[sampleIdx] != treeInfo.draftTokens[draftIdx]) {
                    break;
                }
                candidate.push_back(draftIdx);
            }
            if (candidate.size() > bestCandidate.size()) {
                bestCandidate = std::move(candidate);
            }
        }
    }
    std::vector<int> acceptTokens(bestCandidate.size());
    for (int i = 0; i < static_cast<int>(bestCandidate.size()); i++) {
        acceptTokens[i] = treeInfo.draftTokens[bestCandidate[i]];
    }
    AcceptInfo acceptInfo;
    acceptInfo.sampleTokens = std::move(samples);
    acceptInfo.acceptIndices = std::move(bestCandidate);
    acceptInfo.acceptTokens = std::move(acceptTokens);
    return acceptInfo;
}

bool DFlashGeneration::processAcceptedTokens(const std::vector<int>& tokens, int& len, int maxToken) {
    for (auto token : tokens) {
        if (len >= maxToken) {
            break;
        }
        mContext->history_tokens.push_back(token);
        mContext->output_tokens.push_back(token);
        auto tokenStr = mLlm->tokenizer_decode(token);
        mContext->generate_str += tokenStr;
        if (nullptr != mContext->os) {
            *mContext->os << tokenStr << std::flush;
        }
        len++;
        if (mLlm->is_stop(token)) {
            return true;
        }
    }
    return false;
}

void DFlashGeneration::compactTargetCache(const AcceptInfo& acceptInfo) {
    int acceptLen = static_cast<int>(acceptInfo.acceptTokens.size());
    mLlm->updateContext(acceptLen, acceptLen);
    mLlm->mMeta->remove = acceptInfo.sampleTokens.size();
    mLlm->mMeta->reserveHost.resize(acceptLen * 2);
    mLlm->mMeta->n_reserve = acceptLen;
    mLlm->mMeta->reserve = mLlm->mMeta->reserveHost.data();
    for (int i = 0; i < acceptLen; i++) {
        mLlm->mMeta->reserve[2 * i] = acceptInfo.acceptIndices[i];
        mLlm->mMeta->reserve[2 * i + 1] = 1;
    }
}

VARP DFlashGeneration::lastHidden(VARP hidden) {
    auto dims = hidden->getInfo()->dim;
    int targetLen = dims[0] == 1 && dims.size() > 1 ? dims[1] : dims[0];
    int hiddenSize = dims[dims.size() - 1];
    auto tail = _Input({1, 1, hiddenSize}, NCHW, halide_type_of<float>());
    ::memcpy(tail->writeMap<float>(), hidden->readMap<float>() + (targetLen - 1) * hiddenSize, hiddenSize * sizeof(float));
    return tail;
}

VARP DFlashGeneration::prefixHidden(VARP hidden, int len) {
    auto dims = hidden->getInfo()->dim;
    int seqLen = dims[0] == 1 && dims.size() > 1 ? dims[1] : dims[0];
    int hiddenSize = dims[dims.size() - 1];
    len = std::min(len, seqLen);
    auto prefix = _Input({len, 1, hiddenSize}, NCHW, halide_type_of<float>());
    ::memcpy(prefix->writeMap<float>(), hidden->readMap<float>(), len * hiddenSize * sizeof(float));
    return prefix;
}

void DFlashGeneration::generate(GenerationParams& param) {
    if (mHiddenStateIndex < 0 || mDFlashModule == nullptr || mMaskTokenId < 0) {
        ArGeneration(mLlm, mContext, mLlm->mConfig).generate(param);
        return;
    }

    int maxToken = param.max_new_tokens;
    int len = 0;
    bool draftEmptyWarned = false;
    bool invalidVerifyOutputWarned = false;
    auto targetHidden = prefixHidden(param.outputs[mHiddenStateIndex],
                                     param.outputs[mHiddenStateIndex]->getInfo()->dim[0] == 1 ?
                                     param.outputs[mHiddenStateIndex]->getInfo()->dim[1] :
                                     param.outputs[mHiddenStateIndex]->getInfo()->dim[0]);
    if (mDFlashMeta != nullptr) {
        mDFlashMeta->remove = mDFlashMeta->previous;
        mDFlashMeta->add = 0;
    }
    int sampleToken = mLlm->sample(param.outputs[0], param.validLogitStart, param.validLogitSize);
    mContext->current_token = sampleToken;

    while (len < maxToken) {
        if (mContext->status == LlmStatus::USER_CANCEL) {
            break;
        }

        if (mLlm->is_stop(mContext->current_token)) {
            mContext->history_tokens.push_back(mContext->current_token);
            mContext->output_tokens.push_back(mContext->current_token);
            mLlm->updateContext(0, 1);
            if (nullptr != mContext->os) {
                *mContext->os << mContext->end_with << std::flush;
            }
            break;
        }

        MNN::Timer _t;
        std::vector<int> blockTokens(mBlockSize, mMaskTokenId);
        blockTokens[0] = mContext->current_token;
        MNN::Timer draftTimer;
        auto draftInfo = sampleDraftWithLogits(targetHidden, blockTokens);
        auto draftUs = draftTimer.durationInUs();
        if (draftInfo.draftTokens.empty()) {
            if (!draftEmptyWarned) {
                MNN_PRINT("Warning: DFlash draft returned empty tokens, fallback to autoregressive decoding for this step.\n");
                draftEmptyWarned = true;
            }
            auto outputs = mLlm->forwardVec({mContext->current_token});
            if (outputs.empty()) {
                break;
            }
            mLlm->updateContext(1, 1);
            mContext->history_tokens.push_back(mContext->current_token);
            mContext->output_tokens.push_back(mContext->current_token);
            auto tokenStr = mLlm->tokenizer_decode(mContext->current_token);
            mContext->generate_str += tokenStr;
            if (nullptr != mContext->os) {
                *mContext->os << tokenStr << std::flush;
            }
            mContext->current_token = mLlm->sample(outputs[0]);
            targetHidden = _Concat({targetHidden, lastHidden(outputs[mHiddenStateIndex])}, 0);
            len++;
            mContext->decode_us += _t.durationInUs();
            continue;
        }

        if (mDDTreeTopK > 1) {
            auto treeInfo = buildDDTree(mContext->current_token, draftInfo.logits, mContext->all_seq_len, mContext->all_seq_len);
            if (!treeInfo.draftTokens.empty()) {
                MNN::Timer targetTimer;
                auto inputEmbeds = mLlm->embedding(treeInfo.draftTokens);
                int savedGenSeqLen = mContext->gen_seq_len;
                if (savedGenSeqLen == 0) {
                    mContext->gen_seq_len = 1;
                }
                mLlm->mMeta->add = treeInfo.draftTokens.size();
                auto outputs = mLlm->forwardRaw(inputEmbeds, treeInfo.attentionMask, treeInfo.positionIds);
                mContext->gen_seq_len = savedGenSeqLen;
                auto targetUs = targetTimer.durationInUs();
                if (outputs.size() <= mHiddenStateIndex || outputs[0]->getInfo()->size == 0) {
                    if (!invalidVerifyOutputWarned) {
                        MNN_PRINT("Warning: DFlash DDTree target verification output is invalid, stop DFlash generation.\n");
                        invalidVerifyOutputWarned = true;
                    }
                    break;
                }

                auto acceptInfo = evaluateTreePosterior(treeInfo, outputs[0]);
                int emitLen = std::min(static_cast<int>(acceptInfo.acceptTokens.size()), maxToken - len);
                if (emitLen <= 0) {
                    break;
                }
                if (emitLen < static_cast<int>(acceptInfo.acceptTokens.size())) {
                    acceptInfo.acceptTokens.resize(emitLen);
                    acceptInfo.acceptIndices.resize(emitLen);
                }

                bool stop = processAcceptedTokens(acceptInfo.acceptTokens, len, maxToken);
                compactTargetCache(acceptInfo);
                auto acceptHidden = gatherHidden(outputs[mHiddenStateIndex], acceptInfo.acceptIndices);
                targetHidden = _Concat({targetHidden, acceptHidden}, 0);

                mDFlashContext.steps += 1;
                mDFlashContext.draft += treeInfo.draftTokens.size();
                mDFlashContext.accepted += emitLen;
                mDFlashContext.accept_len_freq[emitLen] += 1;
                mDFlashContext.draft_time_us += draftUs;
                mDFlashContext.target_time_us += targetUs;
                mContext->decode_us += _t.durationInUs();

                int lastAcceptedIndex = acceptInfo.acceptIndices.back();
                mContext->current_token = acceptInfo.sampleTokens[lastAcceptedIndex];
                if (stop) {
                    if (nullptr != mContext->os) {
                        *mContext->os << mContext->end_with << std::flush;
                    }
                    return;
                }
                continue;
            }
        }

        auto draftTokens = std::move(draftInfo.draftTokens);
        std::copy(draftTokens.begin(), draftTokens.end(), blockTokens.begin() + 1);

        MNN::Timer targetTimer;
        int savedGenSeqLen = mContext->gen_seq_len;
        if (savedGenSeqLen == 0) {
            mContext->gen_seq_len = 1;
        }
        auto outputs = mLlm->forwardVec(blockTokens);
        mContext->gen_seq_len = savedGenSeqLen;
        auto targetUs = targetTimer.durationInUs();
        if (outputs.size() <= mHiddenStateIndex || outputs[0]->getInfo()->size == 0) {
            if (!invalidVerifyOutputWarned) {
                MNN_PRINT("Warning: DFlash target verification output is invalid, stop DFlash generation.\n");
                invalidVerifyOutputWarned = true;
            }
            break;
        }

        auto logits = outputs[0];
        int sampleSize = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
        std::vector<int> posterior(blockTokens.size());
        for (int i = 0; i < blockTokens.size(); i++) {
            posterior[i] = mLlm->sample(logits, i * sampleSize, sampleSize);
        }

        int acceptLen = 1;
        while (acceptLen < blockTokens.size() && posterior[acceptLen - 1] == blockTokens[acceptLen]) {
            acceptLen++;
        }
        int emitLen = std::min(acceptLen, maxToken - len);
        bool stop = false;
        for (int i = 0; i < emitLen; i++, len++) {
            int token = blockTokens[i];
            mContext->history_tokens.push_back(token);
            mContext->output_tokens.push_back(token);
            auto tokenStr = mLlm->tokenizer_decode(token);
            mContext->generate_str += tokenStr;
            if (nullptr != mContext->os) {
                *mContext->os << tokenStr << std::flush;
            }
            if (mLlm->is_stop(token)) {
                emitLen = i + 1;
                stop = true;
                break;
            }
        }

        int removeLen = static_cast<int>(blockTokens.size()) - emitLen;
        mLlm->mMeta->remove = removeLen;
        mLlm->updateContext(emitLen, emitLen);
        auto acceptHidden = prefixHidden(outputs[mHiddenStateIndex], emitLen);
        targetHidden = _Concat({targetHidden, acceptHidden}, 0);
        mDFlashContext.steps += 1;
        mDFlashContext.draft += blockTokens.size();
        mDFlashContext.accepted += emitLen;
        mDFlashContext.accept_len_freq[emitLen] += 1;
        mDFlashContext.draft_time_us += draftUs;
        mDFlashContext.target_time_us += targetUs;
        mContext->decode_us += _t.durationInUs();

        if (stop) {
            if (nullptr != mContext->os) {
                *mContext->os << mContext->end_with << std::flush;
            }
            return;
        }
        if (emitLen < acceptLen) {
            break;
        }
        mContext->current_token = posterior[acceptLen - 1];
    }

    if(len >= maxToken) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }
}

} // namespace Transformer
} // namespace MNN
