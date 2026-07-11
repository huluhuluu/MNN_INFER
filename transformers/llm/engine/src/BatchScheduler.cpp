//
//  batchScheduler.cpp
//
//  Created by huluhuluu on 2026/01/12
//  JiahuiZhou
//
//

#include "llm/BatchScheduler.hpp"
#include "llmconfig.hpp"
#include "kvmeta.hpp"
namespace MNN {
namespace Transformer {
BatchScheduler::BatchScheduler(std::shared_ptr<LlmConfig> config, std::shared_ptr<BatchKVMeta> kvMeta) 
    : mConfig(config), mBatchKVMeta(kvMeta) {
    // set config
    if (mConfig->config_.document.HasMember("chunk")) {
        mBlockSize = mConfig->config_.document["chunk"].GetInt();
    }
    // default to no chunking
    if (mBlockSize <= 0) {
        mBlockSize = std::numeric_limits<int>::max(); 
    }
    mMaxNewTokens = config->max_new_tokens();
}

void BatchScheduler::setDualPipelineMode(bool enable, int splitCount) {
    mDualPipelineMode = enable;
    mDualPipelineSplitCount = splitCount > 1 ? splitCount : 2;
    if (!enable) {
        mPendingChunks.clear();
    }
}

int BatchScheduler::addRequest(const std::vector<int>& prompt) {
    mRequests.push_back(std::make_shared<Request>(mIdx, prompt));
    mReqIdToIndex[mIdx++] = mRequests.size() - 1;
    mActiveCount++;
    return mIdx - 1;
}

std::vector<int> BatchScheduler::addRequest(const std::vector<std::vector<int>>& prompts) {
    std::vector<int> req_ids;
    for(const auto& prompt : prompts){
        req_ids.push_back(addRequest(prompt));
    }
    return req_ids;
}

bool BatchScheduler::appendPrompt(int req_id, const std::vector<int>& new_prompt) {
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size()) return false;
    
    auto& req = mRequests[ind];
    
    // if still processing, put into pending queue
    if (!req->finished && req->all_seq_len < (int)req->history_tokens.size()) {
        req->pending_prompt = new_prompt;
        req->has_pending = true;
        return true;
    }
    
    // append new prompt directly
    req->history_tokens.insert(req->history_tokens.end(), new_prompt.begin(), new_prompt.end());
    req->prompt_len = new_prompt.size();
    req->gen_seq_len = 0;
    req->output_tokens.clear();
    
    // reactivate if was finished
    if (req->finished) {
        req->finished = false;
        mActiveCount++;
    }
    return true;
}

// schedule requests by chunk with FIFO policy
// bs: batch size limit, -1 means no limit
std::shared_ptr<BatchScheduler::Chunk> BatchScheduler::schedule(int blockSize, int bs) {
    if (auto pending = _popPendingChunk()) {
        return pending;
    }
    if (!hasValidWork()) return nullptr;
    if (blockSize <= 0) blockSize = mBlockSize;

    // generate inputs with FIFO scheduling
    auto task = std::make_shared<Chunk>();
    std::vector<std::shared_ptr<Chunk>> splitChunks;
    if (mDualPipelineMode) {
        splitChunks.resize(mDualPipelineSplitCount);
        splitChunks[0] = task;
        for (int i = 1; i < mDualPipelineSplitCount; ++i) {
            splitChunks[i].reset(new Chunk);
        }
    }
    int scheduledCount = 0;
    for (int i = 0; i < mRequests.size(); ++i) {
        auto& req = mRequests[i];
        if (req->finished || req->history_tokens.empty()) continue;

        // FIFO: check batch size limit
        if (bs > 0 && scheduledCount >= bs) break;

        int reqState = state(req->id);
        int calLen = 0;
        int pos = 0;
        std::vector<int> chunk;

        // process prefill/decode chunk
        if (req->all_seq_len < (int)req->history_tokens.size()) {
            // Prefill: process remaining prompt tokens
            int remaining = (int)req->history_tokens.size() - req->all_seq_len;
            calLen = std::min(remaining, blockSize);
            pos = req->all_seq_len;
            const int* start_ = req->history_tokens.data() + req->all_seq_len;
            chunk.assign(start_, start_ + calLen);
        } else {
            // Decode: generate one token
            calLen = 1;
            const bool glm2 = mConfig && mConfig->attention_mask() == "glm2";
            pos = glm2 ? req->gen_seq_len : req->all_seq_len;
            chunk.push_back(req->history_tokens.back());
        }

        if (mDualPipelineMode && BatchScheduler::judgeState(reqState, RequestState::PREFILL) && calLen > 1) {
            // Dual-pipeline test mode splits one logical prefill into ordered sub-chunks.
            // Each sub-chunk is committed only when it is returned from schedule().
            int consumed = 0;
            int remaining = calLen;
            for (int split = 0; split < mDualPipelineSplitCount && remaining > 0; ++split) {
                int splitLen = remaining;
                int partsLeft = mDualPipelineSplitCount - split;
                if (partsLeft > 1) {
                    splitLen = (remaining + partsLeft - 1) / partsLeft;
                }
                std::vector<int> splitInput(chunk.begin() + consumed, chunk.begin() + consumed + splitLen);
                _appendToChunk(splitChunks[split], req, splitInput, splitLen, pos + consumed, reqState);
                consumed += splitLen;
                remaining -= splitLen;
            }
        } else {
            _appendToChunk(task, req, chunk, calLen, pos, reqState);
        }
        scheduledCount++;
    }
    for (int i = 1; i < splitChunks.size(); ++i) {
        if (splitChunks[i]->culLen > 0) {
            mPendingChunks.push_back(splitChunks[i]);
        }
    }
    _commitChunk(task);
    return task;
}

// update request status
bool BatchScheduler::update(int req_id, int new_token, int cal_len, bool is_stop_token) {
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size() || mRequests[ind]->finished) return false;
    
    auto& req = mRequests[ind];
    
    // prefill phase: all_seq_len already updated in schedule()
    if (judgeState(state(req_id), RequestState::PREFILL)) {
        // nothing to do, all_seq_len already updated
    }
    
    // decode phase: append new token
    if (judgeState(state(req_id), RequestState::DECODE)) {
        req->history_tokens.push_back(new_token);
        req->output_tokens.push_back(new_token);
        req->gen_seq_len++;
        
        if (is_stop_token || req->gen_seq_len >= mMaxNewTokens) {
            req->finished = true;
            mActiveCount--;
            
            // check if has pending prompt for next round
            if (req->has_pending && !req->pending_prompt.empty()) {
                req->history_tokens.insert(req->history_tokens.end(), 
                                           req->pending_prompt.begin(), 
                                           req->pending_prompt.end());
                req->prompt_len = req->pending_prompt.size();
                req->gen_seq_len = 0;
                req->output_tokens.clear();
                req->pending_prompt.clear();
                req->has_pending = false;
                req->finished = false;
                mActiveCount++;
            }
        }
    }
    return true;
}

int BatchScheduler::state(int req_id) const{
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size()) {
        MNN_PRINT("Request id %d not found in state\n", req_id);
        return RequestState::ERROR; 
    }
    const auto& req = mRequests[ind];
    int code = 0;
    if (req->finished) code |= RequestState::FINISH;
    
    // all_seq_len < history_tokens.size() -> has prompt to process
    if (req->all_seq_len < (int)req->history_tokens.size()) code |= RequestState::PREFILL;
    // all_seq_len >= history_tokens.size() -> can generate
    if (req->all_seq_len >= (int)req->history_tokens.size()) code |= RequestState::DECODE;
    
    return code;
}

bool BatchScheduler::hasValidWork() const {
    for (const auto& req : mRequests) {
        if (!req->finished && !req->history_tokens.empty()) {
            return mActiveCount > 0;
        }
    }
    return false;
}

std::vector<int> BatchScheduler::getResult(int req_id) const {
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size()) return {};
    return mRequests[ind]->output_tokens;
}
    
bool BatchScheduler::releaseReq(int req_id) {
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size() || !mRequests[ind]->finished) return false;
    this->releaseKVCache(req_id);
    
    // swap and del last request
    auto& req = mRequests[ind];
    if (ind != mRequests.size() - 1) {
        std::swap(req, mRequests.back());
        mReqIdToIndex[req->id] = ind;
    }
    mRequests.pop_back();
    mReqIdToIndex.erase(req_id);
    return true;
}

bool BatchScheduler::releaseKVCache(int req_id){
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size() || !mRequests[ind]->finished) return false;

    // release kv meta
    mBatchKVMeta->releaseKV(req_id);
    return true;
}

bool BatchScheduler::isFinished(int req_id) const {
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size()){
        MNN_PRINT("Request id %d not found in isFinished\n", req_id);
        return false;
    }
    return mRequests[ind]->finished;
}

void BatchScheduler::_appendToChunk(const std::shared_ptr<Chunk>& chunk, const std::shared_ptr<Request>& req,
                                    const std::vector<int>& inputs, int calLen, int pos, int state) {
    chunk->inputs.push_back(inputs);
    chunk->calLen.push_back(calLen);
    chunk->pos.push_back(pos);
    chunk->reqId.push_back(req->id);
    chunk->state.push_back(state);
    chunk->culLen += calLen;
}

void BatchScheduler::_commitChunk(const std::shared_ptr<Chunk>& chunk) {
    if (!chunk) {
        return;
    }
    // Keep state advancement tied to the chunk that the caller will execute.
    // This prevents later split chunks from making the request look decoded too early.
    for (int i = 0; i < chunk->reqId.size(); ++i) {
        int ind = mReqIdToIndex.count(chunk->reqId[i]) ? mReqIdToIndex.at(chunk->reqId[i]) : -1;
        if (ind < 0 || ind >= mRequests.size()) {
            continue;
        }
        mRequests[ind]->all_seq_len += chunk->calLen[i];
    }
}

std::shared_ptr<BatchScheduler::Chunk> BatchScheduler::_popPendingChunk() {
    if (mPendingChunks.empty()) {
        return nullptr;
    }
    auto chunk = mPendingChunks.front();
    mPendingChunks.pop_front();
    _commitChunk(chunk);
    return chunk;
}

}
}
