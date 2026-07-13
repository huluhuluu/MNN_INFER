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
    return schedule(blockSize, bs, {});
}

std::shared_ptr<BatchScheduler::Chunk> BatchScheduler::schedule(int blockSize, int bs, const std::set<int>& skipReqIds) {
    if (!hasValidWork()) return nullptr;
    if (blockSize <= 0) blockSize = mBlockSize;

    // generate inputs with FIFO scheduling
    auto task = std::make_shared<Chunk>();
    int scheduledCount = 0;
    for (int i = 0; i < mRequests.size(); ++i) {
        auto& req = mRequests[i];
        if (req->finished || req->history_tokens.empty()) continue;
        if (skipReqIds.find(req->id) != skipReqIds.end()) continue;

        // FIFO: check batch size limit
        if (bs > 0 && scheduledCount >= bs) break;

        // process prefill/decode chunk
        if (req->all_seq_len < (int)req->history_tokens.size()) {
            // Prefill: process remaining prompt tokens
            int remaining = (int)req->history_tokens.size() - req->all_seq_len;
            task->calLen.push_back(std::min(remaining, blockSize));
            task->pos.push_back(req->all_seq_len);
        } else {
            // Decode: generate one token
            task->calLen.push_back(1);
            task->pos.push_back(mConfig->attention_mask() == "glm2" ? req->gen_seq_len : req->all_seq_len);
        }

        // input tokens
        std::vector<int> chunk;
        if (req->all_seq_len < (int)req->history_tokens.size()) {
            // prefill: from all_seq_len position
            const int* start_ = req->history_tokens.data() + req->all_seq_len;
            chunk.assign(start_, start_ + task->calLen.back());
        } else {
            // decode: the last generated token (or last prompt token for first decode)
            chunk.push_back(req->history_tokens.back());
        }
        task->inputs.push_back(chunk);
        
        // chunk info
        task->reqId.push_back(req->id);
        task->culLen += task->calLen.back();
        req->all_seq_len += task->calLen.back();
        scheduledCount++;
    }
    if (scheduledCount == 0) {
        return nullptr;
    }
    return task;
}

// update request status
bool BatchScheduler::update(int req_id, int new_token, int cal_len, bool is_stop_token) {
    return update(req_id, std::vector<int>{new_token}, cal_len, is_stop_token);
}

bool BatchScheduler::update(int req_id, const std::vector<int>& new_tokens, int cal_len, bool is_stop_token) {
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size() || mRequests[ind]->finished || new_tokens.empty()) return false;
    
    auto& req = mRequests[ind];
    
    // prefill phase: all_seq_len already updated in schedule()
    if (judgeState(state(req_id), RequestState::PREFILL)) {
        // nothing to do, all_seq_len already updated
    }
    
    // decode phase: append new token
    if (judgeState(state(req_id), RequestState::DECODE)) {
        int kv_advance = static_cast<int>(new_tokens.size()) - cal_len;
        if (kv_advance > 0) {
            req->all_seq_len += kv_advance;
        }
        req->history_tokens.insert(req->history_tokens.end(), new_tokens.begin(), new_tokens.end());
        req->output_tokens.insert(req->output_tokens.end(), new_tokens.begin(), new_tokens.end());
        req->gen_seq_len += new_tokens.size();
        
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
    if (ind < 0 || ind >= mRequests.size()) return false;
    if (!mRequests[ind]->finished && mActiveCount > 0) {
        mActiveCount--;
    }
    mBatchKVMeta->releaseKV(req_id);
    
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

}
}
