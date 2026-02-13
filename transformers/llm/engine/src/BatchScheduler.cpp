//
//  batchScheduler.cpp
//
//  Created by huluhuluu on 2026/01/12
//  JiahuiZhou
//
//
//  batchScheduler.cpp
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

// TODO: 1. schedule algorithm to satisfy SLOs
//       2. each data chunk must in mValidBlockSize
//       3. batch_size
// schedule requests by chunk
std::shared_ptr<BatchScheduler::Chunk> BatchScheduler::schedule(int blockSize) {
    if (!hasValidWork()) return nullptr;
    if (blockSize <= 0) blockSize = mBlockSize;

    // generate inputs
    auto task = std::make_shared<Chunk>();
    for (int i = 0; i < mRequests.size(); ++i) {
        auto& req = mRequests[i];
        if (req->finished || req->tokens.empty()) continue;

        // process prefill/decode chunk
        if (req->gen_len == 0) {// Prefill
            task->calLen.push_back(std::min((int)req->tokens.size() - req->proc_len, blockSize));
            task->pos.push_back(req->proc_len);
        } 
        else {// Decode
            task->calLen.push_back(1);
            task->pos.push_back(mConfig->attention_mask() == "glm2" ? req->gen_len : req->proc_len);
        }
        
        // input tokens start pointer
        std::vector<int> chunk;
        const int* start_ = req->tokens.data() + req->proc_len;
        chunk.assign(start_, start_ + task->calLen.back());
        task->inputs.push_back(chunk);
        // chunk info
        task->reqId.push_back(req->id);
        task->culLen += task->calLen.back();
        
        // update proc_len
        req->proc_len += task->calLen.back();
    }
    return task;
}


// update request status
bool BatchScheduler::update(int req_id, int new_token, bool is_stop_token) {
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size() || mRequests[ind]->finished) return false;
    
    // append new token and update status
    auto& req = mRequests[ind];
    if(state(req_id) == RequestState::DECODE){
        // decode phase, append new token
        req->tokens.push_back(new_token), req->gen_len++;
        if (is_stop_token || req->gen_len >= mMaxNewTokens) {
            req->finished = true, mActiveCount--;
        }
    }
    return true;
}


BatchScheduler::RequestState BatchScheduler::state(int req_id) const{
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size()) {
        MNN_PRINT("Request id %d not found in state\n", req_id);
        return RequestState::ERROR; 
    }
    const auto& req = mRequests[ind];
    if (req->finished) return RequestState::FINISH;
    if (req->proc_len == req->tokens.size()) return RequestState::DECODE;
    if (req->proc_len <= req->tokens.size()) return RequestState::PREFILL;
    return RequestState::ERROR;
}

bool BatchScheduler::hasValidWork() const {
    bool nowork = true;
    for (const auto& req : mRequests) {
        if (!req->finished && !req->tokens.empty()) {
            nowork = false;
            break;
        }
    }
    return mActiveCount > 0 && !nowork;
}

std::vector<int> BatchScheduler::getResult(int req_id) const {
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size()) return {};
    return std::vector<int>(mRequests[ind]->tokens.end() - mRequests[ind]->gen_len, mRequests[ind]->tokens.end());
}
    
bool BatchScheduler::releaseReq(int req_id) {
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size() || !mRequests[ind]->finished) return false;
    
    // swap and del last request
    auto& req = mRequests[ind];
    if (ind != mRequests.size() - 1) {
        std::swap(req, mRequests.back());
        mReqIdToIndex[req->id] = ind;
    }
    mRequests.pop_back();
    mReqIdToIndex.erase(req_id);
    
    // release kv meta
    mBatchKVMeta->remove.push_back(req_id);
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