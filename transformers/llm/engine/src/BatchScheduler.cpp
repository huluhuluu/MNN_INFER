//
//  batchScheduler.cpp
//
//  Created by huluhuluu on 2026/01/12
//  JiahuiZhou
//
//

#include "llm/BatchScheduler.hpp"
#include "llm/AcceptanceTrace.hpp"
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
        mReqIdToPipeline.clear();
    }
}

int BatchScheduler::addRequest(const std::vector<int>& prompt) {
    auto request = std::make_shared<Request>(mIdx, prompt);
    request->registeredUs = AcceptanceTrace::nowMicros();
    mRequests.push_back(request);
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
    req->registeredUs = AcceptanceTrace::nowMicros();
    req->firstTokenUs = 0;
    req->completedUs = 0;
    
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
    if (auto pending = _popPendingChunk()) {
        return pending;
    }
    if (!hasValidWork()) return nullptr;
    if (blockSize <= 0) blockSize = mBlockSize;

    // generate inputs with FIFO scheduling
    auto task = std::make_shared<Chunk>();
    struct ScheduledItem {
        std::shared_ptr<Request> req;
        std::vector<int> input;
        int calLen = 0;
        int pos = 0;
        int state = 0;
        bool shouldSplit = false;
    };
    std::vector<ScheduledItem> scheduledItems;
    int scheduledCount = 0;
    for (int i = 0; i < mRequests.size(); ++i) {
        auto& req = mRequests[i];
        if (req->finished || req->history_tokens.empty()) continue;
        if (skipReqIds.find(req->id) != skipReqIds.end()) continue;

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

        ScheduledItem item;
        item.req = req;
        item.input = chunk;
        item.calLen = calLen;
        item.pos = pos;
        item.state = reqState;
        item.shouldSplit = BatchScheduler::judgeState(reqState, RequestState::PREFILL) && calLen > 1;
        scheduledItems.push_back(item);
        scheduledCount++;
    }
    if (scheduledCount == 0) {
        return nullptr;
    }
    if (mDualPipelineMode && !scheduledItems.empty()) {
        bool hasSplitRequest = false;
        for (const auto& item : scheduledItems) {
            hasSplitRequest = hasSplitRequest || item.shouldSplit;
        }
        const int pipelineCount = 2;
        std::vector<std::vector<int>> pipelineItems(pipelineCount);
        std::vector<int> unassignedItems;
        for (int i = 0; i < scheduledItems.size(); ++i) {
            const int reqId = scheduledItems[i].req->id;
            std::map<int, int>::const_iterator iter = mReqIdToPipeline.find(reqId);
            if (iter != mReqIdToPipeline.end() && iter->second >= 0 && iter->second < pipelineCount) {
                pipelineItems[iter->second].push_back(i);
            } else {
                unassignedItems.push_back(i);
            }
        }
        int requestBegin = 0;
        for (int pipeline = 0; pipeline < pipelineCount; ++pipeline) {
            int requestsLeft = static_cast<int>(unassignedItems.size()) - requestBegin;
            int pipelinesLeft = pipelineCount - pipeline;
            int requestCount = requestsLeft > 0 ? (requestsLeft + pipelinesLeft - 1) / pipelinesLeft : 0;
            for (int i = requestBegin; i < requestBegin + requestCount; ++i) {
                const int itemIndex = unassignedItems[i];
                mReqIdToPipeline[scheduledItems[itemIndex].req->id] = pipeline;
                pipelineItems[pipeline].push_back(itemIndex);
            }
            requestBegin += requestCount;
        }

        const int segmentCount = hasSplitRequest ? mDualPipelineSplitCount : 1;
        std::vector<std::shared_ptr<Chunk>> orderedChunks;
        for (int segment = 0; segment < segmentCount; ++segment) {
            for (int pipeline = 0; pipeline < pipelineCount; ++pipeline) {
                auto chunk = std::make_shared<Chunk>();
                chunk->pipelineId = pipeline;
                chunk->segmentIndex = segment;
                for (int itemIndex : pipelineItems[pipeline]) {
                    const auto& item = scheduledItems[itemIndex];
                    if (!item.shouldSplit && segment > 0) {
                        continue;
                    }
                    int splitBegin = 0;
                    int splitLen = item.calLen;
                    if (item.shouldSplit) {
                        splitBegin = item.calLen * segment / mDualPipelineSplitCount;
                        int splitEnd = item.calLen * (segment + 1) / mDualPipelineSplitCount;
                        splitLen = splitEnd - splitBegin;
                    }
                    if (splitLen <= 0) {
                        continue;
                    }
                    std::vector<int> splitInput(item.input.begin() + splitBegin, item.input.begin() + splitBegin + splitLen);
                    _appendToChunk(chunk, item.req, splitInput, splitLen, item.pos + splitBegin, item.state);
                }
                if (chunk->culLen > 0) {
                    orderedChunks.push_back(chunk);
                }
            }
        }
        if (!orderedChunks.empty()) {
            task = orderedChunks.front();
            for (int i = 1; i < orderedChunks.size(); ++i) {
                mPendingChunks.push_back(orderedChunks[i]);
            }
            _commitChunk(task);
            return task;
        }
    }
    for (const auto& item : scheduledItems) {
        _appendToChunk(task, item.req, item.input, item.calLen, item.pos, item.state);
    }
    _commitChunk(task);
    return task;
}

std::vector<std::shared_ptr<BatchScheduler::Chunk>> BatchScheduler::scheduleWave(int blockSize, int bs) {
    return scheduleWave(blockSize, bs, {});
}

std::vector<std::shared_ptr<BatchScheduler::Chunk>> BatchScheduler::scheduleWave(
    int blockSize, int bs, const std::set<int>& skipReqIds) {
    std::vector<std::shared_ptr<Chunk>> wave;
    auto first = schedule(blockSize, bs, skipReqIds);
    if (!first) {
        return wave;
    }
    wave.push_back(first);
    if (!mDualPipelineMode) {
        return wave;
    }
    const int segmentIndex = first->segmentIndex;
    while (!mPendingChunks.empty()) {
        const auto& next = mPendingChunks.front();
        if (!next || next->segmentIndex != segmentIndex) {
            break;
        }
        wave.push_back(_popPendingChunk());
    }
    for (size_t waveIndex = 0; waveIndex < wave.size(); ++waveIndex) {
        const std::shared_ptr<Chunk>& chunk = wave[waveIndex];
        if (!chunk) {
            continue;
        }
        for (size_t requestIndex = 0; requestIndex < chunk->reqId.size(); ++requestIndex) {
            AcceptanceTrace::log("event=lane_owner request_id=%d request_scope=engine lane=%d segment=%d chunk_index=%zu",
                                 chunk->reqId[requestIndex], chunk->pipelineId,
                                 chunk->segmentIndex, waveIndex);
        }
    }
    return wave;
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
        const uint64_t tokenUs = AcceptanceTrace::nowMicros();
        if (req->firstTokenUs == 0) {
            req->firstTokenUs = tokenUs;
        }
        int kv_advance = static_cast<int>(new_tokens.size()) - cal_len;
        if (kv_advance > 0) {
            req->all_seq_len += kv_advance;
        }
        req->history_tokens.insert(req->history_tokens.end(), new_tokens.begin(), new_tokens.end());
        req->output_tokens.insert(req->output_tokens.end(), new_tokens.begin(), new_tokens.end());
        req->gen_seq_len += new_tokens.size();
        
        if (is_stop_token || req->gen_seq_len >= mMaxNewTokens) {
            req->finished = true;
            req->completedUs = AcceptanceTrace::nowMicros();
            mActiveCount--;
            
            // check if has pending prompt for next round
            if (req->has_pending && !req->pending_prompt.empty()) {
                req->history_tokens.insert(req->history_tokens.end(), 
                                           req->pending_prompt.begin(), 
                                           req->pending_prompt.end());
                req->prompt_len = req->pending_prompt.size();
                req->gen_seq_len = 0;
                req->output_tokens.clear();
                req->registeredUs = AcceptanceTrace::nowMicros();
                req->firstTokenUs = 0;
                req->completedUs = 0;
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

size_t BatchScheduler::getResultSize(int req_id) const {
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size()) return 0;
    return mRequests[ind]->output_tokens.size();
}

bool BatchScheduler::getRequestTiming(int req_id, RequestTiming& timing) const {
    int ind = mReqIdToIndex.count(req_id) ? mReqIdToIndex.at(req_id) : -1;
    if (ind < 0 || ind >= mRequests.size()) return false;
    const auto& req = mRequests[ind];
    timing.registeredUs = req->registeredUs;
    timing.firstTokenUs = req->firstTokenUs;
    timing.completedUs = req->completedUs;
    timing.completionTokens = req->output_tokens.size();
    return true;
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
    mReqIdToPipeline.erase(req_id);
    return true;
}

void BatchScheduler::clearPendingChunks() {
    mPendingChunks.clear();
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
