//
//  batchScheduler.hpp
//  MNN
//
//  Created by huluhuluu on 2026/01/12
//  JiahuiZhou
//

#ifndef BATCHSCHEDULER_hpp
#define BATCHSCHEDULER_hpp

#include <iostream>
#include <numeric>
#include <map>
#include <vector>
#include <memory>

namespace MNN {
namespace Transformer {

class LlmConfig;
struct BatchKVMeta;
class BatchScheduler {
public:
    enum RequestState {
        PREFILL = 1,
        DECODE = 2,
        ERROR = 4,
        FINISH = 8,
        PENDING = 16 // TODO:
    };
    static const bool judgeState(int code, RequestState state) {
        return (code & state) != 0;
    }
    struct Request {
        int id;
        // tokens (aligned with LlmContext)
        std::vector<int> history_tokens;  // all tokens (history + current prompt + generated)
        std::vector<int> output_tokens;   // current generated tokens (for easy result access)
        // length tracking (aligned with LlmContext)
        int prompt_len = 0;    // current prompt length
        int gen_seq_len = 0;   // current generated length
        int all_seq_len = 0;   // KV cache length (already processed)
        // pending prompt for multi-turn chat
        std::vector<int> pending_prompt;
        bool has_pending = false;
        // state
        bool finished = false;
        
        Request(int i, const std::vector<int>& t) : id(i), history_tokens(t), prompt_len(t.size()) {
            gen_seq_len = 0;
            all_seq_len = 0;
            finished = false;
            has_pending = false;
        }
    };

    struct Chunk {
        std::vector<std::vector<int>> inputs; // only read data to make embedding
        std::vector<int> calLen;        // calculated lengths for each input token in the chunk
        std::vector<int> pos;           // position for each input token in the chunk
        std::vector<int> reqId;         // mapping to request index (Vector index, not Global ID)
        int culLen = 0;                 // cumulative length of the chunk
    };

    BatchScheduler() = default;
    explicit BatchScheduler(std::shared_ptr<LlmConfig> config, std::shared_ptr<BatchKVMeta> kvMeta);

    // add request
    int addRequest(const std::vector<int>& prompt);
    std::vector<int> addRequest(const std::vector<std::vector<int>>& prompts);

    // append new prompt to existing request (for multi-turn chat)
    bool appendPrompt(int req_id, const std::vector<int>& new_prompt);

    // schedule logic
    // blockSize: chunk size for prefill, -1 means use default
    // bs: batch size limit for number of requests per schedule, -1 means no limit (FIFO)
    std::shared_ptr<Chunk> schedule(int blockSize = -1, int bs = -1);

    // update generated token
    bool update(int req_id, int new_token, int cal_len, bool is_stop_token);

    // prefill or decode
    int state(int req_id) const;

    bool hasValidWork() const;

    // get results
    std::vector<int> getResult(int req_id) const;
    
    // remove finished request
    bool releaseReq(int req_id);
    bool releaseKVCache(int req_id);
    bool isFinished(int req_id) const;

    void setMaxNewTokens(int max_new_tokens) { mMaxNewTokens = max_new_tokens; }
private:
    std::vector<std::shared_ptr<Request>> mRequests;
    std::map<int, int> mReqIdToIndex; // requestId:vectorIndex
    std::shared_ptr<LlmConfig> mConfig;
    std::shared_ptr<BatchKVMeta> mBatchKVMeta;
    int mBlockSize = 0;
    int mMaxNewTokens = 0;
    int mActiveCount = 0;
    int mIdx = 0; // global request id(increment)
};

} // namespace Transformer
} // namespace MNN

#endif /* BATCHSCHEDULER_hpp */