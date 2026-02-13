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
        PREFILL,
        DECODE,
        ERROR,
        FINISH,
        PENDING // TODO:
    };
    struct Request {
        int id;
        std::vector<int> tokens;       // history tokens
        int proc_len = 0;              // processed length
        int gen_len = 0;               // generated length
        bool finished = false;         // request state
        Request(int i, const std::vector<int>& t) : id(i), tokens(t) {
            proc_len = 0; 
            gen_len = 0; 
            finished = false;
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

    // schedule logic
    std::shared_ptr<Chunk> schedule(int blockSize = -1);

    // update generated token
    bool update(int req_id, int new_token, bool is_stop_token);

    // prefill or decode
    RequestState state(int req_id) const;

    bool hasValidWork() const;

    // get results
    std::vector<int> getResult(int req_id) const;
    
    // remove finished request
    bool releaseReq(int req_id);
    bool isFinished(int req_id) const;

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