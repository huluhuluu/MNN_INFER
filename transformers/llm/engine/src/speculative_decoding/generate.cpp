//
//  generate.hpp
//
//  Created by MNN on 2025/06/09.
//

#include "generate.hpp"
#include <MNN/AutoTime.hpp>
#include "llm/llm.hpp"
#include "../llmconfig.hpp"
#include "../kvmeta.hpp"
#include "lookahead.hpp"

using namespace MNN::Express;

namespace MNN {
namespace Transformer {

std::shared_ptr<Generation> GenerationStrategyFactory::create(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, bool canSpec) {
    std::shared_ptr<Generation> res;
    if(canSpec) {
        if(config->speculative_type() == "lookahead") {
            res.reset(new LookaheadGeneration(llm, context, config));
        } else if(config->speculative_type() == "mtp") {
            res.reset(new MtpGeneration(llm, context, config));
        } else if(config->speculative_type() == "eagle") {
            res.reset(new EagleGeneration(llm, context, config));
        } else {
            // autoregressive generation
            res.reset(new ArGeneration(llm, context, config));
        }
    } else {
        // autoregressive generation
        res.reset(new ArGeneration(llm, context, config));
    }
    return res;
}

ArGeneration::ArGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) : Generation(llm, context) {
    // do nothing
}

std::vector<std::vector<int>> Generation::generateBatch(const std::vector<std::vector<int>>& inputIds, std::ostream* os, int maxNewTokens) {
    ArGeneration ar(mLlm, mContext, mLlm->mConfig);
    return ar.generateBatch(inputIds, os, maxNewTokens);
}

std::vector<std::vector<int>> ArGeneration::generateBatch(const std::vector<std::vector<int>>& inputIds, std::ostream* os, int maxNewTokens) {
    int bs = inputIds.size();
    std::vector<std::vector<int>> ret(bs, std::vector<int>{});

    mContext->prompt_len = 0;
    mContext->gen_seq_len = 0;
    mContext->all_seq_len = 0;

    std::vector<int> reqIds = mLlm->mScheduler->addRequest(inputIds);
    mLlm->mScheduler->setMaxNewTokens(maxNewTokens > 0 ? maxNewTokens : mLlm->mConfig->max_new_tokens());
    mLlm->applyKVCacheRuntimeHint(mLlm->mRuntimeManager, true);

    while (std::shared_ptr<BatchScheduler::Chunk> chunk =
               mLlm->mScheduler->schedule(-1, BatchScheduler::MAX_BATCH_SIZE)) {
        if (mLlm->cancelRequested()) {
            break;
        }
        const int requiredSize = std::max(chunk->culLen, static_cast<int>(chunk->reqId.size()));
        const int paddedCulLen = mLlm->qnnPaddedCulLen(requiredSize);
        if (paddedCulLen < chunk->culLen) {
            MNN_ERROR("MNN_QNN: no target graph bucket can hold packed length %d.\n", chunk->culLen);
            mContext->status = LlmStatus::INTERNAL_ERROR;
            break;
        }
        Express::VARP hidden_states = mLlm->embedding(chunk->inputs, chunk->calLen, paddedCulLen);
        Express::VARP attention_mask = mLlm->gen_attention_mask(chunk->calLen);
        Express::VARP position_ids = mLlm->gen_position_ids(chunk->pos, chunk->calLen, paddedCulLen);
        Express::VARP logitsIndex = mLlm->logitsAllIdx;
        for(int i = 0; i < chunk->pos.size() ; i++) {
            int req_id = chunk->reqId[i];
            mLlm->mBatchMeta->setKVCacheInfo(req_id, chunk->calLen[i], 0, nullptr, 0);
            mLlm->mBatchMeta->setKVMetaInfo(req_id, mLlm->mConfig->layer_nums(), 0, 0, "", KVMeta::NoChange);
        }
        auto moduleKey = std::make_pair(paddedCulLen, false);
        std::shared_ptr<Module> selectModule = mLlm->mModule;
        if(mLlm->mModulePool.find(moduleKey) == mLlm->mModulePool.end()) {
            mLlm->mModulePool[moduleKey].reset(Module::clone(mLlm->mModule.get()));
        }
        selectModule = mLlm->mModulePool[moduleKey];

        std::vector<Express::VARP> res = selectModule->onForward({hidden_states, attention_mask, position_ids, logitsIndex});
        if (res.empty()) {
            mContext->status = LlmStatus::INTERNAL_ERROR;
            break;
        }
        Express::VARP logits = _Squeeze(res[0], {0});

        int sumLen = 0;
        for (int i = 0; i < chunk->pos.size(); ++i) {
            sumLen += chunk->calLen[i];

            if(chunk->calLen[i] > 1) {
                mLlm->updateContext(chunk->calLen[i], 0);
                mContext->prompt_len += chunk->calLen[i];
            }

            auto state = mLlm->mScheduler->state(chunk->reqId[i]);
            if (!BatchScheduler::judgeState(state, BatchScheduler::RequestState::DECODE)) {
                continue;
            }

            mLlm->updateContext(1, 1);
            Express::VARP logit = MNN::Express::_Gather(logits, _Scalar(sumLen - 1));
            int token  = mLlm->sample(logit);

            int id = chunk->reqId[i];
            mLlm->mScheduler->update(id, token, chunk->calLen[i], mLlm->is_stop(token));
            if(mLlm->mScheduler->isFinished(id)) {
                mLlm->mScheduler->releaseKVCache(id);
            }
        }
        mLlm->mBatchMeta->sync();
    }
    if (mContext->status == LlmStatus::INTERNAL_ERROR ||
        mContext->status == LlmStatus::USER_CANCEL) {
        mLlm->mScheduler->clearPendingChunks();
    }
    for(int id: reqIds){
        const auto result = mLlm->mScheduler->getResult(id);
        for(int j = 0; j < bs; j++) {
            if(reqIds[j] == id) {
                ret[j] = result;
                mLlm->recordBatchRequestMetrics(j, id);
                break;
            }
        }
        if(os!= nullptr){
            *os<<"\n=============================\nReqId: "<<id<<"\n";
            for(int token: result){
                *os<<mLlm->tokenizer_decode(token);
            }
        }
        mLlm->mScheduler->releaseReq(id);
    }
    return ret;
}

void ArGeneration::generate(GenerationParams& param) {
    int max_token = param.max_new_tokens;
    int len = 0;
    while (len < max_token) {
        if (mLlm->cancelRequested()) {
            break;
        }
        AUTOTIME;
        // Update gen seq
        mContext->current_token = mLlm->sample(param.outputs[0], param.validLogitStart, param.validLogitSize);
        mContext->history_tokens.push_back(mContext->current_token);
        mContext->output_tokens.push_back(mContext->current_token);
        mLlm->updateContext(0, 1);
        if (mLlm->is_stop(mContext->current_token)) {
            if (nullptr != mContext->os) {
                *mContext->os << mContext->end_with << std::flush;
            }
            break;
        }
        // Decode and Output
        MNN::Timer _t;
        auto decodeStr = mLlm->tokenizer_decode(mContext->current_token);
        mContext->generate_str += decodeStr;
        if (nullptr != mContext->os) {
            *mContext->os << decodeStr;
            *mContext->os << std::flush;
        }
        // Compute Next Logits
        auto outputs = mLlm->forwardVec({mContext->current_token});
        for (auto o : outputs) {
            if(nullptr == o->readMap<float>()) {
                mContext->status = LlmStatus::INTERNAL_ERROR;
                break;
            }
        }
        if(outputs.empty()) {
            break;
        }
        // Update input seq
        mLlm->updateContext(1, 0);
        mContext->decode_us += _t.durationInUs();
        len++;
    }
    if(len >= max_token) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
    }
}

int Generation::draftVerify(VARP logits, const std::vector<int> &drafts, bool& stop) {
    // verify draft token whether be accepted
    int i_dft = 1;
    {
        //AUTOTIME;
        for(; i_dft < drafts.size(); i_dft++) {
            auto sample_size = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
            auto sample_offset = logits->getInfo()->size - (drafts.size() - i_dft + 1) * sample_size;

            auto predict = mLlm->sample(logits, sample_offset, sample_size);

            // stop token just break the process
            if (mLlm->is_stop(predict)) {
                mContext->current_token = predict;
                if (nullptr != mContext->os) {
                    *mContext->os << mContext->end_with << std::flush;
                }
                stop = true;
                break;
            }
            // draft token id not match
            if(predict != drafts[i_dft]) {
                mContext->current_token = predict;
                break;
            }

            if (nullptr != mContext->os) {
                *mContext->os << mLlm->tokenizer_decode(predict);
                *mContext->os << std::flush;
            }
        }
        // all drafts are corrcet!
        if(i_dft == drafts.size()) {
            auto sample_size = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
            auto sample_offset = logits->getInfo()->size -  sample_size;

            auto predict = mLlm->sample(logits, sample_offset, sample_size);
            mContext->current_token = predict;
        }
    }

    return i_dft;
}



} // namespace Transformer
} // namespace MNN
