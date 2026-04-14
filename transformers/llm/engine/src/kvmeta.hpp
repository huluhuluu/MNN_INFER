//
//  kvmeta.hpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef KVMETA_hpp
#define KVMETA_hpp

#include <vector>
#include <map>
#include <MNN/expr/Expr.hpp>
namespace MNN {
using namespace Express;
namespace Transformer {

struct KVMeta {
    enum {
        NoChange,
        PendingWrite,
        PendingRead
    } file_operation;
    size_t block = 4096;
    size_t previous = 0;
    size_t remove = 0;
    int* reserve = nullptr;
    int n_reserve = 0;
    size_t add = 0;
    std::string file_name = "";
    int file_flag = NoChange;
    int seqlen_in_disk = 0;
    int layer_index = 0;
    int layer_nums = 0;
    std::vector<int> reserveHost;
    void sync();
};

struct BatchKVMeta{
    std::map<int, KVMeta*> mMetas;
    std::vector<int> remove;

    std::vector<int> calId; // calculated request id for next compute
    // sync after remove/add
    void sync();

    // set kv cache info for req_id
    // note: for a batch request, should call this function for each request id sequentially
    void setKVCacheInfo(int req_id, size_t add = 0, size_t remove = 0, int* reserve = nullptr, int n_reserve = 0) {
        if(mMetas.find(req_id) == mMetas.end()) {
            mMetas[req_id] = new KVMeta();
        }
        KVMeta* mMeta = mMetas[req_id];
        if (remove > mMeta->previous) {
            remove = mMeta->previous;
        }

        mMeta->remove = remove;
        mMeta->reserve = reserve;
        mMeta->n_reserve = n_reserve;
        mMeta->add = add;
        if(add > 0){
            calId.push_back(req_id);
        }
    }

    void setKVMetaInfo(int req_id, int layer_nums, int layer_index, int seqlen_in_disk, const std::string& file_name, int file_flag) {
        if(mMetas.find(req_id) == mMetas.end()) {
            mMetas[req_id] = new KVMeta();
        }
        KVMeta* mMeta = mMetas[req_id];
        mMeta->layer_index = layer_index;
        mMeta->layer_nums = layer_nums;
        mMeta->seqlen_in_disk = seqlen_in_disk;
        mMeta->file_name = file_name;
        mMeta->file_flag = file_flag;
    }

    void reset() {

    }

    ~BatchKVMeta() {
        for (auto& kv : mMetas) {
            if (kv.second) {
                delete kv.second;
            }
        }
        mMetas.clear();
        remove.clear();
        calId.clear();
    }
};

}
}
#endif // KVMATE_hpp
