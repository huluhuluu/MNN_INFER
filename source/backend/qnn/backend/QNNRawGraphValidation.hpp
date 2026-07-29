//
//  QNNRawGraphValidation.hpp
//  MNN
//

#ifndef MNN_QNN_RAW_GRAPH_VALIDATION_HPP
#define MNN_QNN_RAW_GRAPH_VALIDATION_HPP

#include <cstddef>

namespace MNN {
namespace QNN {

inline bool validateRawGraphMetadata(size_t graphCount, size_t graphNameCount) {
    return graphCount > 0 && graphCount == graphNameCount;
}

inline bool validateRawGraphShapeIndex(int shapeIndex, size_t graphCount) {
    return shapeIndex >= 0 && static_cast<size_t>(shapeIndex) < graphCount;
}

class RawGraphAliasOwnership {
public:
    void acquire(bool pinResident) {
        ++mReferences;
        if (pinResident) {
            ++mPinnedReferences;
        }
    }

    bool release(bool unpinAfterRelease) {
        if (mReferences == 0) {
            return true;
        }
        --mReferences;
        if (unpinAfterRelease && mPinnedReferences > 0) {
            --mPinnedReferences;
        }
        return mReferences == 0;
    }

    size_t references() const { return mReferences; }
    bool pinned() const { return mPinnedReferences > 0; }

private:
    size_t mReferences = 0;
    size_t mPinnedReferences = 0;
};

} // namespace QNN
} // namespace MNN

#endif // MNN_QNN_RAW_GRAPH_VALIDATION_HPP
