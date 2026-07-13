//
//  QNNPrefetch.hpp
//  MNN
//

#ifndef MNN_QNNPREFETCH_HPP
#define MNN_QNNPREFETCH_HPP

#include <stdint.h>
#include <MNN/MNNDefine.h>
#include <string>
#include <vector>

namespace MNN {
namespace QNN {

struct RawGraphPrefetchConfig {
    std::string graphId;
    std::string path;
    uint64_t offset = 0;
    uint64_t size = 0;
    std::vector<std::string> allGraphName;
    bool pinResident = false;
};

MNN_PUBLIC bool preloadRawGraph(const RawGraphPrefetchConfig& config);
MNN_PUBLIC void releaseRawGraph(const std::string& graphId, bool forceRelease = false, bool unpinAfterRelease = false);
MNN_PUBLIC void releaseAllRawGraphs();

} // namespace QNN
} // namespace MNN

#endif // MNN_QNNPREFETCH_HPP
