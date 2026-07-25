#ifndef MNN_LLM_ACCEPTANCE_TRACE_HPP
#define MNN_LLM_ACCEPTANCE_TRACE_HPP

#include <chrono>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace MNN {
namespace Transformer {
namespace AcceptanceTrace {

inline bool enabled() {
    static const bool traceEnabled = []() {
        const char* value = std::getenv("MNN_ACCEPTANCE_TRACE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return traceEnabled;
}

inline uint64_t nowMicros() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline std::mutex& outputMutex() {
    static std::mutex mutex;
    return mutex;
}

inline void log(const char* format, ...) {
    if (!enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(outputMutex());
    std::fprintf(stderr, "[MNN_ACCEPTANCE_TRACE] ts_us=%llu ",
                 static_cast<unsigned long long>(nowMicros()));
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

} // namespace AcceptanceTrace
} // namespace Transformer
} // namespace MNN

#endif
