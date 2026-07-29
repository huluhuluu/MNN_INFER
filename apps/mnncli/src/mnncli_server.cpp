//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "mnncli_server.hpp"
#include "log_utils.hpp"
#include "llm/AcceptanceTrace.hpp"
#include "llm/BatchScheduler.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <limits>
#include <thread>

namespace mnncli {

class RequestCoordinator {
public:
    struct Task {
        enum class ChunkState {
            Ready,
            Pending,
            Finished,
        };

        struct TimingSnapshot {
            int64_t queueUs = -1;
            int64_t modelTtftUs = -1;
            int64_t modelTpotUs = -1;
            int64_t modelLatencyUs = -1;
            int64_t e2eTtftUs = -1;
            int64_t e2eLatencyUs = -1;
        };

        Task(uint64_t id, MNN::Transformer::ChatMessages request_prompts, int request_max_tokens)
            : request_id(id), prompts(std::move(request_prompts)), max_tokens(request_max_tokens),
              created_us(MNN::Transformer::AcceptanceTrace::nowMicros()) {}

        bool wait() {
            std::unique_lock<std::mutex> lock(mutex);
            completed.wait(lock, [this] { return finished || cancelled.load(); });
            return !cancelled.load() && !failed;
        }

        void markCancelled() {
            cancelled.store(true);
            completed.notify_all();
        }

        bool is_cancelled() const {
            return cancelled.load();
        }

        bool is_finished() const {
            std::lock_guard<std::mutex> lock(mutex);
            return finished;
        }

        std::string answer() const {
            std::lock_guard<std::mutex> lock(mutex);
            return response;
        }

        size_t prompt_tokens() const {
            std::lock_guard<std::mutex> lock(mutex);
            return promptTokens;
        }

        size_t generated_tokens() const {
            std::lock_guard<std::mutex> lock(mutex);
            return generatedTokens;
        }

        TimingSnapshot timing() const {
            std::lock_guard<std::mutex> lock(mutex);
            TimingSnapshot timing;
            if (startedUs >= created_us) {
                timing.queueUs = static_cast<int64_t>(startedUs - created_us);
            }
            if (modelMetrics.valid) {
                timing.modelTtftUs = modelMetrics.model_ttft_us;
                timing.modelTpotUs = modelMetrics.model_tpot_us;
                timing.modelLatencyUs = modelMetrics.model_latency_us;
                if (timing.queueUs >= 0 && timing.modelTtftUs >= 0) {
                    timing.e2eTtftUs = timing.queueUs + timing.modelTtftUs;
                }
            }
            if (completedUs >= created_us) {
                timing.e2eLatencyUs = static_cast<int64_t>(completedUs - created_us);
            }
            return timing;
        }

        ChunkState next_chunk(std::string& chunk) {
            std::unique_lock<std::mutex> lock(mutex);
            completed.wait_for(lock, std::chrono::milliseconds(100),
                               [this] { return !chunks.empty() || finished || cancelled.load(); });
            if (cancelled.load() || (finished && chunks.empty())) {
                return ChunkState::Finished;
            }
            if (chunks.empty()) {
                return ChunkState::Pending;
            }
            chunk = std::move(chunks.front());
            chunks.pop_front();
            return ChunkState::Ready;
        }

        const uint64_t request_id;
        const MNN::Transformer::ChatMessages prompts;
        const int max_tokens;
        const uint64_t created_us;

    private:
        friend class RequestCoordinator;

        void append(const std::string& text) {
            std::lock_guard<std::mutex> lock(mutex);
            if (cancelled.load()) {
                return;
            }
            if (firstChunkUs == 0) {
                firstChunkUs = MNN::Transformer::AcceptanceTrace::nowMicros();
            }
            response += text;
            chunks.push_back(text);
            completed.notify_all();
        }

        void markStarted(size_t promptTokenCount) {
            started.store(true);
            std::lock_guard<std::mutex> lock(mutex);
            if (startedUs == 0) {
                startedUs = MNN::Transformer::AcceptanceTrace::nowMicros();
            }
            promptTokens = promptTokenCount;
        }

        void setGeneratedTokens(size_t tokenCount) {
            std::lock_guard<std::mutex> lock(mutex);
            generatedTokens = tokenCount;
        }

        void setModelMetrics(const MNN::Transformer::LlmBatchRequestMetrics& metrics) {
            std::lock_guard<std::mutex> lock(mutex);
            modelMetrics = metrics;
        }

        mutable std::mutex mutex;
        std::condition_variable completed;
        std::atomic<bool> cancelled{false};
        std::atomic<bool> started{false};
        bool finished = false;
        bool failed = false;
        uint64_t startedUs = 0;
        uint64_t completedUs = 0;
        uint64_t firstChunkUs = 0;
        size_t promptTokens = 0;
        size_t generatedTokens = 0;
        std::string response;
        std::deque<std::string> chunks;
        MNN::Transformer::LlmBatchRequestMetrics modelMetrics;
    };

    RequestCoordinator(MNN::Transformer::Llm* llm, const std::string& scheduler_mode)
        : mLlm(llm), mMode(parseMode(scheduler_mode)), mWorker(&RequestCoordinator::workerLoop, this) {}

    ~RequestCoordinator() {
        stop();
    }

    std::shared_ptr<Task> submit(MNN::Transformer::ChatMessages prompts, int max_tokens) {
        auto task = std::make_shared<Task>(mNextRequestId.fetch_add(1), std::move(prompts), max_tokens);
        size_t pendingCount = 0;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mPending.push_back(task);
            pendingCount = mPending.size();
        }
        MNN::Transformer::AcceptanceTrace::log("event=request_enqueued request_id=%llu request_scope=service mode=%s pending=%zu",
                                               static_cast<unsigned long long>(task->request_id),
                                               modeName(), pendingCount);
        mReady.notify_one();
        return task;
    }

    void cancel(const std::shared_ptr<Task>& task) {
        if (!task) {
            return;
        }
        const bool alreadyFinished = task->is_finished();
        task->markCancelled();
        bool removedFromQueue = false;
        bool cancelModel = false;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            for (std::deque<std::shared_ptr<Task>>::iterator iter = mPending.begin(); iter != mPending.end(); ++iter) {
                if (*iter == task) {
                    mPending.erase(iter);
                    removedFromQueue = true;
                    break;
                }
            }
            if (!removedFromQueue && mExecuting && !mExecutingTasks.empty()) {
                cancelModel = std::all_of(mExecutingTasks.begin(), mExecutingTasks.end(),
                                          [](const std::shared_ptr<Task>& activeTask) {
                                              return activeTask->is_cancelled();
                                          });
            }
        }
        if (removedFromQueue) {
            complete(task, false);
        }
        if (cancelModel) {
            mLlm->requestCancel();
        }
        MNN::Transformer::AcceptanceTrace::log("event=request_cancelled request_id=%llu request_scope=service state=%s",
                                               static_cast<unsigned long long>(task->request_id),
                                               removedFromQueue ? "queued" :
                                               (alreadyFinished ? "completed" :
                                               (task->started.load() ? "executing" : "dequeued")));
        mReady.notify_all();
        mIdle.notify_all();
    }

    void reset() {
        std::unique_lock<std::mutex> lock(mMutex);
        while (!mPending.empty()) {
            auto task = mPending.front();
            mPending.pop_front();
            task->markCancelled();
            complete(task, false);
        }
        mIdle.wait(lock, [this] { return !mExecuting; });
        mLlm->reset();
    }

private:
    enum class Mode {
        SingleRequest,
        ContinuousBatch,
        DualPipeline,
    };

    static Mode parseMode(const std::string& scheduler_mode) {
        if (scheduler_mode == "continuous_batch") {
            return Mode::ContinuousBatch;
        }
        if (scheduler_mode == "dual_pipeline") {
            return Mode::DualPipeline;
        }
        return Mode::SingleRequest;
    }

    static constexpr size_t kMaxBatchSize = MNN::Transformer::BatchScheduler::MAX_BATCH_SIZE;
    static constexpr std::chrono::milliseconds kBatchCollectWindow{100};

    const char* modeName() const {
        switch (mMode) {
            case Mode::ContinuousBatch:
                return "continuous_batch";
            case Mode::DualPipeline:
                return "dual_pipeline";
            case Mode::SingleRequest:
            default:
                return "single_request";
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mStopping) {
                return;
            }
            mStopping = true;
            while (!mPending.empty()) {
                auto task = mPending.front();
                mPending.pop_front();
                task->markCancelled();
                complete(task, false);
            }
            if (mExecuting) {
                for (const auto& task : mExecutingTasks) {
                    task->markCancelled();
                }
                mLlm->requestCancel();
            }
        }
        mReady.notify_all();
        if (mWorker.joinable()) {
            mWorker.join();
        }
    }

    void complete(const std::shared_ptr<Task>& task, bool failed) {
        uint64_t elapsedUs = 0;
        int64_t ttftUs = -1;
        size_t promptTokens = 0;
        size_t generatedTokens = 0;
        size_t responseBytes = 0;
        {
            std::lock_guard<std::mutex> lock(task->mutex);
            if (task->finished) {
                return;
            }
            task->failed = failed;
            task->finished = true;
            task->completedUs = MNN::Transformer::AcceptanceTrace::nowMicros();
            elapsedUs = MNN::Transformer::AcceptanceTrace::nowMicros() - task->created_us;
            if (task->firstChunkUs >= task->created_us) {
                ttftUs = static_cast<int64_t>(task->firstChunkUs - task->created_us);
            }
            promptTokens = task->promptTokens;
            generatedTokens = task->generatedTokens;
            responseBytes = task->response.size();
        }
        const double tokensPerSecond = elapsedUs > 0 ?
            static_cast<double>(generatedTokens) * 1000000.0 / static_cast<double>(elapsedUs) : 0.0;
        const auto timing = task->timing();
        MNN::Transformer::AcceptanceTrace::log("event=request_completed request_id=%llu request_scope=service status=%s ttft_us=%lld elapsed_us=%llu prompt_tokens=%zu generated_tokens=%zu tokens_per_s=%.3f response_bytes=%zu model_ttft_us=%lld model_tpot_us=%lld model_latency_us=%lld queue_us=%lld",
                                               static_cast<unsigned long long>(task->request_id),
                                               task->is_cancelled() ? "cancelled" : (failed ? "failed" : "ok"),
                                               static_cast<long long>(ttftUs),
                                               static_cast<unsigned long long>(elapsedUs),
                                               promptTokens, generatedTokens, tokensPerSecond, responseBytes,
                                               static_cast<long long>(timing.modelTtftUs),
                                               static_cast<long long>(timing.modelTpotUs),
                                               static_cast<long long>(timing.modelLatencyUs),
                                               static_cast<long long>(timing.queueUs));
        task->completed.notify_all();
    }

    void runSingle(const std::shared_ptr<Task>& task) {
        if (task->is_cancelled()) {
            complete(task, false);
            return;
        }
        try {
            const auto prompt = mLlm->apply_chat_template(task->prompts);
            const auto inputIds = mLlm->tokenizer_encode(prompt);
            task->markStarted(inputIds.size());
            MNN::Transformer::AcceptanceTrace::log("event=request_started request_id=%llu request_scope=service mode=single_request",
                                                   static_cast<unsigned long long>(task->request_id));
            Utf8StreamProcessor processor([task](const std::string& text) {
                if (text.find("<eop>") == std::string::npos) {
                    task->append(text);
                }
            });
            LlmStreamBuffer streamBuffer([this, task, &processor](const char* data, size_t len) {
                if (task->is_cancelled()) {
                    mLlm->requestCancel();
                    return;
                }
                processor.processStream(data, len);
            });
            std::ostream output(&streamBuffer);
            mLlm->generate_init(&output, "<eop>");
            if (task->is_cancelled()) {
                mLlm->requestCancel();
            }
            mLlm->generate(inputIds, task->max_tokens);
            const auto context = mLlm->getContext();
            if (context != nullptr) {
                {
                    std::lock_guard<std::mutex> lock(task->mutex);
                    task->promptTokens = std::max(0, context->prompt_len);
                }
                task->setGeneratedTokens(static_cast<size_t>(std::max(0, context->gen_seq_len)));
            }
            complete(task, context != nullptr && context->status == MNN::Transformer::LlmStatus::INTERNAL_ERROR);
        } catch (const std::exception& error) {
            LOG_DEBUG("LLM request failed: " + std::string(error.what()));
            complete(task, true);
        }
    }

    void runBatch(const std::vector<std::shared_ptr<Task>>& tasks) {
        std::vector<std::shared_ptr<Task>> active_tasks;
        std::vector<std::vector<int>> input_ids;
        active_tasks.reserve(tasks.size());
        input_ids.reserve(tasks.size());
        for (const auto& task : tasks) {
            if (task->is_cancelled()) {
                complete(task, false);
                continue;
            }
            const auto prompt = mLlm->apply_chat_template(task->prompts);
            input_ids.push_back(mLlm->tokenizer_encode(prompt));
            task->markStarted(input_ids.back().size());
            active_tasks.push_back(task);
        }
        if (active_tasks.empty()) {
            return;
        }

        MNN::Transformer::AcceptanceTrace::log("event=batch_started mode=%s request_count=%zu delivery=batch_completion",
                                               modeName(), active_tasks.size());
        for (size_t taskIndex = 0; taskIndex < active_tasks.size(); ++taskIndex) {
            MNN::Transformer::AcceptanceTrace::log("event=batch_member request_id=%llu request_scope=service batch_index=%zu",
                                                   static_cast<unsigned long long>(active_tasks[taskIndex]->request_id),
                                                   taskIndex);
        }

        std::vector<std::vector<int>> results;
        try {
            mLlm->generate_init(nullptr, nullptr);
            if (std::all_of(active_tasks.begin(), active_tasks.end(),
                            [](const std::shared_ptr<Task>& task) { return task->is_cancelled(); })) {
                for (const auto& task : active_tasks) {
                    complete(task, false);
                }
                return;
            }
            results = mLlm->generate(input_ids, nullptr, active_tasks.front()->max_tokens);
        } catch (const std::exception& error) {
            LOG_DEBUG("LLM batch request failed: " + std::string(error.what()));
            MNN::Transformer::AcceptanceTrace::log(
                "event=batch_failed mode=%s reason=exception request_count=%zu message=%s",
                modeName(), active_tasks.size(), error.what());
            for (const auto& task : active_tasks) {
                complete(task, true);
            }
            return;
        }
        if (results.size() != active_tasks.size()) {
            LOG_DEBUG("LLM batch request returned an unexpected result count");
            MNN::Transformer::AcceptanceTrace::log(
                "event=batch_failed mode=%s reason=result_count request_count=%zu result_count=%zu",
                modeName(), active_tasks.size(), results.size());
            for (const auto& task : active_tasks) {
                complete(task, true);
            }
            return;
        }
        const auto context = mLlm->getContext();
        if (context != nullptr && context->status == MNN::Transformer::LlmStatus::INTERNAL_ERROR) {
            LOG_DEBUG("LLM batch request ended with INTERNAL_ERROR");
            MNN::Transformer::AcceptanceTrace::log(
                "event=batch_failed mode=%s reason=internal_error request_count=%zu",
                modeName(), active_tasks.size());
            for (const auto& task : active_tasks) {
                complete(task, true);
            }
            return;
        }
        const auto& batchMetrics = mLlm->getLastBatchRequestMetrics();
        for (size_t i = 0; i < active_tasks.size(); ++i) {
            active_tasks[i]->setGeneratedTokens(results[i].size());
            if (i < batchMetrics.size()) {
                active_tasks[i]->setModelMetrics(batchMetrics[i]);
            }
            if (!active_tasks[i]->is_cancelled()) {
                std::string decoded;
                Utf8StreamProcessor processor([&decoded](const std::string& text) {
                    decoded += text;
                });
                for (int token : results[i]) {
                    const std::string tokenText = mLlm->tokenizer_decode(token);
                    processor.processStream(tokenText.data(), tokenText.size());
                }
                if (!decoded.empty()) {
                    active_tasks[i]->append(decoded);
                }
            }
            complete(active_tasks[i], false);
        }
    }

    void workerLoop() {
        while (true) {
            std::vector<std::shared_ptr<Task>> batch;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mReady.wait(lock, [this] { return mStopping || !mPending.empty(); });
                while (!mPending.empty() && mPending.front()->is_cancelled()) {
                    auto task = mPending.front();
                    mPending.pop_front();
                    complete(task, false);
                }
                if (mStopping && mPending.empty()) {
                    return;
                }
                if (mPending.empty()) {
                    continue;
                }
                batch.push_back(mPending.front());
                mPending.pop_front();
                if (mMode != Mode::SingleRequest) {
                    mReady.wait_for(lock, kBatchCollectWindow, [this, &batch] {
                        return mStopping || batch.size() + mPending.size() >= kMaxBatchSize;
                    });
                    const int max_tokens = batch.front()->max_tokens;
                    while (!mPending.empty() && batch.size() < kMaxBatchSize &&
                           mPending.front()->max_tokens == max_tokens) {
                        batch.push_back(mPending.front());
                        mPending.pop_front();
                    }
                }
                for (size_t taskIndex = 0; taskIndex < batch.size(); ++taskIndex) {
                    batch[taskIndex]->started.store(true);
                }
                mExecuting = true;
                mExecutingTasks = batch;
            }

            if (mMode == Mode::SingleRequest) {
                runSingle(batch.front());
            } else {
                runBatch(batch);
            }

            {
                std::lock_guard<std::mutex> lock(mMutex);
                mExecuting = false;
                mExecutingTasks.clear();
            }
            mIdle.notify_all();
        }
    }

    MNN::Transformer::Llm* mLlm;
    const Mode mMode;
    std::atomic<uint64_t> mNextRequestId{1};
    std::mutex mMutex;
    std::condition_variable mReady;
    std::condition_variable mIdle;
    std::deque<std::shared_ptr<Task>> mPending;
    std::vector<std::shared_ptr<Task>> mExecutingTasks;
    bool mExecuting = false;
    bool mStopping = false;
    std::thread mWorker;
};

std::string GetCurrentTimeAsString() {
  // Get the current time since epoch
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

  // Convert to string
  return std::to_string(seconds);
}

bool FromJson(const json& j, PromptItem& item) {
  if (!j.is_object()) {
    return false;
  }
  if (!j.contains("role") || !j["role"].is_string()) {
    return false;
  }
  if (!j.contains("content") || !j["content"].is_string()) {
    return false;
  }

  item.first = j["role"].get<std::string>();   // Role
  item.second = j["content"].get<std::string>(); // Content
  return true;
}

std::string trimLeadingWhitespace(const std::string& str) {
    auto it = std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch); // Find the first non-whitespace character
    });
    return std::string(it, str.end()); // Create a substring from the first non-whitespace character
}

    const std::string getR1AssistantString(std::string assistant_content) {
    std::size_t pos = assistant_content.find("</think>");
    if (pos != std::string::npos) {
        assistant_content.erase(0, pos + std::string("</think>").length());
    }
    return trimLeadingWhitespace(assistant_content) + "<|end_of_sentence|>";
}

std::string GetR1UserString(std::string user_content, bool last) {
    return "<|User|>" + std::string(user_content) + "<|Assistant|>";
}

    std::vector<PromptItem> ConvertToR1(std::vector<PromptItem> chat_prompts) {
    std::vector<PromptItem> result_prompts = {};
    std::string prompt_result = "";
    result_prompts.emplace_back("system", "<|begin_of_sentence|>You are a helpful assistant.");
    auto iter = chat_prompts.begin();
    for (; iter != chat_prompts.end() - 1; ++iter) {
        if (iter->first == "system") {
            continue;
        } else if (iter->first == "assistant") {
            result_prompts.emplace_back("assistant", getR1AssistantString(iter->second));
        } else if (iter->first == "user") {
            result_prompts.emplace_back("user", GetR1UserString(iter->second, false));
        }
    }
    if (iter->first == "user") {
        result_prompts.emplace_back("user", GetR1UserString(iter->second, true));
    } else {
        result_prompts.emplace_back("assistant", getR1AssistantString(iter->second));
    }
    return result_prompts;
}

MnncliServer::MnncliServer() = default;
MnncliServer::~MnncliServer() = default;

void AllowCors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Methods",  "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers",  "Content-Type, Authorization");
}

bool MnncliServer::Start(MNN::Transformer::Llm* llm, bool is_r1, const std::string& host, int port,
                         const std::string& scheduler_mode) {
    this->is_r1_ = is_r1;
    coordinator_.reset(new RequestCoordinator(llm, scheduler_mode));
    // Create a server instance
    httplib::Server server;

    // Define a route for the GET request on "/"
    server.Get("/", [this](const httplib::Request& req, httplib::Response& res) {
        AllowCors(res);
        res.set_content(html_content, "text/html");
    });
    server.Post("/reset", [&](const httplib::Request &req, httplib::Response &res) {
      LOG_DEBUG("POST /reset");
      AllowCors(res);
      coordinator_->reset();
      res.set_content("{\"status\": \"ok\"}", "application/json");
    });
    
    server.Get("/v1/models", [&](const httplib::Request &req, httplib::Response &res) {
      LOG_DEBUG("GET /v1/models");
      AllowCors(res);
      json models_response = {
        {"object", "list"},
        {"data", json::array({
          {
            {"id", "ModelScope/MNN/Qwen2.5-0.5B-Instruct"},
            {"object", "model"},
            {"created", static_cast<int>(time(nullptr))},
            {"owned_by", "mnn"}
          }
        })}
      };
      res.set_content(models_response.dump(), "application/json");
    });
    server.Options("/v1/models", [](const httplib::Request& /*req*/, httplib::Response& res) {
        AllowCors(res);
        res.status = 200;
    });
    
    server.Options("/chat/completions", [](const httplib::Request& /*req*/, httplib::Response& res) {
        AllowCors(res);
        res.status = 200;
    });
    
    server.Options("/v1/chat/completions", [](const httplib::Request& /*req*/, httplib::Response& res) {
        AllowCors(res);
        res.status = 200;
    });
    // Handler function for chat completions
    auto chatCompletionsHandler = [&](const httplib::Request &req, httplib::Response &res) {
        LOG_DEBUG("POST chat/completions, handled by thread: " + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())));
        AllowCors(res);
        if (!json::accept(req.body)) {
            res.status = 400;
            res.set_content(json{{"error", "Invalid JSON in request body."}}.dump(), "application/json");
            return;
        }
        json request_json = json::parse(req.body, nullptr, false);
        if (!request_json.is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "Request body must be a JSON object."}}.dump(), "application/json");
            return;
        }
        if (!request_json.contains("messages") || !request_json["messages"].is_array()) {
            res.status = 400;
            res.set_content(json{{"error", "messages must be a non-empty array."}}.dump(), "application/json");
            return;
        }
        MNN::Transformer::ChatMessages prompts;
        for (const auto& item_json : request_json["messages"]) {
            PromptItem item;
            if (!FromJson(item_json, item)) {
                res.status = 400;
                res.set_content(json{{"error", "Each message needs string role and content fields."}}.dump(), "application/json");
                return;
            }
            prompts.push_back(std::move(item));
        }
        if (prompts.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "messages must be a non-empty array."}}.dump(), "application/json");
            return;
        }
        if (request_json.contains("model") && !request_json["model"].is_string()) {
            res.status = 400;
            res.set_content(json{{"error", "model must be a string."}}.dump(), "application/json");
            return;
        }
        if (request_json.contains("stream") && !request_json["stream"].is_boolean()) {
            res.status = 400;
            res.set_content(json{{"error", "stream must be a boolean."}}.dump(), "application/json");
            return;
        }
        int max_tokens = -1;
        if (request_json.contains("max_tokens")) {
            bool validMaxTokens = false;
            if (request_json["max_tokens"].is_number_unsigned()) {
                const uint64_t value = request_json["max_tokens"].get<uint64_t>();
                validMaxTokens = value > 0 && value <= static_cast<uint64_t>(std::numeric_limits<int>::max());
                if (validMaxTokens) {
                    max_tokens = static_cast<int>(value);
                }
            } else if (request_json["max_tokens"].is_number_integer()) {
                const int64_t value = request_json["max_tokens"].get<int64_t>();
                validMaxTokens = value > 0 && value <= std::numeric_limits<int>::max();
                if (validMaxTokens) {
                    max_tokens = static_cast<int>(value);
                }
            }
            if (!validMaxTokens) {
                res.status = 400;
                res.set_content(json{{"error", "max_tokens must be a positive integer."}}.dump(), "application/json");
                return;
            }
        }
        if (is_r1_) {
            prompts = ConvertToR1(std::move(prompts));
        }
        const std::string model = request_json.value("model", "undefined-model");
        const bool stream = request_json.value("stream", false);
        if (stream && scheduler_mode != "single_request") {
            res.status = 400;
            res.set_content(json{{"error", "stream=true is only supported in single_request mode."}}.dump(),
                            "application/json");
            return;
        }
        auto task = coordinator_->submit(std::move(prompts), max_tokens);
        if (!stream) {
            if (!task->wait()) {
                res.status = 500;
                res.set_content(json{{"error", "Model request failed."}}.dump(), "application/json");
                return;
            }
            const auto answer = task->answer();
            const auto timing = task->timing();
            json response_json = {
                {"id", "chatcmpl" + GetCurrentTimeAsString()},
                {"object", "chat.completion"},
                {"created", static_cast<int>(time(nullptr))},
                {"model", model},
                {"choices", json::array({{{"index", 0},
                    {"message", {{"role", "assistant"}, {"content", answer}}},
                    {"finish_reason", "stop"}}})},
                {"usage", {{"prompt_tokens", task->prompt_tokens()},
                            {"completion_tokens", task->generated_tokens()},
                            {"total_tokens", task->prompt_tokens() + task->generated_tokens()}}},
                {"timings", {{"queue_ms", timing.queueUs / 1000.0},
                              {"model_ttft_ms", timing.modelTtftUs / 1000.0},
                              {"model_tpot_ms", timing.modelTpotUs / 1000.0},
                              {"model_latency_ms", timing.modelLatencyUs / 1000.0},
                              {"e2e_ttft_ms", timing.e2eTtftUs / 1000.0},
                              {"e2e_latency_ms", timing.e2eLatencyUs / 1000.0},
                              {"scope", "model_and_service"}}}
            };
            res.set_content(response_json.dump(), "application/json");
            return;
        }
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider("text/event-stream",
            [task, model, coordinator = coordinator_.get()](size_t /*offset*/, httplib::DataSink& sink) {
                if (sink.is_writable && !sink.is_writable()) {
                    coordinator->cancel(task);
                    sink.done();
                    return false;
                }
                const auto write_sse = [&sink](const json& payload) {
                    const std::string data = "data: " + payload.dump() + "\n\n";
                    return sink.write(data.data(), data.size());
                };
                std::string answer;
                while (true) {
                    if (sink.is_writable && !sink.is_writable()) {
                        coordinator->cancel(task);
                        sink.done();
                        return false;
                    }
                    const auto chunkState = task->next_chunk(answer);
                    if (chunkState == RequestCoordinator::Task::ChunkState::Pending) {
                        continue;
                    }
                    if (chunkState == RequestCoordinator::Task::ChunkState::Finished) {
                        break;
                    }
                    const json content = {
                        {"id", "chatcmpl-" + GetCurrentTimeAsString()},
                        {"object", "chat.completion.chunk"},
                        {"created", static_cast<int>(std::time(nullptr))},
                        {"model", model},
                        {"choices", json::array({{{"delta", {{"content", answer}}}, {"index", 0}, {"finish_reason", nullptr}}})}
                    };
                    if (!write_sse(content)) {
                        coordinator->cancel(task);
                        sink.done();
                        return false;
                    }
                }
                if (!task->wait()) {
                    sink.done();
                    return false;
                }
                const json finish = {
                    {"id", "chatcmpl-" + GetCurrentTimeAsString()},
                    {"object", "chat.completion.chunk"},
                    {"created", static_cast<int>(std::time(nullptr))},
                    {"model", model},
                    {"choices", json::array({{{"delta", json::object()}, {"index", 0}, {"finish_reason", "stop"}}})}
                };
                if (!write_sse(finish) || !sink.write("data: [DONE]\n\n", 14)) {
                    coordinator->cancel(task);
                }
                sink.done();
                return false;
            });
    };
    
    // Register both endpoints with the same handler
    server.Post("/chat/completions", chatCompletionsHandler);
    server.Post("/v1/chat/completions", chatCompletionsHandler);
    // Start the server on specified host and port
    LOG_DEBUG("✅ Model initialized successfully!");
    LOG_DEBUG("🚀 Server ready at http://" + host + ":" + std::to_string(port));
    LOG_DEBUG("💡 Press Ctrl+C to stop the server");
    const bool listenSucceeded = server.listen(host.c_str(), port);
    if (!listenSucceeded) {
        LOG_DEBUG("Error: Could not start server on " + host + ":" + std::to_string(port));
    }
    return listenSucceeded;
}
}
