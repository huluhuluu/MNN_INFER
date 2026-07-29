# Dual Pipeline C6/C8 Metrics Design

## Scope

Run QNN `dual_pipeline` with total concurrency 6 and 8. With two lanes this exercises lane batch sizes 3+3 and 4+4. Runs are sequential in one detached tmux session so they do not contend for the same device resources.

## Timing Semantics

- `queue_ms`: HTTP task creation to model admission after prompt tokenization.
- `model_ttft_ms`: batch request registration to its first generated token.
- `model_tpot_ms`: first-to-last generated-token interval divided by `completion_tokens - 1`.
- `model_latency_ms`: batch request registration to request completion.
- `e2e_latency_ms`: HTTP task creation to response completion.
- `e2e_ttft_ms`: `queue_ms + model_ttft_ms`.

The model timings are captured in `BatchScheduler::Request`, not inferred from HTTP chunk delivery. This matters because dual batch responses are delivered only after the complete batch finishes.

## Batch Limits

The service coordinator and dual AR scheduler both use a maximum total batch of 8. `dual_pipeline_split_count=2` remains unchanged. The scheduler's balanced assignment yields 3+3 for C6 and 4+4 for C8.

## Resource Protection

The host monitor polls every five seconds and records PhoneTemp, `/proc/meminfo` `MemAvailable`, and the owned service PID's `VmRSS`. It stops only that PID when PhoneTemp exceeds 42 C, available memory falls below 2 GiB, service RSS exceeds 80% of physical memory, or ADB resource polling fails repeatedly. C6 and C8 run sequentially and each gets a fresh server.

## Artifacts

Each request JSONL includes token counts, queue/model/e2e timings, TTFT, TPOT, and per-request token rates. Cell JSON includes aggregate throughput, requests/s, latency/TTFT/TPOT distributions, wave drift, temperatures, memory samples, hashes, and acceptance status. tmux stdout/stderr and a status file remain on the host.

## Execution

Use the existing QNN model/runtime and current Android service build path. First run a short C6/C8 smoke workload; only if both pass and resource limits remain healthy does the same detached session continue to the longer workload.
