# Server + Dual Pipeline + ADB Review Summary

## Scope

- Baseline: `origin/feature/batch` (`d37ea6d2`)
- Committed head: `c57e89bd`
- Included: current staged, unstaged, and untracked worktree files
- Review date: 2026-07-28

## Outcome

- Server support is implemented. The service has startup-selected `single_request`, `continuous_batch`, and `dual_pipeline` modes, a serialized coordinator, JSON/SSE endpoints, cancellation plumbing, and batch response delivery.
- Dual-pipeline execution is implemented as real runtime work rather than metadata-only splitting. It owns two lane executors/runtime managers/modules/KV views, runs lane forwards on separate threads, schedules Host and QNN stages through separate one-at-a-time queues, and calls QNN graph preload/release callbacks.
- Android build and ADB test paths exist, and repository history records earlier QNN and CPU/OpenCL device runs. However, the newest uncommitted C6/C8 and timing changes have not been deployed and rerun in this review. Current-worktree end-to-end ADB acceptance is therefore not established.
- The implementation is not ready to claim complete C6/C8 Eagle validation or production-safe deployment. The highest-impact gaps are the Eagle four-request wave cap, stale split chunks after cancellation/error, and unsafe QNN binary/shape/name validation.

## Fresh Verification

- `llm`, `run_test.out`, and `spec_eval` rebuilt successfully from `/tmp/mnn-pp-review-llm`.
- `run_test.out llm`: 40 passed, 0 failed.
- Throughput harness unit tests: 26 passed, 0 failed.
- Python compilation, shell syntax, and diff whitespace checks passed.
- Read-only ADB preflight: one online arm64 RMX5090, no inference process or forward, 115 GiB free under `/data`, and existing model/QNN runtime directories present.
- No binary was pushed and no service or model inference was started during this review.

## Scope Cleanup

- Restore the unrelated deletion of top-level `build_lib.sh` or move it to a separate intentional change.
- Consolidate the two overlapping review documents and refresh their stale 2 ms/four-request descriptions.
- Decide whether the untracked throughput harness is product-owned validation code; if it is, track it with its tests and README.
- Treat `docs/superpowers/` as process material unless the repository explicitly adopts it.
