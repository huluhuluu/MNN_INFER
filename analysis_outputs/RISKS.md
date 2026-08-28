# Confirmed Risks

## High

1. QNN-exported Eagle models still schedule at most four requests per internal wave, while the HTTP coordinator and AR path accept eight. C6/C8 therefore does not produce 3+3 or 4+4 Eagle lane groups.
2. `BatchScheduler` prioritizes queued split chunks, but request release does not remove them. AR clears them only for `INTERNAL_ERROR`, not `USER_CANCEL`; Eagle clears neither path. A later batch can consume an old segment.
3. `RawExecutorWrapper` trusts mmap/file results, graph-count/name parity, `shapeIndex`, and tensor-name matches. Invalid or stale deployment metadata can cause out-of-range access or execute with unbound buffers.

## Medium

1. Model TTFT/TPOT publication is filled only by the AR dual branch; continuous and Eagle responses retain invalid metrics.
2. The full C6/C8 plan uses 20 requests and partial final waves, so C6 measures 6+6+6+2 and C8 measures 8+8+4.
3. Trace acceptance checks lane presence and Host/QNN maxima but does not require overlap, zero cancellation, disjoint lane ownership, or exact lane cardinality.
4. ADB commands and thermal wait loops have no total timeout. A blocked ADB subprocess can also outlive the monitor's ten-second join.
5. `dual_pipeline_max_resident_graphs` is a soft target. If every resident graph is pinned or active, eviction stops and the incoming graph is still loaded.
6. Destroying one `Llm` instance clears a process-global QNN raw graph pool after instance-specific scheduler cleanup, invalidating residency assumptions in other live instances.
7. Server listen failure is logged but not propagated, request scalar types and positive `max_tokens` are not validated as 4xx errors, and batch-mode SSE emits only after the whole batch finishes.
8. Binding the plaintext service beyond loopback exposes unauthenticated wildcard-CORS chat and reset endpoints.

## Deployment And Maintenance

1. The QNN exporter exposes `--buckets` but rejects every set except `1 32 256 512`; C6/C8 both select bucket 32.
2. Generated `config_qnn.json` does not set `scheduler_mode`, so dual mode requires a CLI override or a separate config edit.
3. `project/android/mv2adb.sh` only pushes files. It lacks strict error handling, serial selection, environment setup, model/config deployment, execution, and assertions.
4. The actual ADB throughput harness and its tests are untracked, so the committed implementation does not yet carry that automated acceptance surface.
5. `build_lib.sh` is deleted in the worktree even though it is a repository-wide packaging entrypoint unrelated to this feature.

## Evidence Boundary

Component and host tests demonstrate scheduler invariants and build integrity. Historical device artifacts demonstrate that earlier revisions ran on Android. Neither proves that the current uncommitted worktree passes C6/C8 QNN Eagle execution, cancellation recovery, or all-mode timing acceptance.
