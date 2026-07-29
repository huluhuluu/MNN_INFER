# Dual Pipeline C6/C8 Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure real per-request TTFT/TPOT and throughput for QNN dual-pipeline C6/C8 while protecting the attached Android device.

**Architecture:** Record request timing at the batch scheduler token boundary, expose aligned metrics through `Llm`, serialize them in the server response, and aggregate them in the Python harness. Raise both service and engine batch limits to eight, then run sequential smoke and long tests in detached tmux with thermal/memory/RSS guards.

**Tech Stack:** C++17, MNN LLM scheduler, cpp-httplib JSON API, Python 3 standard library, ADB, tmux.

## Global Constraints

- Use `apps/mnncli/build.sh --android-service`; do not create a second Android build tree.
- Use at most eight build jobs and do not use `--clean`.
- Do not modify or copy existing model/runtime directories.
- Stop only the exact PID owned by this run.
- Run C6 and C8 sequentially.
- Do not commit or push unless explicitly requested.

---

### Task 1: Scheduler Timing And Batch Eight

**Files:**
- Modify: `transformers/llm/engine/include/llm/BatchScheduler.hpp`
- Modify: `transformers/llm/engine/src/BatchScheduler.cpp`
- Modify: `test/llm/BatchSchedulerTest.cpp`
- Modify: `transformers/llm/engine/src/llm.cpp`

- [ ] Add failing scheduler tests proving an eight-request wave splits 4+4 and request timing records first/completion timestamps with nonnegative TTFT/TPOT.
- [ ] Run focused tests and confirm they fail for missing timing API or the four-request engine limit.
- [ ] Add request registration, first-token, and completion timestamps plus a read-only timing result API.
- [ ] Raise the dual AR `scheduleWave` total request limit from 4 to 8.
- [ ] Run focused scheduler tests and confirm they pass.

### Task 2: LLM And HTTP Timing Contract

**Files:**
- Modify: `transformers/llm/engine/include/llm/llm.hpp`
- Modify: `transformers/llm/engine/src/llm.cpp`
- Modify: `apps/mnncli/src/mnncli_server.cpp`

- [ ] Add a public aligned batch metric structure with model TTFT, TPOT, and latency.
- [ ] Populate metrics before scheduler requests are released.
- [ ] Raise the coordinator maximum batch from 4 to 8.
- [ ] Store queue/e2e timestamps on service tasks and serialize a `timings` object in nonstreaming responses.
- [ ] Add trace fields for model TTFT/TPOT and verify compilation.

### Task 3: Harness Metrics And Resource Guard

**Files:**
- Modify: `apps/mnncli/test/test_server_throughput.py`
- Modify: `apps/mnncli/test/server_throughput.py`
- Modify: `apps/mnncli/test/server_throughput.md`

- [ ] Add failing tests for timing parsing, per-request TPOT/TTFT distributions, C6/C8 plan construction, and memory/RSS guard decisions.
- [ ] Run Python tests and confirm the new cases fail.
- [ ] Parse response `timings`, add per-request rates and aggregate distributions.
- [ ] Extend the device monitor with MemAvailable, MemTotal, owned PID RSS, consecutive ADB failures, and exact-PID stop reasons.
- [ ] Add a dual-only C6/C8 execution profile and host resource JSONL.
- [ ] Run Python tests, Ruff, and py_compile.

### Task 4: Build, Smoke, And Detached Execution

**Files:**
- Update: `task_plan.md`
- Update: `findings.md`
- Update: `progress.md`

- [ ] Incrementally build with `MNN_BUILD_JOBS=8 apps/mnncli/build.sh --android-service`.
- [ ] Run focused host scheduler/LLM tests and verify the Android artifact is newer than modified sources.
- [ ] Verify device idle state, free memory, temperature, disk, staging, and no forward.
- [ ] Start one detached tmux session that runs C6 smoke then C8 smoke, followed by longer C6/C8 runs only after smoke success.
- [ ] Record the tmux session name, output directory, status/log commands, and resource guard thresholds.
