#!/usr/bin/env python3
"""Android mnncli server throughput and correctness harness."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import posixpath
import re
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT_S = 600.0
DEFAULT_ADB_TIMEOUT_S = 600.0
DEFAULT_COOLDOWN_TIMEOUT_S = 1800.0
MIN_DEVICE_FREE_BYTES = 10 * 1024**3
MAX_DEVICE_STAGE_BYTES = 64 * 1024**2
MAX_HOST_RESULTS_BYTES = 256 * 1024**2
REMOTE_PORT = 18080
MIN_DEVICE_AVAILABLE_BYTES = 2 * 1024**3
MAX_PROCESS_RSS_FRACTION = 0.80


@dataclass(frozen=True)
class MatrixConfig:
    backend: str
    mode: str
    config: dict[str, Any]


@dataclass(frozen=True)
class CellSpec:
    backend: str
    mode: str
    config: dict[str, Any]
    concurrency: int
    limit: int
    max_tokens: int
    warmup: int
    trace: bool

    @property
    def cell_id(self) -> str:
        return f"{self.backend}-{self.mode}-c{self.concurrency}"


class Adb:
    def __init__(self, serial: str | None):
        self.serial = serial or self._discover_serial()

    @staticmethod
    def _discover_serial() -> str:
        result = subprocess.run(
            ["adb", "devices"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=DEFAULT_ADB_TIMEOUT_S,
        )
        devices = [
            line.split("\t", 1)[0] for line in result.stdout.splitlines()
            if line.endswith("\tdevice")
        ]
        if len(devices) != 1:
            raise RuntimeError(
                f"--adb-serial is required when connected device count is {len(devices)}"
            )
        return devices[0]

    @property
    def prefix(self) -> list[str]:
        return ["adb", "-s", self.serial]

    def run(self, *arguments: str, check: bool = True, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        kwargs.setdefault("timeout", DEFAULT_ADB_TIMEOUT_S)
        return subprocess.run(
            self.prefix + list(arguments), check=check, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs,
        )

    def shell(self, command: str, check: bool = True) -> str:
        return self.run("shell", command, check=check).stdout.strip()

    def popen_shell(self, command: str, output: Any) -> subprocess.Popen[str]:
        return subprocess.Popen(
            self.prefix + ["shell", command], text=True,
            stdout=output, stderr=subprocess.STDOUT,
        )


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class RemoteServer:
    def __init__(
        self,
        adb: Adb,
        remote_stage: str,
        local_log: Path,
        cell_id: str,
        qnn_runtime_dir: str | None,
        trace: bool,
    ):
        self.adb = adb
        self.remote_stage = remote_stage.rstrip("/")
        self.local_log = local_log
        self.cell_id = cell_id
        self.qnn_runtime_dir = qnn_runtime_dir
        self.trace = trace
        self.pid: int | None = None
        self.local_port: int | None = None
        self.remote_config = posixpath.join(
            self.remote_stage, f"throughput-{self.cell_id}.json"
        )
        self.remote_pid_file = posixpath.join(
            self.remote_stage, f"throughput-{self.cell_id}.pid"
        )
        self.process: subprocess.Popen[str] | None = None
        self._log_stream: Any = None
        self._stopped = False
        self._stop_lock = threading.Lock()

    def start(
        self,
        local_config: Path,
        readiness_timeout: float,
        pid_ready: Callable[[int], None] | None = None,
    ) -> str:
        self.adb.run("push", str(local_config), self.remote_config)
        self.local_port = _unused_local_port()
        self.adb.run("forward", f"tcp:{self.local_port}", f"tcp:{REMOTE_PORT}")
        self.local_log.parent.mkdir(parents=True, exist_ok=True)
        self._log_stream = self.local_log.open("w", encoding="utf-8")
        environment = []
        if self.qnn_runtime_dir:
            environment.extend([
                f"LD_LIBRARY_PATH={self.qnn_runtime_dir}",
                f"ADSP_LIBRARY_PATH={self.qnn_runtime_dir}",
            ])
        if self.trace:
            environment.append("MNN_ACCEPTANCE_TRACE=1")
        env_command = "env " + " ".join(shlex.quote(item) for item in environment) + " " if environment else ""
        command = (
            f"cd {shlex.quote(self.remote_stage)} && "
            f"echo $$ > {shlex.quote(self.remote_pid_file)} && "
            f"exec {env_command}./mnncli-throughput serve "
            f"--config {shlex.quote(self.remote_config)} --host 127.0.0.1 --port {REMOTE_PORT}"
        )
        self.process = self.adb.popen_shell(command, self._log_stream)
        deadline = time.monotonic() + min(30.0, readiness_timeout)
        while time.monotonic() < deadline:
            result = self.adb.run("shell", f"cat {shlex.quote(self.remote_pid_file)}", check=False)
            value = result.stdout.strip()
            if result.returncode == 0 and value.isdigit():
                self.pid = int(value)
                break
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited before writing PID; see {self.local_log}")
            time.sleep(0.2)
        if self.pid is None:
            raise RuntimeError("server did not publish its PID")
        if pid_ready is not None:
            pid_ready(self.pid)
        base_url = f"http://127.0.0.1:{self.local_port}"
        self._wait_ready(base_url, readiness_timeout)
        return base_url

    def _wait_ready(self, base_url: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_error = "not attempted"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"server exited during model load; see {self.local_log}")
            try:
                with urlopen(base_url + "/v1/models", timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, HTTPError, URLError) as exc:
                last_error = str(exc)
            time.sleep(1)
        raise RuntimeError(f"server readiness timed out after {timeout}s: {last_error}")

    def stop(self) -> None:
        with self._stop_lock:
            if self._stopped or self.pid is None:
                return
            self.adb.run("shell", f"kill {self.pid}", check=False)
            time.sleep(0.2)
            alive = self.adb.run("shell", f"kill -0 {self.pid}", check=False)
            if alive.returncode == 0:
                self.adb.run("shell", f"kill -9 {self.pid}", check=False)
            self._stopped = True

    def close(self) -> None:
        self.stop()
        if self.local_port is not None:
            self.adb.run("forward", "--remove", f"tcp:{self.local_port}", check=False)
            self.local_port = None
        self.adb.run(
            "shell",
            f"rm -f {shlex.quote(self.remote_config)} {shlex.quote(self.remote_pid_file)}",
            check=False,
        )
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None


def load_dataset(
    path: str | Path,
    limit: int,
    prompt_field: str = "prompt",
) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if not prompt_field:
        raise ValueError("prompt_field must be non-empty")
    records: list[dict[str, Any]] = []
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{dataset_path.name}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{dataset_path.name}:{line_number}: expected a JSON object")
            prompt = record.get(prompt_field)
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(
                    f"{dataset_path.name}:{line_number}: {prompt_field} must be a non-empty string"
                )
            normalized = dict(record)
            normalized["prompt"] = prompt
            normalized.setdefault("id", len(records))
            records.append(normalized)
            if limit and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"{dataset_path}: dataset has no prompts")
    if limit and len(records) < limit:
        raise ValueError(f"{dataset_path}: requested {limit} prompts, found {len(records)}")
    return records


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: list[float], percentiles: tuple[int, ...]) -> dict[str, float | None]:
    result: dict[str, float | None] = {f"p{item}": _percentile(values, item) for item in percentiles}
    result["max"] = max(values) if values else None
    return result


def summarize(
    responses: list[dict[str, Any]],
    waves: list[dict[str, Any]],
    wall_time_s: float,
) -> dict[str, Any]:
    successful = [record for record in responses if record.get("ok")]
    failed = [record for record in responses if not record.get("ok")]
    completion_tokens = sum(int(record.get("completion_tokens", 0)) for record in successful)
    prompt_tokens = sum(int(record.get("prompt_tokens", 0)) for record in successful)
    wave_rates = [
        float(wave["completion_tokens"]) / float(wave["wall_time_s"])
        for wave in waves
        if wave.get("successful_requests", 0) and wave.get("wall_time_s", 0) > 0
    ]
    first = statistics.fmean(wave_rates[:5]) if wave_rates else None
    last = statistics.fmean(wave_rates[-5:]) if wave_rates else None
    drift = None
    if first is not None and last is not None and first > 0:
        drift = (last - first) / first * 100.0
    summary = {
        "requested_requests": len(responses),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "requests_per_s": len(successful) / wall_time_s if wall_time_s > 0 else 0.0,
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "wall_time_s": wall_time_s,
        "aggregate_completion_tokens_per_s": (
            completion_tokens / wall_time_s if wall_time_s > 0 else 0.0
        ),
        "request_latency_s": _distribution(
            [float(record["latency_s"]) for record in successful], (50, 90, 95)
        ),
        "wave_tokens_per_s": _distribution(wave_rates, (50, 90)),
        "first_five_wave_tokens_per_s_mean": first,
        "last_five_wave_tokens_per_s_mean": last,
        "first_to_last_wave_drift_percent": drift,
        "http_errors": sum(record.get("error_type") == "http" for record in failed),
        "timeout_errors": sum(record.get("error_type") == "timeout" for record in failed),
        "transport_errors": sum(record.get("error_type") == "transport" for record in failed),
        "protocol_errors": sum(record.get("error_type") == "protocol" for record in failed),
        "empty_outputs": sum(bool(record.get("empty_output")) for record in responses),
        "zero_usage_responses": sum(bool(record.get("zero_usage")) for record in responses),
    }
    for field in (
        "model_ttft_ms", "model_tpot_ms", "model_latency_ms", "queue_ms", "e2e_ttft_ms"
    ):
        values = [
            float(record[field]) for record in successful
            if isinstance(record.get(field), (int, float)) and record[field] >= 0
        ]
        summary[field] = _distribution(values, (50, 90, 95))
    return summary


def post_completion(
    base_url: str,
    record: dict[str, Any],
    max_tokens: int,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": record["prompt"]}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0,
    }
    request = Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body)


def _error_record(
    record: dict[str, Any],
    position: int,
    wave_index: int,
    latency_s: float,
    error: BaseException,
) -> dict[str, Any]:
    error_type = "transport"
    http_status = None
    detail = str(error)
    if isinstance(error, HTTPError):
        error_type = "http"
        http_status = error.code
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except Exception:
            pass
    elif isinstance(error, (socket.timeout, TimeoutError)):
        error_type = "timeout"
    elif isinstance(error, URLError) and isinstance(error.reason, (socket.timeout, TimeoutError)):
        error_type = "timeout"
    elif isinstance(error, json.JSONDecodeError):
        error_type = "protocol"
    return {
        "id": record["id"],
        "dataset_index": position,
        "wave_index": wave_index,
        "ok": False,
        "status": "error",
        "response": "",
        "http_status": http_status,
        "latency_s": latency_s,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error_type": error_type,
        "error": detail,
    }


def _success_record(
    source: dict[str, Any],
    position: int,
    wave_index: int,
    latency_s: float,
    http_status: int,
    payload: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    try:
        content = payload["choices"][0]["message"]["content"]
        usage = payload["usage"]
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
        if not isinstance(content, str):
            raise TypeError("choice content is not a string")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return {
            "id": source["id"],
            "dataset_index": position,
            "wave_index": wave_index,
            "ok": False,
            "status": "error",
            "response": "",
            "http_status": http_status,
            "latency_s": latency_s,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error_type": "protocol",
            "error": f"invalid OpenAI response: {exc}",
        }
    empty_output = not content.strip()
    zero_usage = prompt_tokens <= 0 or completion_tokens <= 0
    ok = http_status == 200 and not empty_output and not zero_usage
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
    timing_values = {}
    for field in (
        "queue_ms", "model_ttft_ms", "model_tpot_ms", "model_latency_ms",
        "e2e_ttft_ms", "e2e_latency_ms",
    ):
        value = timings.get(field, -1)
        timing_values[field] = float(value) if isinstance(value, (int, float)) else -1.0
    return {
        "id": source["id"],
        "dataset_index": position,
        "wave_index": wave_index,
        "ok": ok,
        "status": (
            "max_tokens_finished" if ok and completion_tokens >= max_tokens
            else "normal_finished" if ok else "error"
        ),
        "response": content,
        "http_status": http_status,
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "empty_output": empty_output,
        "zero_usage": zero_usage,
        "error_type": None if ok else "protocol",
        "error": None if ok else "empty output or zero usage",
        "e2e_completion_tokens_per_s": completion_tokens / latency_s if latency_s > 0 else 0.0,
        "decode_tokens_per_s": (
            1000.0 / timing_values["model_tpot_ms"]
            if timing_values["model_tpot_ms"] > 0 else None
        ),
        **timing_values,
    }


def resource_guard_reason(
    sample: dict[str, Any],
    min_available_bytes: int = MIN_DEVICE_AVAILABLE_BYTES,
    max_rss_fraction: float = MAX_PROCESS_RSS_FRACTION,
) -> str | None:
    temperature = float(sample.get("temperature_c", 0.0))
    if temperature > 42.0:
        return f"temperature {temperature:.1f} C exceeds 42.0 C"
    available = int(sample.get("mem_available_bytes", 0))
    if available < min_available_bytes:
        return f"available memory {available} bytes is below {min_available_bytes}"
    total = int(sample.get("mem_total_bytes", 0))
    rss = int(sample.get("process_rss_bytes", 0))
    if total > 0 and rss > total * max_rss_fraction:
        return f"service RSS {rss} bytes exceeds {max_rss_fraction:.0%} of {total}"
    return None


def resource_guard_enabled(args: argparse.Namespace) -> bool:
    return not getattr(args, "disable_resource_guard", False)


def close_optional_monitor(monitor: TemperatureMonitor | None) -> None:
    if monitor is not None:
        monitor.close()


def run_load(
    base_url: str,
    records: list[dict[str, Any]],
    max_tokens: int,
    concurrency: int,
    timeout: float = DEFAULT_TIMEOUT_S,
    request_fn: Callable[[str, dict[str, Any], int, float], tuple[int, dict[str, Any]]] = post_completion,
    allow_partial_final_wave: bool = False,
) -> dict[str, Any]:
    if concurrency <= 0:
        raise ValueError("concurrency must be > 0")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if len(records) % concurrency and not allow_partial_final_wave:
        raise ValueError(
            f"partial wave refused: {len(records)} requests is not divisible by concurrency {concurrency}"
        )

    responses: list[dict[str, Any]] = []
    waves: list[dict[str, Any]] = []
    measurement_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for wave_index, offset in enumerate(range(0, len(records), concurrency)):
            wave_records = records[offset:offset + concurrency]
            barrier = threading.Barrier(len(wave_records) + 1)

            def invoke(position: int, source: dict[str, Any]) -> dict[str, Any]:
                barrier.wait()
                started = time.monotonic()
                try:
                    status, payload = request_fn(base_url, source, max_tokens, timeout)
                    return _success_record(
                        source, position, wave_index, time.monotonic() - started,
                        status, payload, max_tokens,
                    )
                except Exception as exc:
                    return _error_record(
                        source, position, wave_index, time.monotonic() - started, exc
                    )

            futures = [
                executor.submit(invoke, offset + index, source)
                for index, source in enumerate(wave_records)
            ]
            wave_start = time.monotonic()
            barrier.wait()
            wave_responses = [future.result() for future in futures]
            wave_wall_time = time.monotonic() - wave_start
            responses.extend(wave_responses)
            wave_successes = [record for record in wave_responses if record["ok"]]
            waves.append({
                "wave_index": wave_index,
                "request_count": len(wave_responses),
                "successful_requests": len(wave_successes),
                "failed_requests": len(wave_responses) - len(wave_successes),
                "wall_time_s": wave_wall_time,
                "completion_tokens": sum(record["completion_tokens"] for record in wave_successes),
                "completion_tokens_per_s": (
                    sum(record["completion_tokens"] for record in wave_successes) / wave_wall_time
                    if wave_wall_time > 0 else 0.0
                ),
            })
    wall_time_s = time.monotonic() - measurement_start
    responses.sort(key=lambda item: item["dataset_index"])
    return {
        "summary": summarize(responses, waves, wall_time_s),
        "waves": waves,
        "responses": responses,
    }


def build_matrix_configs(
    host_base: dict[str, Any], qnn_base: dict[str, Any]
) -> list[MatrixConfig]:
    configs: list[MatrixConfig] = []
    for backend in ("cpu", "opencl", "qnn"):
        for mode in ("single_request", "continuous_batch", "dual_pipeline"):
            config = dict(qnn_base if backend == "qnn" else host_base)
            for key in (
                "scheduler_mode", "packed_attention_mode", "dual_pipeline_mode",
                "dual_pipeline", "dual_pipeline_split_count",
                "dual_pipeline_max_resident_graphs", "dual_pipeline_prefetch_window",
            ):
                config.pop(key, None)
            config.update({
                "backend_type": "cpu" if backend == "qnn" else backend,
                "thread_num": 1 if backend == "qnn" else 4,
                "precision": "low",
                "memory": "low",
                "power": "high",
                "sampler_type": "greedy",
                "scheduler_mode": mode,
                "packed_attention_mode": mode != "single_request",
                "dual_pipeline_mode": mode == "dual_pipeline",
            })
            if mode == "dual_pipeline":
                config["dual_pipeline_split_count"] = 2
                if backend == "qnn":
                    config["dual_pipeline_max_resident_graphs"] = 30
                    config["dual_pipeline_prefetch_window"] = 2
            configs.append(MatrixConfig(backend, mode, config))
    return configs


def build_execution_plan(configs: list[MatrixConfig], phase: str) -> list[CellSpec]:
    if phase not in {"dev", "full"}:
        raise ValueError("phase must be dev or full")
    indexed = {(item.backend, item.mode): item for item in configs}
    mode_order = {
        "cpu": ("single_request", "continuous_batch", "dual_pipeline"),
        "opencl": ("continuous_batch", "dual_pipeline", "single_request"),
        "qnn": ("dual_pipeline", "single_request", "continuous_batch"),
    }
    plan: list[CellSpec] = []
    for backend, modes in mode_order.items():
        for mode in modes:
            matrix_config = indexed[(backend, mode)]
            profiles = ((4, 8, 16, 4, True),) if phase == "dev" else (
                (1, 20, 512, 1, False),
                (4, 100, 512, 4, False),
            )
            for concurrency, limit, max_tokens, warmup, trace in profiles:
                plan.append(CellSpec(
                    backend, mode, matrix_config.config, concurrency, limit,
                    max_tokens, warmup, trace,
                ))
    return plan


def build_dual_benchmark_plan(configs: list[MatrixConfig], phase: str) -> list[CellSpec]:
    if phase not in {"smoke", "full"}:
        raise ValueError("dual benchmark phase must be smoke or full")
    config = next(
        item.config for item in configs
        if item.backend == "qnn" and item.mode == "dual_pipeline"
    )
    if phase == "smoke":
        profiles = ((6, 6, 16, 6, True), (8, 8, 16, 8, True))
    else:
        profiles = ((6, 20, 512, 6, False), (8, 20, 512, 8, False))
    return [
        CellSpec("qnn", "dual_pipeline", config, concurrency, limit,
                 max_tokens, warmup, trace)
        for concurrency, limit, max_tokens, warmup, trace in profiles
    ]


def _trace_values(line: str) -> dict[str, int]:
    return {key: int(value) for key, value in re.findall(r"(\w+)=(-?\d+)", line)}


def validate_server_log(
    text: str,
    backend: str,
    mode: str,
    require_trace: bool,
    expected_batch_size: int = 4,
) -> list[str]:
    errors: list[str] = []
    common_markers = (
        (r"\bINTERNAL_ERROR\b", "server reported INTERNAL_ERROR"),
        (r"segmentation fault|fatal signal|\bSIGSEGV\b|\babort(?:ed)?\b", "server crashed"),
    )
    qnn_markers = (
        r"(?=.*(?:QNN|Qnn))(?=.*(?:graphValidate|graph validate|graphExecute|deviceCreate))"
        r"(?=.*(?:fail|error|(?:=|code\s+)[1-9]\d*)).*",
        r"(?:graphValidate|graph validate|graphExecute|deviceCreate).*"
        r"(?:fail|error|(?:=|code\s+)[1-9]\d*)",
    )
    opencl_markers = (
        r"(?=.*OpenCL)(?=.*(?:kernel|program|runtime|build))(?=.*(?:fail|error)).*",
        r"(?:CL_BUILD_PROGRAM_FAILURE|CL_OUT_OF_RESOURCES|CL_INVALID_KERNEL)",
    )
    for pattern, message in common_markers:
        if re.search(pattern, text, re.IGNORECASE):
            errors.append(message)
    if backend == "qnn" and any(
        re.search(pattern, text, re.IGNORECASE | re.MULTILINE) for pattern in qnn_markers
    ):
        errors.append("QNN graph validate/execute/deviceCreate error found")
    if backend == "opencl" and any(
        re.search(pattern, text, re.IGNORECASE | re.MULTILINE) for pattern in opencl_markers
    ):
        errors.append("OpenCL kernel build/runtime error found")
    if not require_trace:
        return errors

    lines = text.splitlines()
    if mode == "single_request":
        active: set[int] = set()
        starts = 0
        for line in lines:
            values = _trace_values(line)
            if "event=request_started" in line and "request_scope=service" in line:
                request_id = values.get("request_id", -1)
                if active:
                    errors.append("single_request trace has overlapping requests")
                    break
                active.add(request_id)
                starts += 1
            elif "event=request_completed" in line and "request_scope=service" in line:
                active.discard(values.get("request_id", -1))
        if starts == 0:
            errors.append("single_request trace has no request_started event")
    else:
        batch_sizes = [
            _trace_values(line).get("request_count", 0)
            for line in lines if f"event=batch_started mode={mode}" in line
        ]
        if expected_batch_size not in batch_sizes:
            errors.append(f"{mode} trace has no {expected_batch_size}-request batch")

    if mode != "dual_pipeline":
        return errors
    lanes = {
        _trace_values(line).get("lane")
        for line in lines if "event=lane_owner" in line
    }
    if not {0, 1}.issubset(lanes):
        errors.append("dual_pipeline trace does not use both lane 0 and lane 1")
    summaries = [
        _trace_values(line) for line in lines if "event=stage_summary" in line
    ]
    if not summaries:
        errors.append("dual_pipeline trace has no stage_summary")
        return errors
    if any(item.get("max_host", 0) > 1 for item in summaries):
        errors.append("dual_pipeline stage_summary has max_host > 1")
    if any(item.get("max_qnn", 0) > 1 for item in summaries):
        errors.append("dual_pipeline stage_summary has max_qnn > 1")
    if backend == "qnn":
        if sum(item.get("qnn_completed", 0) for item in summaries) <= 0:
            errors.append("QNN stage did not execute")
    elif any(item.get("max_qnn", 0) != 0 for item in summaries):
        errors.append(f"{backend} Host-only dual trace has max_qnn != 0")
    return errors


def _responses_path(output: Path) -> Path:
    return output.with_name(output.stem + ".responses.jsonl")


def write_load_result(output: str | Path, result: dict[str, Any], metadata: dict[str, Any]) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"metadata": metadata, "summary": result["summary"], "waves": result["waves"]}
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with _responses_path(output_path).open("w", encoding="utf-8") as stream:
        for response in result["responses"]:
            stream.write(json.dumps(response, ensure_ascii=False) + "\n")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_host_config() -> dict[str, Any]:
    return {
        "llm_model": "llm.mnn",
        "llm_weight": "llm.mnn.weight",
        "backend_type": "cpu",
        "thread_num": 4,
        "precision": "low",
        "memory": "low",
    }


def _default_qnn_config() -> dict[str, Any]:
    return {
        "llm_model": "qnn/llm.mnn",
        "backend_type": "cpu",
        "thread_num": 1,
        "precision": "low",
        "chunk_limits": [128, 8, 1],
        "memory": "low",
    }


def _config_as_dict(item: MatrixConfig) -> dict[str, Any]:
    return {"backend": item.backend, "mode": item.mode, "config": item.config}


def _cell_as_dict(item: CellSpec) -> dict[str, Any]:
    return {
        "cell_id": item.cell_id,
        "backend": item.backend,
        "mode": item.mode,
        "concurrency": item.concurrency,
        "limit": item.limit,
        "max_tokens": item.max_tokens,
        "warmup": item.warmup,
        "acceptance_trace": item.trace,
    }


def _dry_run_report(args: argparse.Namespace) -> dict[str, Any]:
    records = load_dataset(args.dataset, 8 if args.phase == "dev" else 100)
    configs = build_matrix_configs(_default_host_config(), _default_qnn_config())
    plan = build_execution_plan(configs, args.phase)
    config_bytes = sum(
        len(json.dumps(item.config, ensure_ascii=False).encode("utf-8")) for item in configs
    )
    binary_bytes = Path(args.binary).stat().st_size
    return {
        "dry_run": True,
        "phase": args.phase,
        "device_connected": False,
        "dataset_records_checked": len(records),
        "paths": {
            "binary": str(Path(args.binary).resolve()),
            "dataset": str(Path(args.dataset).resolve()),
            "host_model_dir": args.host_model_dir,
            "qnn_model_dir": args.qnn_model_dir,
            "qnn_runtime_dir": args.qnn_runtime_dir,
            "remote_stage": args.remote_stage,
            "output_dir": str(Path(args.output_dir).resolve()),
        },
        "space_budget": {
            "minimum_device_data_free_bytes": MIN_DEVICE_FREE_BYTES,
            "maximum_device_stage_bytes": MAX_DEVICE_STAGE_BYTES,
            "maximum_host_results_bytes": MAX_HOST_RESULTS_BYTES,
            "estimated_device_stage_bytes": binary_bytes + config_bytes,
            "binary_bytes": binary_bytes,
            "generated_config_bytes": config_bytes,
        },
        "configs": [_config_as_dict(item) for item in configs],
        "execution_order": [_cell_as_dict(item) for item in plan],
    }


def _dual_dry_run_report(args: argparse.Namespace) -> dict[str, Any]:
    required = 8 if args.phase == "smoke" else 20
    records = load_dataset(args.dataset, required, args.prompt_field)
    configs = build_matrix_configs(_default_host_config(), _default_qnn_config())
    plan = build_dual_benchmark_plan(configs, args.phase)
    config_bytes = len(json.dumps(plan[0].config, ensure_ascii=False).encode("utf-8"))
    binary_bytes = Path(args.binary).stat().st_size
    return {
        "dry_run": True,
        "profile": "qnn-dual-c6-c8",
        "phase": args.phase,
        "prompt_field": args.prompt_field,
        "device_connected": False,
        "dataset_records_checked": len(records),
        "space_budget": {
            "maximum_device_stage_bytes": MAX_DEVICE_STAGE_BYTES,
            "estimated_device_stage_bytes": binary_bytes + config_bytes,
        },
        "execution_order": [_cell_as_dict(item) for item in plan],
    }


def run_matrix_command(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(json.dumps(_dry_run_report(args), ensure_ascii=False, indent=2))
        return 0
    return _run_matrix_live(args)


def run_dual_benchmark_command(args: argparse.Namespace) -> int:
    args.dual_benchmark = True
    if args.dry_run:
        print(json.dumps(_dual_dry_run_report(args), ensure_ascii=False, indent=2))
        return 0
    return _run_matrix_live(args)


def _read_remote_json(adb: Adb, directory: str, names: tuple[str, ...]) -> dict[str, Any]:
    failures = []
    for name in names:
        path = posixpath.join(directory.rstrip("/"), name)
        result = adb.run("shell", f"cat {shlex.quote(path)}", check=False)
        if result.returncode != 0:
            failures.append(path)
            continue
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"device config {path} is invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"device config {path} is not a JSON object")
        return parsed
    raise RuntimeError(f"none of the device configs exist: {', '.join(failures)}")


def _device_storage_inventory(adb: Adb, paths: list[str]) -> dict[str, Any]:
    df_output = adb.shell("df -Pk /data")
    rows = [line.split() for line in df_output.splitlines() if line.strip()]
    data_rows = [row for row in rows if len(row) >= 6 and row[3].isdigit()]
    if not data_rows:
        raise RuntimeError(f"cannot parse device df output: {df_output!r}")
    available_bytes = int(data_rows[-1][3]) * 1024
    sizes: dict[str, int] = {}
    for path in paths:
        output = adb.shell(f"du -sk {shlex.quote(path)}")
        value = output.split()[0] if output.split() else ""
        if not value.isdigit():
            raise RuntimeError(f"cannot parse device du output for {path}: {output!r}")
        sizes[path] = int(value) * 1024
    return {"df": df_output, "data_available_bytes": available_bytes, "directory_bytes": sizes}


def _device_path_size(adb: Adb, path: str) -> int:
    output = adb.shell(f"du -sk {shlex.quote(path)}")
    value = output.split()[0] if output.split() else ""
    if not value.isdigit():
        raise RuntimeError(f"cannot parse device staging size: {output!r}")
    return int(value) * 1024


def _check_device_stage(adb: Adb, remote_stage: str) -> int:
    size = _device_path_size(adb, remote_stage)
    if size > MAX_DEVICE_STAGE_BYTES:
        raise RuntimeError(
            f"device staging is {size} bytes, above the {MAX_DEVICE_STAGE_BYTES}-byte limit"
        )
    return size


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _check_host_results(path: Path) -> int:
    size = _directory_size(path)
    if size > MAX_HOST_RESULTS_BYTES:
        raise RuntimeError(
            f"host results are {size} bytes, above the {MAX_HOST_RESULTS_BYTES}-byte limit"
        )
    return size


def _remote_model_hashes(adb: Adb, model_dir: str, config: dict[str, Any]) -> dict[str, str]:
    paths = []
    for key in ("llm_model", "llm_weight"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            continue
        paths.append(value if value.startswith("/") else posixpath.join(model_dir, value))
    hashes: dict[str, str] = {}
    for path in dict.fromkeys(paths):
        output = adb.shell(f"sha256sum {shlex.quote(path)}")
        fields = output.split()
        if not fields or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            raise RuntimeError(f"cannot hash device model artifact {path}: {output!r}")
        hashes[path] = fields[0].lower()
    if not hashes:
        raise ValueError(f"model config under {model_dir} has no model artifacts to hash")
    return hashes


def read_phone_temperature(adb: Adb) -> float:
    output = adb.shell("dumpsys battery")
    match = re.search(r"^\s*PhoneTemp\s*:\s*(-?\d+(?:\.\d+)?)\s*$", output, re.MULTILINE)
    if match is None:
        match = re.search(r"^\s*temperature\s*:\s*(-?\d+(?:\.\d+)?)\s*$", output, re.MULTILINE)
    if match is None:
        raise RuntimeError("dumpsys battery has no PhoneTemp or temperature field")
    value = float(match.group(1))
    return value / 10.0 if abs(value) > 100 else value


def _kib_field(text: str, name: str) -> int:
    match = re.search(rf"^{re.escape(name)}:\s+(\d+)\s+kB$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"device status has no {name} field")
    return int(match.group(1)) * 1024


def read_device_resource_sample(adb: Adb, pid: int) -> dict[str, Any]:
    meminfo = adb.shell("cat /proc/meminfo")
    process_status = adb.shell(f"cat /proc/{pid}/status")
    return {
        "time": time.time(),
        "temperature_c": read_phone_temperature(adb),
        "mem_total_bytes": _kib_field(meminfo, "MemTotal"),
        "mem_available_bytes": _kib_field(meminfo, "MemAvailable"),
        "process_rss_bytes": _kib_field(process_status, "VmRSS"),
        "pid": pid,
    }


class OverheatError(RuntimeError):
    pass


def wait_for_thermal_gate(
    adb: Adb,
    stop_server: Callable[[], None],
    timeout_s: float = DEFAULT_COOLDOWN_TIMEOUT_S,
) -> list[dict[str, Any]]:
    readings: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s
    while True:
        temperature = read_phone_temperature(adb)
        readings.append({"time": time.time(), "temperature_c": temperature})
        readings = readings[-7:]
        if temperature > 42.0:
            stop_server()
            raise OverheatError(f"PhoneTemp reached {temperature:.1f} C during thermal gate")
        if (
            len(readings) == 7
            and max(item["temperature_c"] for item in readings) <= 32.0
            and max(item["temperature_c"] for item in readings)
            - min(item["temperature_c"] for item in readings) <= 1.0
        ):
            return readings
        if time.monotonic() >= deadline:
            raise TimeoutError(f"thermal gate did not pass within {timeout_s:.0f} seconds")
        time.sleep(10)


class TemperatureMonitor:
    def __init__(
        self,
        adb: Adb,
        stop_server: Callable[[], None],
        pid: int,
        output_path: Path | None = None,
    ):
        self.adb = adb
        self.stop_server = stop_server
        self.pid = pid
        self.output_path = output_path
        self.readings: list[dict[str, Any]] = []
        self.overheat: float | None = None
        self.error: str | None = None
        self.stop_reason: str | None = None
        self._consecutive_errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="device-resource", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sample = read_device_resource_sample(self.adb, self.pid)
                self._consecutive_errors = 0
                self.readings.append(sample)
                self._write_sample(sample)
                reason = resource_guard_reason(sample)
                if reason:
                    self.stop_reason = reason
                    if sample["temperature_c"] > 42.0:
                        self.overheat = sample["temperature_c"]
                    self.stop_server()
                    return
            except Exception as exc:
                self._consecutive_errors += 1
                sample = {
                    "time": time.time(), "pid": self.pid,
                    "poll_error": str(exc),
                    "consecutive_poll_errors": self._consecutive_errors,
                }
                self.readings.append(sample)
                self._write_sample(sample)
                if self._consecutive_errors >= 3:
                    self.error = str(exc)
                    self.stop_reason = f"ADB resource polling failed {self._consecutive_errors} times"
                    self.stop_server()
                    return
            self._stop.wait(5)

    def _write_sample(self, sample: dict[str, Any]) -> None:
        if self.output_path is None:
            return
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("device resource monitor did not stop within 10 seconds")


def _gzip_log(path: Path) -> Path:
    compressed = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(compressed, "wb", compresslevel=6) as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    path.unlink()
    return compressed


def _score_predictions(scorer: str, dataset: str, predictions: Path, output: Path) -> None:
    subprocess.run([
        sys.executable, scorer, "score", "--input", dataset,
        "--predictions", str(predictions), "--output", str(output),
    ], check=True)


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_cell_attempt(
    adb: Adb,
    args: argparse.Namespace,
    cell: CellSpec,
    attempt: int,
    records: list[dict[str, Any]],
    output_dir: Path,
    fingerprints: dict[str, Any],
) -> dict[str, Any]:
    attempt_id = f"{cell.cell_id}-attempt{attempt}"
    config_path = output_dir / "configs" / f"{attempt_id}.json"
    result_path = output_dir / f"{attempt_id}.json"
    log_path = output_dir / f"{attempt_id}.server.log"
    resource_path = output_dir / f"{attempt_id}.resources.jsonl"
    _write_config(config_path, cell.config)
    server = RemoteServer(
        adb, args.remote_stage, log_path, attempt_id,
        args.qnn_runtime_dir if cell.backend == "qnn" else None,
        cell.trace,
    )
    result: dict[str, Any] | None = None
    warmup_result: dict[str, Any] | None = None
    thermal_gate: list[dict[str, Any]] = []
    start_temperature: float | None = None
    end_temperature: float | None = None
    monitor: TemperatureMonitor | None = None
    caught: BaseException | None = None
    started_at = time.time()
    guard_enabled = resource_guard_enabled(args)

    def start_monitor(pid: int) -> None:
        nonlocal monitor
        monitor = TemperatureMonitor(adb, server.stop, pid, resource_path)
        monitor.start()

    try:
        base_url = server.start(
            config_path,
            args.readiness_timeout,
            start_monitor if guard_enabled else None,
        )
        _check_device_stage(adb, args.remote_stage)
        warmup_result = run_load(
            base_url, records[:cell.warmup], cell.max_tokens,
            cell.concurrency, args.timeout,
        )
        if warmup_result["summary"]["failed_requests"]:
            raise RuntimeError(f"{attempt_id}: warmup request failed")
        if args.phase == "full" and guard_enabled:
            thermal_gate = wait_for_thermal_gate(adb, server.stop)
        start_temperature = read_phone_temperature(adb)
        if args.phase == "full" and guard_enabled and start_temperature > 32.0:
            raise RuntimeError(
                f"{attempt_id}: start PhoneTemp {start_temperature:.1f} C exceeds 32.0 C"
            )
        result = run_load(
            base_url, records[:cell.limit], cell.max_tokens,
            cell.concurrency, args.timeout,
            allow_partial_final_wave=getattr(args, "allow_partial_final_wave", False),
        )
        close_optional_monitor(monitor)
        end_temperature = read_phone_temperature(adb)
    except BaseException as exc:
        caught = exc
    finally:
        close_optional_monitor(monitor)
        server.close()

    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    log_errors = validate_server_log(
        log_text, cell.backend, cell.mode, cell.trace, cell.concurrency
    )
    completed_at = time.time()
    metadata = {
        "cell_id": cell.cell_id,
        "attempt": attempt,
        "phase": args.phase,
        "backend": cell.backend,
        "mode": cell.mode,
        "concurrency": cell.concurrency,
        "limit": cell.limit,
        "max_tokens": cell.max_tokens,
        "warmup_requests": cell.warmup,
        "started_at_unix_s": started_at,
        "completed_at_unix_s": completed_at,
        "start_temperature_c": start_temperature,
        "end_temperature_c": end_temperature,
        "thermal_gate_readings": thermal_gate,
        "measurement_temperature_readings": monitor.readings if monitor else [],
        "resource_stop_reason": monitor.stop_reason if monitor else None,
        "resource_guard_enabled": guard_enabled,
        "warmup_summary": warmup_result["summary"] if warmup_result else None,
        "log_errors": log_errors,
        "config": cell.config,
        "config_sha256": _sha256_file(config_path),
        **fingerprints,
    }
    if result is not None:
        write_load_result(result_path, result, metadata)
    else:
        result_path.write_text(json.dumps({
            "metadata": metadata,
            "error": str(caught) if caught else "cell ended without a result",
            "log_errors": log_errors,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compressed_log = _gzip_log(log_path) if log_path.exists() else None

    if monitor is not None and monitor.overheat is not None:
        raise OverheatError(
            f"{attempt_id}: PhoneTemp reached {monitor.overheat:.1f} C; server stopped"
        )
    if monitor is not None and monitor.error is not None:
        raise RuntimeError(f"{attempt_id}: temperature monitor failed: {monitor.error}")
    if monitor is not None and monitor.stop_reason is not None:
        raise RuntimeError(f"{attempt_id}: resource guard stopped server: {monitor.stop_reason}")
    if caught is not None:
        raise caught
    assert result is not None
    if result["summary"]["failed_requests"]:
        raise RuntimeError(
            f"{attempt_id}: {result['summary']['failed_requests']} measured requests failed"
        )
    if log_errors:
        raise RuntimeError(f"{attempt_id}: log acceptance failed: {'; '.join(log_errors)}")
    score_path = None
    score_input_path = None
    if args.phase == "full" and args.scorer:
        score_path = result_path.with_suffix(".score.json")
        score_input_path = result_path.with_suffix(".score-input.jsonl")
        with score_input_path.open("w", encoding="utf-8") as stream:
            for record in records[:cell.limit]:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        _score_predictions(
            args.scorer, str(score_input_path), _responses_path(result_path), score_path
        )
    return {
        **_cell_as_dict(cell),
        "attempt": attempt,
        "accepted": True,
        "summary": result["summary"],
        "metadata": metadata,
        "artifacts": {
            "result": str(result_path),
            "responses": str(_responses_path(result_path)),
            "server_log_gzip": str(compressed_log) if compressed_log else None,
            "resources": str(resource_path) if monitor else None,
            "config": str(config_path),
            "score": str(score_path) if score_path else None,
            "score_input": str(score_input_path) if score_input_path else None,
        },
    }


def _wait_for_cool_device(
    adb: Adb,
    timeout_s: float = DEFAULT_COOLDOWN_TIMEOUT_S,
) -> None:
    readings: list[float] = []
    deadline = time.monotonic() + timeout_s
    while True:
        readings.append(read_phone_temperature(adb))
        readings = readings[-7:]
        if len(readings) == 7 and max(readings) <= 32.0 and max(readings) - min(readings) <= 1.0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"device cooldown did not finish within {timeout_s:.0f} seconds")
        time.sleep(10)


def _speedup_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        (cell["backend"], cell["concurrency"]):
            cell["summary"]["aggregate_completion_tokens_per_s"]
        for cell in cells if cell["mode"] == "single_request" and cell.get("accepted")
    }
    rows = []
    for cell in cells:
        if not cell.get("accepted"):
            continue
        rate = cell["summary"]["aggregate_completion_tokens_per_s"]
        baseline = baselines.get((cell["backend"], cell["concurrency"]), 0.0)
        rows.append({
            "backend": cell["backend"],
            "mode": cell["mode"],
            "concurrency": cell["concurrency"],
            "requests": cell["summary"]["successful_requests"],
            "completion_tokens": cell["summary"]["total_completion_tokens"],
            "wall_time_s": cell["summary"]["wall_time_s"],
            "requests_per_s": cell["summary"]["requests_per_s"],
            "completion_tokens_per_s": rate,
            "speedup_vs_single": rate / baseline if baseline > 0 else None,
            "latency_p50_s": cell["summary"]["request_latency_s"]["p50"],
            "latency_p90_s": cell["summary"]["request_latency_s"]["p90"],
            "latency_p95_s": cell["summary"]["request_latency_s"]["p95"],
            "latency_max_s": cell["summary"]["request_latency_s"]["max"],
            "wave_tokens_p50": cell["summary"]["wave_tokens_per_s"]["p50"],
            "wave_tokens_p90": cell["summary"]["wave_tokens_per_s"]["p90"],
            "drift_percent": cell["summary"]["first_to_last_wave_drift_percent"],
            "start_temperature_c": cell["metadata"]["start_temperature_c"],
            "end_temperature_c": cell["metadata"]["end_temperature_c"],
        })
    return rows


def _write_matrix_summary(
    output_dir: Path,
    phase: str,
    cells: list[dict[str, Any]],
    inventory: dict[str, Any],
    error: str | None = None,
) -> tuple[Path, Path, Path]:
    rows = _speedup_rows(cells)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps({
        "phase": phase,
        "status": "failed" if error else "complete",
        "error": error,
        "inventory": inventory,
        "cells": cells,
        "comparison_rows": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0]) if rows else ["backend", "mode", "concurrency"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# Android Server Throughput ({phase})",
        "",
        "Cross-backend values use different model exports. Speedup is calculated only against "
        "`single_request` within the same backend and concurrency.",
        "",
    ]
    if error:
        lines.extend([f"Status: failed: `{error}`", ""])
    if rows:
        lines.extend([
            "| Backend | Mode | C | Requests | Completion tok/s | Requests/s | Speedup vs single | Drift |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            speedup = row["speedup_vs_single"]
            drift = row["drift_percent"]
            lines.append(
                f"| {row['backend']} | {row['mode']} | {row['concurrency']} | {row['requests']} | "
                f"{row['completion_tokens_per_s']:.3f} | {row['requests_per_s']:.4f} | "
                f"{speedup:.3f}x | {drift:.2f}% |"
                if speedup is not None and drift is not None else
                f"| {row['backend']} | {row['mode']} | {row['concurrency']} | {row['requests']} | "
                f"{row['completion_tokens_per_s']:.3f} | {row['requests_per_s']:.4f} | n/a | n/a |"
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path


def _run_matrix_live(args: argparse.Namespace) -> int:
    binary = Path(args.binary).resolve()
    dataset = Path(args.dataset).resolve()
    if not binary.is_file():
        raise ValueError(f"binary does not exist: {binary}")
    if not dataset.is_file():
        raise ValueError(f"dataset does not exist: {dataset}")
    if args.scorer and not Path(args.scorer).is_file():
        raise ValueError(f"scorer does not exist: {args.scorer}")
    dual_benchmark = getattr(args, "dual_benchmark", False)
    required_records = (
        8 if args.phase == "smoke" else 20
    ) if dual_benchmark else (8 if args.phase == "dev" else 100)
    prompt_field = getattr(args, "prompt_field", "prompt")
    records = load_dataset(dataset, required_records, prompt_field)
    adb = Adb(args.adb_serial)
    if adb.shell("getprop ro.product.cpu.abi") != "arm64-v8a":
        raise RuntimeError("throughput matrix requires an arm64-v8a Android device")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    host_base = _read_remote_json(adb, args.host_model_dir, ("config.json",))
    qnn_base = _read_remote_json(adb, args.qnn_model_dir, ("config_qnn.json", "config.json"))
    host_base["base_dir"] = args.host_model_dir.rstrip("/") + "/"
    qnn_base["base_dir"] = args.qnn_model_dir.rstrip("/") + "/"
    configs = build_matrix_configs(host_base, qnn_base)
    plan = (
        build_dual_benchmark_plan(configs, args.phase)
        if dual_benchmark else build_execution_plan(configs, args.phase)
    )

    inventory: dict[str, Any] = {
        "adb_serial": adb.serial,
        "before": _device_storage_inventory(adb, [
            args.host_model_dir, args.qnn_model_dir, args.qnn_runtime_dir,
        ]),
    }
    if inventory["before"]["data_available_bytes"] < MIN_DEVICE_FREE_BYTES:
        raise RuntimeError(
            f"/data has {inventory['before']['data_available_bytes']} bytes free; "
            f"at least {MIN_DEVICE_FREE_BYTES} required"
        )
    adb.shell(f"mkdir -p {shlex.quote(args.remote_stage)}")
    if binary.stat().st_size > MAX_DEVICE_STAGE_BYTES:
        raise RuntimeError("mnncli binary alone exceeds the 64 MiB staging limit")
    remote_binary = posixpath.join(args.remote_stage.rstrip("/"), "mnncli-throughput")
    adb.run("push", str(binary), remote_binary)
    adb.shell(f"chmod 755 {shlex.quote(remote_binary)}")
    inventory["staging_after_binary_bytes"] = _check_device_stage(adb, args.remote_stage)

    fingerprints = {
        "binary_sha256": _sha256_file(binary),
        "dataset_sha256": _sha256_file(dataset),
        "host_model_sha256": _remote_model_hashes(adb, args.host_model_dir, host_base),
        "qnn_model_sha256": _remote_model_hashes(adb, args.qnn_model_dir, qnn_base),
        "binary_path": str(binary),
        "dataset_path": str(dataset),
        "dataset_prompt_field": prompt_field,
        "adb_serial": adb.serial,
    }
    cells: list[dict[str, Any]] = []
    deferred_errors: list[str] = []
    matrix_error: str | None = None
    try:
        for cell in plan:
            max_attempts = 3 if args.phase == "full" else 1
            accepted = False
            for attempt in range(1, max_attempts + 1):
                try:
                    cell_result = _run_cell_attempt(
                        adb, args, cell, attempt, records, output_dir, fingerprints
                    )
                except OverheatError as exc:
                    cells.append({
                        **_cell_as_dict(cell), "attempt": attempt, "accepted": False,
                        "rejection_reason": str(exc),
                    })
                    if attempt == max_attempts:
                        raise
                    _wait_for_cool_device(adb)
                    continue
                except Exception as exc:
                    cells.append({
                        **_cell_as_dict(cell), "attempt": attempt, "accepted": False,
                        "rejection_reason": str(exc),
                        "artifacts": {
                            "result": str(output_dir / f"{cell.cell_id}-attempt{attempt}.json"),
                            "responses": str(
                                _responses_path(output_dir / f"{cell.cell_id}-attempt{attempt}.json")
                            ),
                            "server_log_gzip": str(
                                output_dir / f"{cell.cell_id}-attempt{attempt}.server.log.gz"
                            ),
                        },
                    })
                    if getattr(args, "continue_on_cell_failure", False):
                        deferred_errors.append(f"{cell.cell_id}: {exc}")
                        break
                    raise
                drift = cell_result["summary"]["first_to_last_wave_drift_percent"]
                if args.phase == "full" and drift is not None and abs(drift) > 10.0:
                    cell_result["accepted"] = False
                    cell_result["rejection_reason"] = (
                        f"first/last five-wave throughput drift is {drift:.2f}%"
                    )
                    cells.append(cell_result)
                    if attempt == max_attempts:
                        raise RuntimeError(
                            f"{cell.cell_id}: drift remained above 10% after {max_attempts} attempts"
                        )
                    _wait_for_cool_device(adb)
                    continue
                cells.append(cell_result)
                accepted = True
                break
            if not accepted:
                if getattr(args, "continue_on_cell_failure", False):
                    _check_device_stage(adb, args.remote_stage)
                    _check_host_results(output_dir)
                    continue
                raise RuntimeError(f"{cell.cell_id}: no accepted attempt")
            _check_device_stage(adb, args.remote_stage)
            _check_host_results(output_dir)
        if deferred_errors:
            raise RuntimeError("; ".join(deferred_errors))
    except BaseException as exc:
        matrix_error = str(exc)
        try:
            inventory["after_failure"] = _device_storage_inventory(adb, [
                args.host_model_dir, args.qnn_model_dir, args.qnn_runtime_dir,
            ])
        except Exception as inventory_error:
            inventory["after_failure_error"] = str(inventory_error)
        _write_matrix_summary(output_dir, args.phase, cells, inventory, matrix_error)
        raise

    inventory["after"] = _device_storage_inventory(adb, [
        args.host_model_dir, args.qnn_model_dir, args.qnn_runtime_dir,
    ])
    inventory["host_results_bytes"] = _check_host_results(output_dir)
    summary_json, _, _ = _write_matrix_summary(output_dir, args.phase, cells, inventory)
    remote_summary = posixpath.join(args.remote_stage.rstrip("/"), "throughput-summary.json")
    adb.run("push", str(summary_json), remote_summary)
    inventory["final_staging_bytes"] = _check_device_stage(adb, args.remote_stage)
    _write_matrix_summary(output_dir, args.phase, cells, inventory)
    _check_host_results(output_dir)
    adb.run("push", str(summary_json), remote_summary)
    print(f"wrote matrix results to {output_dir}")
    return 0


def run_load_command(args: argparse.Namespace) -> int:
    records = load_dataset(args.dataset, args.limit)
    if args.warmup:
        if args.warmup > len(records):
            raise ValueError("warmup exceeds the loaded dataset size")
        if args.warmup % args.concurrency:
            raise ValueError("warmup would create a partial wave")
        run_load(
            args.base_url, records[:args.warmup], args.max_tokens,
            args.concurrency, args.timeout,
        )
    result = run_load(
        args.base_url, records, args.max_tokens, args.concurrency, args.timeout
    )
    write_load_result(args.output, result, {
        "base_url": args.base_url,
        "dataset": str(Path(args.dataset).resolve()),
        "limit": args.limit,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
    })
    print(json.dumps(result["summary"], indent=2))
    return 0 if result["summary"]["failed_requests"] == 0 else 1


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    load_parser = subparsers.add_parser("load", help="run fixed-wave HTTP load")
    load_parser.add_argument("--base-url", required=True)
    load_parser.add_argument("--dataset", required=True)
    load_parser.add_argument("--limit", type=int, required=True)
    load_parser.add_argument("--max-tokens", type=int, required=True)
    load_parser.add_argument("--concurrency", type=int, required=True)
    load_parser.add_argument("--warmup", type=int, default=0)
    load_parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    load_parser.add_argument("--output", required=True)
    load_parser.set_defaults(func=run_load_command)
    matrix_parser = subparsers.add_parser("matrix", help="run the Android backend/mode matrix")
    matrix_parser.add_argument("--adb-serial")
    matrix_parser.add_argument("--binary", required=True)
    matrix_parser.add_argument("--dataset", required=True)
    matrix_parser.add_argument(
        "--host-model-dir", default="/data/local/tmp/Qwen3-1.7B-MNN-int4"
    )
    matrix_parser.add_argument(
        "--qnn-model-dir", default="/data/local/tmp/qnn_fixed_s1_8_128"
    )
    matrix_parser.add_argument("--qnn-runtime-dir", default="/data/local/tmp/mnn-qnn")
    matrix_parser.add_argument(
        "--remote-stage", default="/data/local/tmp/mnn-server-throughput"
    )
    matrix_parser.add_argument("--phase", choices=("dev", "full"), default="dev")
    matrix_parser.add_argument("--output-dir", required=True)
    matrix_parser.add_argument("--dry-run", action="store_true")
    matrix_parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    matrix_parser.add_argument("--readiness-timeout", type=float, default=900.0)
    matrix_parser.add_argument("--scorer", help="optional path to the legacy gsm8k_eval.py")
    matrix_parser.set_defaults(func=run_matrix_command)
    dual_parser = subparsers.add_parser(
        "dual-bench", help="run protected QNN dual-pipeline C6/C8 tests"
    )
    dual_parser.add_argument("--adb-serial")
    dual_parser.add_argument("--binary", required=True)
    dual_parser.add_argument("--dataset", required=True)
    dual_parser.add_argument("--prompt-field", default="prompt")
    dual_parser.add_argument(
        "--host-model-dir", default="/data/local/tmp/Qwen3-1.7B-MNN-int4"
    )
    dual_parser.add_argument(
        "--qnn-model-dir", default="/data/local/tmp/qnn_fixed_s1_8_128"
    )
    dual_parser.add_argument("--qnn-runtime-dir", default="/data/local/tmp/mnn-qnn")
    dual_parser.add_argument(
        "--remote-stage", default="/data/local/tmp/mnn-server-throughput"
    )
    dual_parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    dual_parser.add_argument("--output-dir", required=True)
    dual_parser.add_argument("--dry-run", action="store_true")
    dual_parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    dual_parser.add_argument("--readiness-timeout", type=float, default=900.0)
    dual_parser.add_argument("--scorer")
    dual_parser.add_argument(
        "--disable-resource-guard",
        action="store_true",
        help="disable thermal, memory, RSS, and ADB automatic stop checks",
    )
    dual_parser.set_defaults(
        func=run_dual_benchmark_command,
        allow_partial_final_wave=True,
        continue_on_cell_failure=True,
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    try:
        return args.func(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
