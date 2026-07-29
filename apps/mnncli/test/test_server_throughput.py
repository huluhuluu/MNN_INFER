#!/usr/bin/env python3

import io
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock
from contextlib import redirect_stdout
from urllib.error import HTTPError

import server_throughput as throughput


def completion_response(prompt_tokens=11, completion_tokens=4, content="answer"):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class DatasetTest(unittest.TestCase):
    def test_load_dataset_reads_prompt_and_preserves_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                '{"id":"a","prompt":"first"}\n\n{"id":7,"prompt":"second"}\n',
                encoding="utf-8",
            )

            records = throughput.load_dataset(path, limit=1)

        self.assertEqual(records, [{"id": "a", "prompt": "first"}])

    def test_load_dataset_can_select_question_as_prompt_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                '{"id":7,"question":"short","prompt":"long"}\n',
                encoding="utf-8",
            )

            records = throughput.load_dataset(path, limit=1, prompt_field="question")

        self.assertEqual(records[0]["id"], 7)
        self.assertEqual(records[0]["prompt"], "short")
        self.assertEqual(records[0]["question"], "short")

    def test_load_dataset_reports_line_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"prompt":\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"bad\.jsonl:1: invalid JSON"):
                throughput.load_dataset(path, limit=0)

    def test_load_dataset_rejects_missing_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"id":1,"question":"missing"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"bad\.jsonl:1: prompt"):
                throughput.load_dataset(path, limit=0)


class LoadRunnerTest(unittest.TestCase):
    def test_four_requests_cross_worker_barrier_before_transport(self):
        entered = []
        lock = threading.Lock()
        all_entered = threading.Event()

        def request_fn(base_url, record, max_tokens, timeout):
            with lock:
                entered.append((record["id"], time.monotonic()))
                if len(entered) == 4:
                    all_entered.set()
            if not all_entered.wait(1):
                raise AssertionError("requests did not enter transport concurrently")
            return 200, completion_response(content=str(record["id"]))

        result = throughput.run_load(
            base_url="http://127.0.0.1:8000",
            records=[{"id": index, "prompt": f"p{index}"} for index in range(4)],
            max_tokens=4,
            concurrency=4,
            timeout=1,
            request_fn=request_fn,
        )

        self.assertEqual(result["summary"]["successful_requests"], 4)
        self.assertLess(max(item[1] for item in entered) - min(item[1] for item in entered), 0.1)

    def test_partial_wave_is_rejected_before_requests_are_sent(self):
        called = False

        def request_fn(*args):
            nonlocal called
            called = True
            return 200, completion_response()

        with self.assertRaisesRegex(ValueError, "partial wave"):
            throughput.run_load(
                "http://localhost:8000",
                [{"id": index, "prompt": "p"} for index in range(5)],
                max_tokens=4,
                concurrency=4,
                timeout=1,
                request_fn=request_fn,
            )
        self.assertFalse(called)

    def test_partial_final_wave_can_be_enabled(self):
        result = throughput.run_load(
            "http://localhost:8000",
            [{"id": index, "prompt": "p"} for index in range(5)],
            max_tokens=4,
            concurrency=4,
            timeout=1,
            request_fn=lambda *args: (200, completion_response()),
            allow_partial_final_wave=True,
        )

        self.assertEqual(result["summary"]["successful_requests"], 5)
        self.assertEqual([wave["request_count"] for wave in result["waves"]], [4, 1])

    def test_timeout_and_http_error_are_recorded(self):
        def request_fn(base_url, record, max_tokens, timeout):
            if record["id"] == 0:
                raise socket.timeout("late")
            raise HTTPError(base_url, 503, "busy", {}, io.BytesIO(b'{"error":"busy"}'))

        result = throughput.run_load(
            "http://localhost:8000",
            [{"id": 0, "prompt": "a"}, {"id": 1, "prompt": "b"}],
            max_tokens=4,
            concurrency=2,
            timeout=1,
            request_fn=request_fn,
        )

        self.assertEqual(result["summary"]["failed_requests"], 2)
        self.assertEqual(result["summary"]["timeout_errors"], 1)
        self.assertEqual(result["summary"]["http_errors"], 1)
        self.assertEqual({record["status"] for record in result["responses"]}, {"error"})

    def test_invalid_response_json_is_a_protocol_error(self):
        def request_fn(*args):
            raise json.JSONDecodeError("bad response", "{", 1)

        result = throughput.run_load(
            "http://localhost:8000", [{"id": 0, "prompt": "a"}],
            max_tokens=4, concurrency=1, timeout=1, request_fn=request_fn,
        )

        self.assertEqual(result["summary"]["protocol_errors"], 1)

    def test_empty_output_and_zero_usage_fail_validation(self):
        def request_fn(base_url, record, max_tokens, timeout):
            return 200, completion_response(0, 0, "")

        result = throughput.run_load(
            "http://localhost:8000",
            [{"id": 0, "prompt": "a"}],
            max_tokens=4,
            concurrency=1,
            timeout=1,
            request_fn=request_fn,
        )

        self.assertEqual(result["summary"]["empty_outputs"], 1)
        self.assertEqual(result["summary"]["zero_usage_responses"], 1)
        self.assertEqual(result["summary"]["successful_requests"], 0)

    def test_scorer_status_distinguishes_token_limit_from_early_stop(self):
        def request_fn(base_url, record, max_tokens, timeout):
            tokens = 4 if record["id"] == 0 else 3
            return 200, completion_response(completion_tokens=tokens)

        result = throughput.run_load(
            "http://localhost:8000",
            [{"id": 0, "prompt": "a"}, {"id": 1, "prompt": "b"}],
            max_tokens=4,
            concurrency=2,
            timeout=1,
            request_fn=request_fn,
        )

        self.assertEqual(
            [record["status"] for record in result["responses"]],
            ["max_tokens_finished", "normal_finished"],
        )


class StatisticsTest(unittest.TestCase):
    def test_aggregate_statistics_use_successful_tokens_and_wall_time(self):
        responses = [
            {"ok": True, "latency_s": 1.0, "completion_tokens": 10, "wave_index": 0},
            {"ok": True, "latency_s": 2.0, "completion_tokens": 20, "wave_index": 0},
            {"ok": False, "latency_s": 9.0, "completion_tokens": 99, "wave_index": 1,
             "error_type": "timeout"},
        ]
        waves = [{"wave_index": 0, "wall_time_s": 2.0, "completion_tokens": 30,
                  "successful_requests": 2}]

        summary = throughput.summarize(responses, waves, wall_time_s=3.0)

        self.assertEqual(summary["successful_requests"], 2)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["total_completion_tokens"], 30)
        self.assertAlmostEqual(summary["aggregate_completion_tokens_per_s"], 10.0)
        self.assertAlmostEqual(summary["request_latency_s"]["p50"], 1.5)
        self.assertAlmostEqual(summary["wave_tokens_per_s"]["p50"], 15.0)

    def test_timing_distributions_use_real_model_fields(self):
        responses = [
            {"ok": True, "latency_s": 2.0, "completion_tokens": 4, "wave_index": 0,
             "model_ttft_ms": 100.0, "model_tpot_ms": 20.0,
             "model_latency_ms": 160.0, "queue_ms": 10.0, "e2e_ttft_ms": 110.0},
            {"ok": True, "latency_s": 3.0, "completion_tokens": 4, "wave_index": 0,
             "model_ttft_ms": 200.0, "model_tpot_ms": 40.0,
             "model_latency_ms": 320.0, "queue_ms": 20.0, "e2e_ttft_ms": 220.0},
        ]

        summary = throughput.summarize(responses, [], wall_time_s=3.0)

        self.assertEqual(summary["model_ttft_ms"]["p50"], 150.0)
        self.assertEqual(summary["model_tpot_ms"]["p50"], 30.0)
        self.assertEqual(summary["model_latency_ms"]["max"], 320.0)
        self.assertEqual(summary["queue_ms"]["p50"], 15.0)
        self.assertEqual(summary["e2e_ttft_ms"]["p50"], 165.0)


class TimingResponseTest(unittest.TestCase):
    def test_success_record_preserves_server_timing_and_per_task_rates(self):
        payload = completion_response(prompt_tokens=10, completion_tokens=4, content="answer")
        payload["timings"] = {
            "queue_ms": 5.0,
            "model_ttft_ms": 100.0,
            "model_tpot_ms": 20.0,
            "model_latency_ms": 160.0,
            "e2e_ttft_ms": 105.0,
            "e2e_latency_ms": 200.0,
        }

        record = throughput._success_record(
            {"id": 7, "prompt": "p"}, 0, 0, 0.2, 200, payload, 4
        )

        self.assertEqual(record["model_ttft_ms"], 100.0)
        self.assertEqual(record["model_tpot_ms"], 20.0)
        self.assertAlmostEqual(record["decode_tokens_per_s"], 50.0)
        self.assertAlmostEqual(record["e2e_completion_tokens_per_s"], 20.0)


class ResourceGuardTest(unittest.TestCase):
    def test_resource_guard_stops_on_temperature_memory_or_rss(self):
        healthy = {
            "temperature_c": 35.0,
            "mem_total_bytes": 16 * 1024**3,
            "mem_available_bytes": 8 * 1024**3,
            "process_rss_bytes": 4 * 1024**3,
        }

        self.assertIsNone(throughput.resource_guard_reason(healthy))
        self.assertIn("temperature", throughput.resource_guard_reason({**healthy, "temperature_c": 42.1}))
        self.assertIn("available memory", throughput.resource_guard_reason(
            {**healthy, "mem_available_bytes": 2 * 1024**3 - 1}
        ))
        self.assertIn("RSS", throughput.resource_guard_reason(
            {**healthy, "process_rss_bytes": int(16 * 1024**3 * 0.81)}
        ))


class HarnessTimeoutTest(unittest.TestCase):
    def test_adb_commands_have_a_default_timeout(self):
        completed = mock.Mock(stdout="", returncode=0)
        with mock.patch.object(throughput.subprocess, "run", return_value=completed) as run:
            throughput.Adb("serial").run("shell", "true")

        self.assertEqual(run.call_args.kwargs["timeout"], throughput.DEFAULT_ADB_TIMEOUT_S)

    def test_thermal_gate_has_a_total_timeout(self):
        with mock.patch.object(throughput, "read_phone_temperature", return_value=33.0), \
                mock.patch.object(throughput.time, "sleep"):
            with self.assertRaisesRegex(TimeoutError, "thermal gate"):
                throughput.wait_for_thermal_gate(mock.Mock(), mock.Mock(), timeout_s=0)

    def test_cooldown_has_a_total_timeout(self):
        with mock.patch.object(throughput, "read_phone_temperature", return_value=33.0), \
                mock.patch.object(throughput.time, "sleep"):
            with self.assertRaisesRegex(TimeoutError, "cooldown"):
                throughput._wait_for_cool_device(mock.Mock(), timeout_s=0)

    def test_monitor_close_fails_if_worker_does_not_exit(self):
        monitor = throughput.TemperatureMonitor(mock.Mock(), mock.Mock(), 123)
        monitor._thread = mock.Mock()
        monitor._thread.is_alive.return_value = True

        with self.assertRaisesRegex(RuntimeError, "resource monitor"):
            monitor.close()


class MatrixConfigTest(unittest.TestCase):
    def test_matrix_builds_nine_greedy_low_precision_configs(self):
        host = {"llm_model": "llm.mnn", "llm_weight": "llm.mnn.weight"}
        qnn = {"llm_model": "qnn/llm.mnn", "chunk_limits": [128, 8, 1]}

        configs = throughput.build_matrix_configs(host, qnn)

        self.assertEqual(len(configs), 9)
        by_key = {(item.backend, item.mode): item.config for item in configs}
        self.assertEqual(by_key[("cpu", "single_request")]["backend_type"], "cpu")
        self.assertEqual(by_key[("opencl", "single_request")]["backend_type"], "opencl")
        self.assertEqual(by_key[("qnn", "single_request")]["thread_num"], 1)
        self.assertFalse(by_key[("cpu", "single_request")]["packed_attention_mode"])
        self.assertTrue(by_key[("cpu", "continuous_batch")]["packed_attention_mode"])
        self.assertEqual(by_key[("qnn", "dual_pipeline")]["dual_pipeline_split_count"], 2)
        self.assertEqual(by_key[("qnn", "dual_pipeline")]["dual_pipeline_max_resident_graphs"], 30)
        self.assertEqual(by_key[("qnn", "dual_pipeline")]["dual_pipeline_prefetch_window"], 2)
        for config in by_key.values():
            self.assertEqual(config["sampler_type"], "greedy")
            self.assertEqual(config["precision"], "low")
            self.assertEqual(config["memory"], "low")
            self.assertEqual(config["power"], "high")

    def test_dev_and_full_execution_plans_use_rotated_mode_order(self):
        host = {"llm_model": "llm.mnn", "llm_weight": "llm.mnn.weight"}
        qnn = {"llm_model": "qnn/llm.mnn"}
        configs = throughput.build_matrix_configs(host, qnn)

        dev = throughput.build_execution_plan(configs, "dev")
        full = throughput.build_execution_plan(configs, "full")

        self.assertEqual(len(dev), 9)
        self.assertTrue(all((cell.limit, cell.max_tokens, cell.concurrency, cell.warmup) ==
                            (8, 16, 4, 4) for cell in dev))
        self.assertTrue(all(cell.trace for cell in dev))
        self.assertEqual(
            [(cell.backend, cell.mode) for cell in dev],
            [
                ("cpu", "single_request"),
                ("cpu", "continuous_batch"),
                ("cpu", "dual_pipeline"),
                ("opencl", "continuous_batch"),
                ("opencl", "dual_pipeline"),
                ("opencl", "single_request"),
                ("qnn", "dual_pipeline"),
                ("qnn", "single_request"),
                ("qnn", "continuous_batch"),
            ],
        )
        self.assertEqual(len(full), 18)
        self.assertEqual(
            [(cell.concurrency, cell.limit, cell.warmup) for cell in full[:2]],
            [(1, 20, 1), (4, 100, 4)],
        )
        self.assertTrue(all(cell.max_tokens == 512 and not cell.trace for cell in full))

    def test_dual_benchmark_plan_uses_twenty_requests_per_cell(self):
        configs = throughput.build_matrix_configs(
            {"llm_model": "llm.mnn"}, {"llm_model": "qnn/llm.mnn"}
        )

        smoke = throughput.build_dual_benchmark_plan(configs, "smoke")
        full = throughput.build_dual_benchmark_plan(configs, "full")

        self.assertEqual(
            [(cell.concurrency, cell.limit, cell.max_tokens, cell.warmup, cell.trace)
             for cell in smoke],
            [(6, 6, 16, 6, True), (8, 8, 16, 8, True)],
        )
        self.assertEqual(
            [(cell.concurrency, cell.limit, cell.max_tokens, cell.warmup, cell.trace)
             for cell in full],
            [(6, 20, 512, 6, False), (8, 20, 512, 8, False)],
        )
        self.assertTrue(all(cell.backend == "qnn" and cell.mode == "dual_pipeline"
                            for cell in smoke + full))


class AcceptanceTraceTest(unittest.TestCase):
    def test_single_continuous_and_host_dual_require_expected_scheduling(self):
        single = "\n".join([
            "event=request_started request_id=1 request_scope=service mode=single_request",
            "event=request_completed request_id=1 request_scope=service status=ok",
            "event=request_started request_id=2 request_scope=service mode=single_request",
            "event=request_completed request_id=2 request_scope=service status=ok",
        ])
        continuous = "event=batch_started mode=continuous_batch request_count=4 delivery=batch_completion"
        dual = "\n".join([
            "event=batch_started mode=dual_pipeline request_count=4 delivery=batch_completion",
            "event=lane_owner request_id=0 request_scope=engine lane=0 segment=0 chunk_index=0",
            "event=lane_owner request_id=1 request_scope=engine lane=1 segment=0 chunk_index=0",
            "event=stage_summary host_completed=12 qnn_completed=0 overlap_grants=0 max_host=1 max_qnn=0 cancelled=0",
        ])

        self.assertEqual(throughput.validate_server_log(single, "cpu", "single_request", True), [])
        self.assertEqual(
            throughput.validate_server_log(continuous, "opencl", "continuous_batch", True), []
        )
        self.assertEqual(throughput.validate_server_log(dual, "cpu", "dual_pipeline", True), [])

    def test_qnn_dual_requires_qnn_work_and_bounded_stages(self):
        valid = "\n".join([
            "event=batch_started mode=dual_pipeline request_count=4 delivery=batch_completion",
            "event=lane_owner request_id=0 request_scope=engine lane=0 segment=0 chunk_index=0",
            "event=lane_owner request_id=1 request_scope=engine lane=1 segment=0 chunk_index=0",
            "event=stage_summary host_completed=12 qnn_completed=14 overlap_grants=5 max_host=1 max_qnn=1 cancelled=0",
        ])
        invalid = valid.replace("qnn_completed=14", "qnn_completed=0").replace("max_host=1", "max_host=2")

        self.assertEqual(throughput.validate_server_log(valid, "qnn", "dual_pipeline", True), [])
        errors = throughput.validate_server_log(invalid, "qnn", "dual_pipeline", True)
        self.assertTrue(any("QNN stage" in error for error in errors))
        self.assertTrue(any("max_host" in error for error in errors))

    def test_runtime_error_markers_fail_even_without_trace(self):
        errors = throughput.validate_server_log(
            "QNN graphExecute failed with code 1002\nINTERNAL_ERROR", "qnn", "single_request", False
        )

        self.assertTrue(any("INTERNAL_ERROR" in error for error in errors))
        self.assertTrue(any("QNN" in error for error in errors))

        numeric_errors = throughput.validate_server_log(
            "QNN backend deviceCreate=14001", "qnn", "single_request", False
        )
        self.assertTrue(any("QNN" in error for error in numeric_errors))


class MatrixOrchestrationTest(unittest.TestCase):
    def test_dual_benchmark_can_disable_resource_guard(self):
        args = throughput.create_parser().parse_args([
            "dual-bench",
            "--binary", "/tmp/mnncli",
            "--dataset", "/tmp/data.jsonl",
            "--phase", "full",
            "--output-dir", "/tmp/results",
            "--disable-resource-guard",
        ])

        self.assertFalse(throughput.resource_guard_enabled(args))
        self.assertTrue(args.allow_partial_final_wave)
        self.assertTrue(args.continue_on_cell_failure)

    def test_optional_monitor_close_accepts_none(self):
        throughput.close_optional_monitor(None)

    def test_dry_run_never_constructs_adb_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "mnncli"
            binary.write_bytes(b"binary")
            dataset = root / "gsm8k.jsonl"
            dataset.write_text(
                "".join(json.dumps({"id": index, "prompt": "p"}) + "\n" for index in range(100)),
                encoding="utf-8",
            )
            args = throughput.create_parser().parse_args([
                "matrix",
                "--adb-serial", "must-not-connect",
                "--binary", str(binary),
                "--dataset", str(dataset),
                "--host-model-dir", "/host",
                "--qnn-model-dir", "/qnn",
                "--qnn-runtime-dir", "/runtime",
                "--remote-stage", "/stage",
                "--phase", "full",
                "--output-dir", str(root / "results"),
                "--dry-run",
            ])

            with mock.patch.object(throughput, "Adb", side_effect=AssertionError("ADB used")):
                with redirect_stdout(io.StringIO()):
                    status = throughput.run_matrix_command(args)

        self.assertEqual(status, 0)

    def test_remote_server_cleanup_targets_only_owned_pid_and_forward(self):
        class Result:
            def __init__(self, returncode=0):
                self.returncode = returncode
                self.stdout = ""

        class FakeAdb:
            def __init__(self):
                self.calls = []

            def run(self, *arguments, check=True, **kwargs):
                self.calls.append(arguments)
                if arguments[:2] == ("shell", "kill -0 321"):
                    return Result(1)
                return Result()

        adb = FakeAdb()
        server = throughput.RemoteServer(
            adb=adb,
            remote_stage="/data/local/tmp/stage",
            local_log=Path("unused.log"),
            cell_id="cpu-single-c4-a1",
            qnn_runtime_dir=None,
            trace=False,
        )
        server.pid = 321
        server.local_port = 19001
        server.remote_config = "/data/local/tmp/stage/throughput-cpu-single-c4-a1.json"
        server.remote_pid_file = "/data/local/tmp/stage/throughput-cpu-single-c4-a1.pid"

        server.close()

        flattened = "\n".join(" ".join(call) for call in adb.calls)
        self.assertIn("shell kill 321", flattened)
        self.assertIn("forward --remove tcp:19001", flattened)
        self.assertIn(
            "shell rm -f /data/local/tmp/stage/throughput-cpu-single-c4-a1.json "
            "/data/local/tmp/stage/throughput-cpu-single-c4-a1.pid",
            flattened,
        )
        self.assertNotIn("pkill", flattened)

    def test_remote_server_reports_pid_before_waiting_for_readiness(self):
        class Result:
            def __init__(self, returncode=0, stdout=""):
                self.returncode = returncode
                self.stdout = stdout

        class Process:
            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        class FakeAdb:
            def run(self, *arguments, check=True, **kwargs):
                if arguments[:2] == ("shell", "cat /stage/throughput-cell.pid"):
                    return Result(stdout="321\n")
                return Result()

            def popen_shell(self, command, log_stream):
                return Process()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}", encoding="utf-8")
            server = throughput.RemoteServer(
                adb=FakeAdb(), remote_stage="/stage", local_log=root / "server.log",
                cell_id="cell", qnn_runtime_dir=None, trace=False,
            )
            events = []
            server._wait_ready = lambda base_url, timeout: events.append("readiness")

            server.start(config, 1, lambda pid: events.append(f"pid:{pid}"))
            server.close()

        self.assertEqual(events, ["pid:321", "readiness"])


if __name__ == "__main__":
    unittest.main()
