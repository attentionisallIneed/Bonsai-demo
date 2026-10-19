"""Offline SSE, fairness and reporting checks; no server or GPU is required."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


spec = importlib.util.spec_from_file_location(
    "benchmark_throughput", Path(__file__).resolve().parents[1] / "scripts/benchmark_throughput.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def config(**overrides):
    values = dict(base_url="http://localhost:8080", model="test", api_key="test-secret",
                  max_tokens=128, temperature=0.0, cache="cold", budget=None,
                  timeout=10, requests=4, warmup=0, seed=42, no_gpu_info=True,
                  server_binary=None, needle_words=None, prompt_file=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def chunk(content=None, reasoning=None, finish=None, **extra):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {"choices": [{"delta": delta, "finish_reason": finish}], **extra}


class Response:
    status = 200

    def __init__(self, events, clock=None, eof_delay=0):
        self.events = events
        self.clock = clock
        self.eof_delay = eof_delay

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        for event in self.events:
            if self.clock is not None:
                self.clock[0] += 1
            if isinstance(event, Exception):
                raise event
            value = event if isinstance(event, str) else json.dumps(event, ensure_ascii=False)
            yield ("data: " + value + "\r\n").encode("utf-8")
            yield b"\r\n"
        if self.clock is not None:
            self.clock[0] += self.eof_delay


class StreamingChecks(unittest.TestCase):
    def request(self, events, settings=None, case=None):
        with patch.object(benchmark.urllib.request, "urlopen", return_value=Response(events)):
            return benchmark.send_chat_request(0, case or {"id": "q", "prompt": "Question?"},
                                               "medium", settings or config(), 42)

    def test_times_reasoning_answer_and_full_body_separately(self):
        clock = [0]
        events = [chunk(), chunk(reasoning="想一想"), chunk(content="答复"),
                  chunk(finish="stop", timings={"predicted_n": 5, "predicted_per_second": 10}),
                  {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}}, "[DONE]"]
        with patch.object(benchmark.time, "perf_counter", side_effect=lambda: clock[0]), \
                patch.object(benchmark.urllib.request, "urlopen", return_value=Response(events, clock, 7)) as opener:
            result = benchmark.send_chat_request(0, {"id": "q", "prompt": "Question?"}, "medium", config(), 42)
        self.assertEqual(result["ttft_sec"], 2)
        self.assertEqual(result["first_content_sec"], 3)
        self.assertEqual(result["latency_sec"], 13)  # Includes all events + EOF delay.
        self.assertTrue(result["success"] and result["complete"] and result["body_complete"])
        self.assertEqual(result["reasoning_chars"], 3)
        self.assertEqual(result["content_chars"], 2)
        self.assertEqual(result["predicted_tokens"], 5)
        self.assertEqual(result["timings"]["predicted_per_second"], 10)
        self.assertEqual(opener.call_args.args[0].headers["Authorization"], "Bearer test-secret")
        payload = json.loads(opener.call_args.args[0].data)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True, "reasoning_effort": "medium"})
        self.assertNotIn("reasoning_effort", payload)

    def test_length_is_successful_transport_but_incomplete_answer(self):
        result = self.request([chunk(reasoning="still thinking"), chunk(finish="length"), "[DONE]"])
        self.assertTrue(result["success"])
        self.assertTrue(result["truncated"])
        self.assertFalse(result["complete"])
        self.assertIsNone(result["first_content_sec"])

    def test_missing_done_or_finish_is_detected(self):
        for events in ([chunk(content="answer", finish="stop")], [chunk(content="answer"), "[DONE]"]):
            with self.subTest(events=events):
                result = self.request(events)
                self.assertFalse(result["success"])
                self.assertTrue(result["body_complete"])
                self.assertIn("incomplete SSE", result["error"])

    def test_partial_stream_keeps_text_and_reports_error(self):
        result = self.request([chunk(content="partial"), OSError("socket broke")])
        self.assertEqual(result["content"], "partial")
        self.assertFalse(result["success"] or result["body_complete"])
        self.assertIn("socket broke", result["error"])

    def test_api_and_json_errors_are_not_silent_success(self):
        for value in ({"error": {"message": "busy"}}, "malformed"):
            result = self.request([value])
            self.assertFalse(result["success"])
            self.assertIsNotNone(result["error"])

    def test_http_error_body_is_saved_and_secret_redacted(self):
        error = HTTPError("http://localhost", 401, "Unauthorized", {}, io.BytesIO(b"bad key: test-secret"))
        with patch.object(benchmark.urllib.request, "urlopen", side_effect=error):
            result = benchmark.send_chat_request(0, {"id": "q", "prompt": "Hi"}, "off", config(), 42)
        self.assertEqual(result["http_status"], 401)
        self.assertIn("bad key", result["error"])
        self.assertNotIn("test-secret", json.dumps(result))

    def test_tool_call_fragments_and_no_content_completion(self):
        events = [chunk(), chunk(), chunk(finish="tool_calls"), "[DONE]"]
        events[0]["choices"][0]["delta"] = {"tool_calls": [{"index": 0, "id": "call0", "type": "function",
            "function": {"name": "lookup", "arguments": '{"x":'}}]}
        events[1]["choices"][0]["delta"] = {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}
        result = self.request(events)
        self.assertTrue(result["complete"])
        self.assertEqual(result["tool_calls"][0]["function"], {"name": "lookup", "arguments": '{"x":1}'})

    def test_retrieval_checks_answer_only_and_requires_completion(self):
        case = {"id": "needle", "prompt": "find secret", "expected": {"secret": "abc123"}}
        thought_only = self.request([chunk(reasoning="abc123", content="missing", finish="stop"), "[DONE]"], case=case)
        self.assertFalse(thought_only["retrieval_passed"])
        good = self.request([chunk(content="abc123", finish="stop"), "[DONE]"], case=case)
        self.assertTrue(good["retrieval_passed"])
        short = self.request([chunk(content="abc123", finish="length"), "[DONE]"], case=case)
        self.assertFalse(short["retrieval_passed"])

    def test_sse_comments_multiline_and_unseparated_final_frame(self):
        raw = io.BytesIO(b': heartbeat\r\nevent: message\r\ndata: {"choices":\r\ndata: []}\r\n\r\ndata: [DONE]')
        values = list(benchmark.iter_sse(raw))
        self.assertEqual(json.loads(values[0]), {"choices": []})
        self.assertEqual(values[1], "[DONE]")


class BenchmarkChecks(unittest.TestCase):
    def fake_result(self, req_id, case, mode, settings, seed, marker=None, max_tokens=None):
        result = {"req_id": req_id, "prompt_id": case["id"], "seed": seed, "marker": marker,
                  "success": True, "complete": True, "truncated": False, "finish_reason": "stop",
                  "latency_sec": 1, "ttft_sec": 0.1, "first_content_sec": 0.2,
                  "predicted_tokens": 4, "timings": {"predicted_per_second": 20}}
        return result

    def test_identical_question_count_order_and_seed_across_concurrency(self):
        cases = [{"id": str(i), "prompt": str(i)} for i in range(4)]
        with patch.object(benchmark, "send_chat_request", side_effect=self.fake_result):
            one = benchmark.run_benchmark_cell(config(requests=8), cases, "off", 1, 0, "run")
            four = benchmark.run_benchmark_cell(config(requests=8), cases, "off", 4, 0, "run")
        self.assertEqual(len(one["results"]), 8)
        self.assertEqual([r["prompt_id"] for r in one["results"]], [r["prompt_id"] for r in four["results"]])
        self.assertEqual([r["seed"] for r in one["results"]], [r["seed"] for r in four["results"]])
        markers = [r["marker"] for r in one["results"] + four["results"]]
        self.assertEqual(len(set(markers)), 16)

    def test_warm_mode_primes_each_scheduled_prompt_and_excludes_primes(self):
        cases = [{"id": str(i), "prompt": str(i)} for i in range(4)]
        with patch.object(benchmark, "send_chat_request", side_effect=self.fake_result) as sender:
            cell = benchmark.run_benchmark_cell(config(cache="warm", requests=8), cases, "medium", 2, 0, "run")
        self.assertEqual(sender.call_count, 12)
        self.assertEqual(len(cell["warmup_results"]), 4)
        self.assertTrue(all(call.kwargs.get("max_tokens") == 1 for call in sender.call_args_list[:4]))
        self.assertEqual(cell["summary"]["total_requests"], 8)
        self.assertTrue(all(r["marker"] is None for r in cell["results"]))

    def test_summary_discloses_truncation_and_missing_metrics(self):
        good = self.fake_result(0, {"id": "a"}, "off", config(), 42)
        truncated = dict(good, req_id=1, complete=False, truncated=True, finish_reason="length", latency_sec=3)
        failed = dict(good, req_id=2, success=False, complete=False, finish_reason=None, predicted_tokens=None)
        summary = benchmark.summarize([good, truncated, failed], 2)
        self.assertEqual(summary["system_throughput_tps"], 4)
        self.assertEqual(summary["latency_sec"]["p50"], 2)
        self.assertAlmostEqual(summary["latency_sec"]["p95"], 2.9)
        self.assertEqual(summary["complete_latency_sec"]["p50"], 1)
        self.assertEqual(summary["truncated_count"], 1)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["finish_reasons"], {"stop": 1, "length": 1, "missing": 1})
        unknown = benchmark.summarize([dict(good, predicted_tokens=None)], 2)
        self.assertIsNone(unknown["system_throughput_tps"])
        self.assertIsNone(benchmark.distribution([])["p95"])

    def test_effort_budget_and_cold_marker_payload(self):
        cold = benchmark.make_payload("Question", "medium", config(budget=100), 12, "unique")
        self.assertFalse(cold["cache_prompt"])
        self.assertIn("unique", cold["messages"][0]["content"])
        self.assertEqual(cold["thinking_budget_tokens"], 100)
        off = benchmark.make_payload("Question", "off", config(cache="warm", budget=100), 12)
        self.assertEqual(off["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(off["thinking_budget_tokens"], 0)
        self.assertTrue(off["cache_prompt"])

    def test_metadata_only_collects_safe_props_subset(self):
        props = {"total_slots": 1, "model_path": "model.gguf", "chat_template": "private template",
                 "webui_settings": {"secret": "hidden"}, "systemMessage": "private system",
                 "default_generation_settings": {"n_ctx": 65536, "params": {"temperature": 0.6,
                     "prompt": "private prompt", "api_key": "another-secret"}}}
        with patch.object(benchmark, "get_json", return_value=props):
            result = benchmark.metadata(config())
        serialized = json.dumps(result)
        self.assertNotIn("private", serialized)
        self.assertNotIn("secret", serialized)
        self.assertEqual(result["props"]["default_generation_settings"]["n_ctx"], 65536)

    def test_cli_environment_url_normalization_and_fair_counts(self):
        with patch.dict(benchmark.os.environ, {"BONSAI_BASE_URL": "http://localhost:1234/v1", "BONSAI_API_KEY": "secret"}):
            args = benchmark.parse_args(["--modes", "OFF,medium", "--concurrency", "1,3", "--requests", "6"])
        self.assertEqual(args.base_url, "http://localhost:1234")
        self.assertEqual(args.modes, ["off", "medium"])
        self.assertEqual(args.api_key, "secret")
        self.assertEqual(args.requests, 6)
        with patch("sys.stderr", new_callable=io.StringIO):
            for argv in (["--concurrency", "2,4", "--requests", "6"], ["--base-url", "http://secret@localhost"],
                         ["--temperature", "nan"], ["--needle-words", "3"], ["--budget", "-2"]):
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    benchmark.parse_args(argv)

    def test_needle_and_prompt_files_are_reproducible(self):
        case = benchmark.make_needle_case(100, 42)
        self.assertEqual(case, benchmark.make_needle_case(100, 42))
        for value in case["expected"].values():
            self.assertIn(value, case["prompt"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.json"
            path.write_text(json.dumps(["question", {"id": "needle", "prompt": "text", "expected": {"key": "value"}}]), encoding="utf-8")
            cases = benchmark.load_cases(config(prompt_file=str(path)))
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[1]["expected"], {"key": "value"})
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                benchmark.load_cases(config(prompt_file=str(path)))

    def test_complete_cli_report_and_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            for finish, expected_code in (("stop", 0), ("length", 1)):
                with self.subTest(finish=finish), patch.object(benchmark, "metadata", return_value={}), \
                        patch.object(benchmark.urllib.request, "urlopen", side_effect=lambda *a, **k: Response([
                            chunk(content="answer", finish=finish, usage={"completion_tokens": 1}), "[DONE]"])), \
                        patch("sys.stdout", new_callable=io.StringIO), patch.dict(benchmark.os.environ, {"BONSAI_API_KEY": "test-secret"}):
                    exit_code = benchmark.main(["--modes", "off", "--concurrency", "1", "--requests", "1",
                                               "--warmup", "0", "--no-gpu-info", "--output", str(output)])
                self.assertEqual(exit_code, expected_code)
                data = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(len(data["cells"]), 1)
                self.assertEqual(len(data["aggregates"]), 1)
                self.assertEqual(data["cells"][0]["results"][0]["content"], "answer")
                self.assertNotIn("api_key", data["config"])
                self.assertNotIn("test-secret", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
