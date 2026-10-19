"""Offline behavioral checks for near-limit construction and real-answer gates."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("benchmark_context", SCRIPTS / "benchmark_context.py")
context = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    spec.loader.exec_module(context)


def config(output="unused.json", **overrides):
    values = dict(context_size=4096, reserve_tokens=1024, max_tokens=128,
                  base_url="http://localhost:8080", model="test", seed=42,
                  timeout=5, tolerance=64, skip_followup=False, api_key="test-secret",
                  server_binary=None, no_gpu_info=True, cache="warm", budget=0,
                  temperature=0.0, output=output)
    values.update(overrides)
    return argparse.Namespace(**values)


def render(messages):
    return "\n".join(message["role"] + "\n" + message["content"] for message in messages) + "\nassistant\n"


class FakeClient:
    def __init__(self, fallback=False, template_status=None):
        self.calls = []
        self.fallback = fallback
        self.template_status = template_status
        self.bodies = []

    def post(self, path, body):
        self.calls.append({"path": path, "success": True})
        self.bodies.append((path, body))
        if path == "/apply-template":
            if self.fallback or self.template_status:
                raise context.APIError("endpoint unavailable", self.template_status or 404)
            return {"prompt": render(body["messages"])}
        if path == "/tokenize":
            return {"tokens": [1] * len(body["content"].split())}
        raise AssertionError(path)


class ConstructionChecks(unittest.TestCase):
    def test_binary_construction_matches_entire_template_and_preserves_three_positions(self):
        client = FakeClient()
        counter = context.TokenCounter(client)
        corpus = context.Inventory(42, "stable-id")
        messages, details = context.construct_input(corpus, counter, 3072)
        self.assertLessEqual(details["counted_prompt_tokens"], 3072)
        self.assertGreaterEqual(details["counted_prompt_tokens"], 3072 - 10)
        self.assertEqual(details["counted_prompt_tokens"], len(render(messages).split()))
        self.assertEqual(details["count_method"], "apply_template_and_tokenize")
        prompt = messages[0]["content"]
        fractions = [prompt.index(value) / len(prompt) for value in corpus.expected.values()]
        self.assertTrue(0.05 < fractions[0] < 0.2)
        self.assertTrue(0.4 < fractions[1] < 0.6)
        self.assertTrue(0.8 < fractions[2] < 0.95)
        self.assertIn("depot=", prompt)
        self.assertIn("stock=", prompt)
        self.assertLess(len(counter.history), 40)
        template_requests = [body for path, body in client.bodies if path == "/apply-template"]
        self.assertTrue(all(body["chat_template_kwargs"]["enable_thinking"] is False for body in template_requests))
        token_requests = [body for path, body in client.bodies if path == "/tokenize"]
        self.assertTrue(all(body["parse_special"] and not body["add_special"] for body in token_requests))

    def test_seeded_inventory_is_stable_despite_different_generation_orders(self):
        first = context.Inventory(1, "id")
        first.render(900)
        second = context.Inventory(1, "id")
        self.assertEqual(first.render(100, 3), second.render(100, 3))
        self.assertNotEqual(first.expected, context.Inventory(2, "id").expected)

    def test_missing_template_endpoint_has_explicit_estimate_fallback(self):
        client = FakeClient(fallback=True)
        counter = context.TokenCounter(client)
        messages, details = context.construct_input(context.Inventory(42, "id"), counter, 3072)
        self.assertEqual(details["count_method"], "content_plus_estimated_template")
        self.assertIn("endpoint unavailable", details["fallback_reason"])
        self.assertEqual(sum(path == "/apply-template" for path, _ in client.bodies), 1)
        self.assertEqual(details["counted_prompt_tokens"], len(messages[0]["content"].split()) + 64)

    def test_authentication_or_server_failure_does_not_silently_fallback(self):
        for status in (400, 401, 403, 500, 503):
            with self.subTest(status=status), self.assertRaises(context.APIError):
                context.TokenCounter(FakeClient(template_status=status)).count([{"role": "user", "content": "text"}])

    def test_too_small_input_never_runs_a_fake_success(self):
        with self.assertRaisesRegex(ValueError, "minimum test document"):
            context.construct_input(context.Inventory(42, "id"), context.TokenCounter(FakeClient()), 30)


class ResultChecks(unittest.TestCase):
    def result(self, content=None, **overrides):
        expected = context.Inventory(42, "id").expected
        value = {"complete": True, "truncated": False, "content": content or json.dumps(expected),
                 "usage": {"prompt_tokens": 3064, "completion_tokens": 30},
                 "timings": {"prompt_n": 3064, "cache_n": 0, "prompt_ms": 10}}
        value.update(overrides)
        return value

    def validate(self, result, **overrides):
        return context.validate_result(result, context.Inventory(42, "id").expected, config(), 3064, first=True, **overrides)

    def test_exact_map_requires_json_and_every_value_without_extra_keys(self):
        expected = context.Inventory(42, "id").expected
        self.assertTrue(context.parse_answer(json.dumps(expected), expected)["passed"])
        for content in ("```json\n" + json.dumps(expected) + "\n```", json.dumps({**expected, "extra": "x"}),
                        json.dumps({**expected, "middle": "wrong"}), "prefix " + json.dumps(expected)):
            with self.subTest(content=content):
                self.assertFalse(context.parse_answer(content, expected)["passed"])

    def test_duplicate_json_keys_cannot_hide_a_contradictory_value(self):
        expected = context.Inventory(42, "id").expected
        content = '{"beginning":"wrong",' + json.dumps(expected)[1:]
        self.assertEqual(json.loads(content), expected)  # Standard parsing would conceal the error.
        result = context.parse_answer(content, expected)
        self.assertFalse(result["passed"])
        self.assertIn("duplicate JSON key", result["error"])

    def test_actual_usage_not_configuration_or_correct_answer_is_required(self):
        self.assertTrue(self.validate(self.result())["passed"])
        for result in (self.result(usage={}),
                       self.result(usage={"prompt_tokens": 2000, "completion_tokens": 30}),
                       self.result(usage={"prompt_tokens": 3073, "completion_tokens": 30}),
                       self.result(usage={"prompt_tokens": 3064, "completion_tokens": 2000}),
                       self.result(truncated=True), self.result(complete=False)):
            with self.subTest(result=result):
                self.assertFalse(self.validate(result)["passed"])

    def test_counter_mismatch_detects_possible_truncation_even_near_target(self):
        value = self.result(usage={"prompt_tokens": 3052, "completion_tokens": 30})
        validation = self.validate(value)
        self.assertTrue(validation["token_checks"]["near_requested_input"])
        self.assertFalse(validation["token_checks"]["matches_tokenizer"])
        self.assertFalse(validation["passed"])

    def test_cache_observations_do_not_invent_missing_reuse(self):
        self.assertEqual(context.cache_observation({})["observation"], "unavailable")
        self.assertEqual(context.cache_observation(self.result())["observation"], "reported_zero")
        self.assertEqual(context.cache_observation(self.result(timings={"cache_n": 3000, "prompt_n": 64}))["observation"], "reported_reuse")


class WorkflowChecks(unittest.TestCase):
    def sender(self, req_id, case, mode, settings, seed):
        self.sent.append(case)
        expected = context.Inventory(seed, "ignored").expected
        actual_count = len(render(case["messages"]).split())
        return {"success": True, "complete": True, "truncated": False, "finish_reason": "stop",
                "content": json.dumps(expected), "ttft_sec": 0.2, "latency_sec": 0.4,
                "usage": {"prompt_tokens": actual_count, "completion_tokens": 30},
                "timings": {"cache_n": 0 if req_id == 0 else actual_count - 20, "prompt_n": actual_count if req_id == 0 else 20}}

    def execute(self, settings, client=None, sender=None):
        self.sent = []
        with patch.object(context.bench, "metadata", return_value={"safe": True}), \
                patch.object(context.bench, "send_chat_request", side_effect=sender or self.sender), \
                patch("sys.stdout", new_callable=io.StringIO):
            return context.run(settings, client or FakeClient())

    def test_two_actual_turns_replay_history_and_save_measured_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            report = self.execute(config(str(path)))
            self.assertTrue(report["passed"])
            self.assertTrue(report["cache_reuse_observed"])
            self.assertEqual(len(self.sent), 2)
            history = self.sent[1]["messages"]
            self.assertEqual([message["role"] for message in history], ["user", "assistant", "user"])
            self.assertEqual(history[0], self.sent[0]["messages"][0])
            self.assertEqual(json.loads(history[1]["content"]), report["expected"])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(saved["passed"])
            self.assertIn("counting_history", saved)
            self.assertIn("wall_sec", saved)
            self.assertEqual(saved["turns"][0]["response"]["ttft_sec"], 0.2)
            self.assertNotIn("api_key", saved["config"])

    def test_initial_failure_never_passes_or_attempts_followup(self):
        def failed(*args, **kwargs):
            result = self.sender(*args, **kwargs)
            result["content"] = "{}"
            return result
        with tempfile.TemporaryDirectory() as temporary:
            report = self.execute(config(str(Path(temporary) / "report.json")), sender=failed)
        self.assertFalse(report["passed"])
        self.assertEqual(len(self.sent), 1)
        self.assertIn("initial input validation failed", report["followup_skipped_reason"])

    def test_skip_followup_and_template_fallback_still_require_actual_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self.execute(config(str(Path(temporary) / "report.json"), skip_followup=True), FakeClient(fallback=True))
        self.assertTrue(report["passed"])
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(report["construction"]["count_method"], "content_plus_estimated_template")
        self.assertTrue(report["turns"][0]["validation"]["token_checks"]["usage_available"])
        self.assertFalse(report["cache_reuse_observed"])

    def test_preparation_error_saves_failed_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            report = self.execute(config(str(path)), FakeClient(template_status=401))
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(report["passed"])
        self.assertFalse(self.sent)
        self.assertIn("endpoint unavailable", saved["error"])

    def test_client_http_error_and_call_log_redact_secret(self):
        client = context.Client(config())
        error = HTTPError("http://localhost", 401, "Unauthorized", {}, io.BytesIO(b"bad test-secret"))
        with patch.object(context.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(context.APIError):
                client.post("/tokenize", {"content": "text"})
        self.assertNotIn("test-secret", json.dumps(client.calls))
        self.assertEqual(client.calls[0]["status"], 401)

    def test_cli_guards_output_reserve_environment_and_exit_status(self):
        with patch.dict(context.os.environ, {"BONSAI_BASE_URL": "http://localhost/v1", "BONSAI_API_KEY": "secret"}):
            args = context.parse_args(["--context-size", "65536", "--output", "x.json"])
        self.assertEqual(args.base_url, "http://localhost")
        self.assertEqual(args.api_key, "secret")
        self.assertEqual(args.cache, "warm")
        with patch("sys.stderr", new_callable=io.StringIO):
            for extra in (["--reserve-tokens", "512"], ["--timeout", "nan"], ["--context-size", "512"]):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    context.parse_args(["--context-size", "65536", "--output", "x.json", *extra])
        for passed, exit_code in ((True, 0), (False, 1)):
            with patch.object(context, "run", return_value={"passed": passed, "wall_sec": 1}), \
                    patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(context.main(["--context-size", "65536", "--output", "x.json"]), exit_code)


if __name__ == "__main__":
    unittest.main()
