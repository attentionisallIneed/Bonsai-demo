#!/usr/bin/env python3
"""Validate a real near-limit input and its follow-up against a running server.

Example: python scripts/benchmark_context.py --context-size 65536
    --reserve-tokens 2048 --output performance-results/context-64k.json

Three seeded values sit near 10%, 50%, and 90% of diverse synthetic inventory
records. The entire chat template is tokenized, then rows and short padding are
bisected to approach context-size minus reserve-tokens. Response usage, complete
JSON answers, and follow-up history must pass; merely loading a context is not
success. /apply-template absence is recorded as a fallback estimate, and actual
usage remains mandatory. Cache counters are observations, not guaranteed hits.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid

import benchmark_throughput as bench


class APIError(ValueError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class Client:
    def __init__(self, config):
        self.config = config
        self.calls = []

    def post(self, path, body):
        started = time.perf_counter()
        call = {"path": path, "status": None, "success": False}
        request_headers = bench.headers(self.config.api_key)
        request_headers["Accept"] = "application/json"
        request = urllib.request.Request(self.config.base_url + path,
            data=json.dumps(body).encode("utf-8"), headers=request_headers)
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                call["status"] = response.status
                result = json.load(response)
            call["success"] = True
            return result
        except urllib.error.HTTPError as error:
            call["status"] = error.code
            try:
                detail = error.read().decode("utf-8", errors="replace")
            finally:
                error.close()
            call["error"] = f"HTTP {error.code}: {detail}"
            raise APIError(call["error"], error.code) from error
        except Exception as error:
            call["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            call["seconds"] = time.perf_counter() - started
            self.calls.append(bench.redact(call, self.config.api_key))


class TokenCounter:
    def __init__(self, client):
        self.client = client
        self.method = None
        self.fallback_reason = None
        self.history = []

    def count(self, messages, **details):
        started = time.perf_counter()
        if self.method != "content_plus_estimated_template":
            try:
                rendered = self.client.post("/apply-template", {
                    "messages": messages, "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False}})
                if not isinstance(rendered, dict) or not isinstance(rendered.get("prompt"), str):
                    raise APIError("/apply-template returned no prompt string", 501)
                content = rendered["prompt"]
                self.method = "apply_template_and_tokenize"
                overhead = 0
            except APIError as error:
                if error.status not in (404, 405, 501):
                    raise
                self.method = "content_plus_estimated_template"
                self.fallback_reason = str(error)
        if self.method == "content_plus_estimated_template":
            # A deliberately explicit heuristic, never represented as exact.
            content = "\n".join(message.get("content") or "" for message in messages)
            overhead = 64 + 16 * max(0, len(messages) - 1)
        response = self.client.post("/tokenize", {"content": content,
            "add_special": False, "parse_special": True})
        if not isinstance(response, dict) or not isinstance(response.get("tokens"), list):
            raise ValueError("/tokenize returned no tokens array")
        count = len(response["tokens"]) + overhead
        self.history.append({"tokens": count, "method": self.method,
                             "estimated_template_tokens": overhead,
                             "seconds": time.perf_counter() - started, **details})
        return count


class Inventory:
    def __init__(self, seed, run_id):
        self.seed = seed
        self.run_id = run_id
        self.random = random.Random(seed)
        self.expected = {key: f"{self.random.getrandbits(48):012X}" for key in ("beginning", "middle", "end")}
        self.lines = []

    def ensure_lines(self, count):
        products = ("relay", "sensor", "bearing", "filter", "valve", "cable", "bracket", "adapter")
        depots = ("North", "East", "West", "South", "Central", "Harbor")
        states = ("available", "reserved", "inspection", "returned", "transit")
        while len(self.lines) < count:
            index = len(self.lines)
            self.lines.append(
                f"Record {index:06d}: sku={self.random.choice(products)}-{self.random.randrange(1000, 10000)}; "
                f"depot={self.random.choice(depots)}; stock={self.random.randrange(1, 500)}; "
                f"reserved={self.random.randrange(0, 20)}; state={self.random.choice(states)}; "
                f"batch={self.random.getrandbits(20):05X}.\n")

    def render(self, rows, padding=0):
        self.ensure_lines(rows)
        positions = {}
        for fraction, name in ((0.1, "beginning"), (0.5, "middle"), (0.9, "end")):
            positions.setdefault(int(rows * fraction), []).append(name)
        body = []
        for index in range(rows):
            for name in positions.get(index, []):
                body.append(f"AUDIT-NEEDLE location={name} value={self.expected[name]}.\n")
            body.append(self.lines[index])
        prompt = (f"Context benchmark marker {self.run_id}.\n"
                  "Read the inventory. Three AUDIT-NEEDLE records contain values at locations beginning, middle, end. "
                  "Ignore ordinary inventory batch codes. Remember each exact audit value.\n<inventory>\n"
                  + "".join(body) + "</inventory>\n"
                  + ("padding " * padding)
                  + '\nReturn only a JSON object with exactly the keys "beginning", "middle", "end" '
                    "mapped to their exact audit values. No explanation or markdown.")
        return [{"role": "user", "content": prompt}]


def construct_input(corpus, counter, target, guard=8):
    """Maximize rows then padding under a guarded total rendered-token target."""
    limit = target - guard
    def measure(rows, padding=0):
        messages = corpus.render(rows, padding)
        return counter.count(messages, rows=rows, padding_words=padding)
    low = 3
    low_count = measure(low)
    if low_count > limit:
        raise ValueError(f"input budget {target} is smaller than the minimum test document ({low_count} tokens)")
    high = max(4, target // 48)
    while True:
        high_count = measure(high)
        if high_count > limit:
            break
        low, low_count = high, high_count
        high *= 2
        if high > target * 2:
            raise ValueError("tokenizer did not grow with document length")
    while low + 1 < high:
        middle = (low + high) // 2
        count = measure(middle)
        if count <= limit:
            low, low_count = middle, count
        else:
            high = middle
    pad_low = 0
    pad_high = max(16, (limit - low_count) * 2 + 16)
    while measure(low, pad_high) <= limit:
        pad_high *= 2
        if pad_high > target * 2:
            raise ValueError("tokenizer did not grow with padding length")
    while pad_low + 1 < pad_high:
        middle = (pad_low + pad_high) // 2
        count = measure(low, middle)
        if count <= limit:
            pad_low, low_count = middle, count
        else:
            pad_high = middle
    messages = corpus.render(low, pad_low)
    final_count = counter.count(messages, rows=low, padding_words=pad_low, final=True)
    return messages, {"rows": low, "padding_words": pad_low, "counted_prompt_tokens": final_count,
                      "target_prompt_tokens": target, "guard_tokens": guard,
                      "count_method": counter.method, "fallback_reason": counter.fallback_reason}


def parse_answer(content, expected):
    try:
        answer = json.loads(content.strip())
    except (ValueError, AttributeError) as error:
        return {"passed": False, "answer": None, "error": f"answer is not exact JSON: {error}"}
    return {"passed": answer == expected, "answer": answer, "expected": expected,
            "error": None if answer == expected else "JSON map does not exactly match all three expected values"}


def cache_observation(result):
    usage, timings = result.get("usage", {}), result.get("timings", {})
    details = usage.get("prompt_tokens_details") or {}
    cache_n = timings.get("cache_n")
    usage_cached = details.get("cached_tokens")
    counts = [value for value in (cache_n, usage_cached) if isinstance(value, (int, float))]
    state = "reported_reuse" if any(value > 0 for value in counts) else ("reported_zero" if counts else "unavailable")
    return {"observation": state, "cache_n": cache_n, "prompt_n": timings.get("prompt_n"),
            "usage_prompt_tokens": usage.get("prompt_tokens"), "usage_cached_tokens": usage_cached,
            "prompt_ms": timings.get("prompt_ms"), "prompt_per_second": timings.get("prompt_per_second")}


def validate_result(result, expected, config, counted_tokens, fallback=False, first=False):
    answer = parse_answer(result.get("content", ""), expected)
    usage = result.get("usage", {})
    prompt_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    counts_known = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                       for value in (prompt_tokens, output_tokens))
    target = config.context_size - config.reserve_tokens
    tolerance = max(config.tolerance, 128) if fallback else config.tolerance
    token_checks = {"usage_available": counts_known,
                    "inside_context": counts_known and prompt_tokens + output_tokens <= config.context_size,
                    "input_within_reserved_budget": counts_known and (prompt_tokens <= target if first else prompt_tokens + config.max_tokens <= config.context_size),
                    "near_requested_input": counts_known and (target - tolerance <= prompt_tokens <= target if first else True),
                    "matches_tokenizer": counts_known and abs(prompt_tokens - counted_tokens) <= (128 if fallback else 8)}
    passed = bool(result.get("complete") and not result.get("truncated") and answer["passed"] and all(token_checks.values()))
    return {"passed": passed, "answer_check": answer, "token_checks": token_checks,
            "actual_prompt_tokens": prompt_tokens, "actual_output_tokens": output_tokens,
            "counted_prompt_tokens": counted_tokens,
            "usage_minus_counted": prompt_tokens - counted_tokens if counts_known else None,
            "effective_closeness_tolerance": tolerance, "cache": cache_observation(result)}


def run(config, client=None):
    started = time.perf_counter()
    run_id = uuid.uuid4().hex
    client = client or Client(config)
    counter = TokenCounter(client)
    corpus = Inventory(config.seed, run_id)
    report = {"schema_version": 1, "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
              "config": {key: value for key, value in vars(config).items() if key != "api_key"},
              "expected": corpus.expected, "passed": False, "turns": [],
              "notes": ["Synthetic three-position retrieval and history validation, not general long-context reasoning quality.",
                        "Actual usage is mandatory; loading the server alone cannot pass.",
                        "Cache counters report observed reuse only. An absent counter is not evidence of a cache hit."]}
    report["metadata_before"] = bench.metadata(config)
    try:
        messages, construction = construct_input(corpus, counter, config.context_size - config.reserve_tokens)
        report["construction"] = construction
        report["prompt"] = messages[0]["content"]
        report["prompt_sha256"] = hashlib.sha256(report["prompt"].encode()).hexdigest()
        fallback = counter.method == "content_plus_estimated_template"
        print(f"Input prepared: {construction['counted_prompt_tokens']} tokens ({counter.method}), "
              f"target={construction['target_prompt_tokens']}, rows={construction['rows']}", flush=True)
        case = {"id": "near-limit", "prompt": report["prompt"], "messages": messages}
        first = bench.send_chat_request(0, case, "off", config, config.seed)
        validation = validate_result(first, corpus.expected, config, construction["counted_prompt_tokens"], fallback, first=True)
        report["turns"].append({"name": "initial", "response": first, "validation": validation})
        bench.write_report(config.output, report, config.api_key)
        if not config.skip_followup and validation["passed"]:
            history = messages + [{"role": "assistant", "content": first["content"]},
                {"role": "user", "content": "Return the same exact beginning, middle, end audit-value JSON map again. Only JSON."}]
            follow_count = counter.count(history, followup=True)
            if follow_count + config.max_tokens > config.context_size:
                raise ValueError("follow-up history plus its output allowance exceeds context; increase reserve-tokens")
            follow = bench.send_chat_request(1, {"id": "follow-up", "prompt": report["prompt"], "messages": history}, "off", config, config.seed)
            follow_validation = validate_result(follow, corpus.expected, config, follow_count, fallback)
            report["turns"].append({"name": "followup", "response": follow, "validation": follow_validation})
        elif not config.skip_followup:
            report["followup_skipped_reason"] = "initial input validation failed"
        required_turns = 1 if config.skip_followup else 2
        report["passed"] = len(report["turns"]) == required_turns and all(turn["validation"]["passed"] for turn in report["turns"])
        report["cache_reuse_observed"] = any(turn["validation"]["cache"]["observation"] == "reported_reuse"
                                              for turn in report["turns"] if turn["name"] == "followup")
    except KeyboardInterrupt:
        report["interrupted"] = True
        report["error"] = "interrupted"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["counting_history"] = counter.history
        report["preparation_requests"] = client.calls
        report["metadata_after"] = bench.metadata(config)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["wall_sec"] = time.perf_counter() - started
        bench.write_report(config.output, report, config.api_key)
    return bench.redact(report, config.api_key)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context-size", type=int, required=True)
    parser.add_argument("--reserve-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=512, help="output allowance for each turn")
    parser.add_argument("--base-url", default=os.getenv("BONSAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080")))
    parser.add_argument("--model", default=os.getenv("BONSAI_API_MODEL", "bonsai"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--tolerance", type=int, default=64, help="maximum shortfall from requested first-input size")
    parser.add_argument("--skip-followup", action="store_true")
    parser.add_argument("--server-binary", help="optional executable to query with --version")
    parser.add_argument("--no-gpu-info", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        args.base_url = bench.normalize_base_url(args.base_url)
        if not 1024 <= args.context_size <= 262144:
            raise ValueError("context-size must be between 1024 and 262144")
        if args.max_tokens < 16 or args.reserve_tokens < args.max_tokens or args.context_size - args.reserve_tokens < 512:
            raise ValueError("max-tokens must be >=16; reserve must cover output and leave >=512 input tokens")
        if not args.skip_followup and args.reserve_tokens < 2 * args.max_tokens + 128:
            raise ValueError("follow-up needs reserve-tokens >= 2 * max-tokens + 128, or use --skip-followup")
        if args.timeout <= 0 or not math.isfinite(args.timeout) or args.tolerance < 8:
            raise ValueError("timeout must be positive and finite; tolerance must be >=8")
    except ValueError as error:
        parser.error(str(error))
    args.api_key = os.getenv("BONSAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    args.cache, args.budget, args.temperature = "warm", 0, 0.0
    return args


def main(argv=None):
    args = parse_args(argv)
    report = run(args)
    print(json.dumps({"passed": report["passed"], "output": args.output,
                      "cache_reuse_observed": report.get("cache_reuse_observed"),
                      "wall_sec": report["wall_sec"], "error": report.get("error")}), flush=True)
    return 130 if report.get("interrupted") else (0 if report["passed"] else 1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
