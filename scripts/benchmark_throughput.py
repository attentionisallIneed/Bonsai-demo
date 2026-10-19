#!/usr/bin/env python3
"""Reproducible stdlib-only SSE benchmark for a local compatible chat server.

Quick cell: python scripts/benchmark_throughput.py --modes off --concurrency 1
    --requests 1 --max-tokens 256 --cache cold --output benchmark_results.json

Keys come from BONSAI_API_KEY or OPENAI_API_KEY, never reports. Latency includes
the entire body. TTFT includes reasoning; first_content_sec measures the answer.
Length stops are truncated, not completed answers. Cold requests use a unique
prefix and cache_prompt=false; warm requests prime the same prompt. Inspect
server timings for actual cache reuse. No third-party dependencies are required.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

PROMPTS = [
    "三个箱子贴着苹果、橘子、苹果和橘子的标签，所有标签都贴错了。只能从一个箱子取一个水果查看，应该选哪个箱子，如何确定全部内容？请解释。",
    "高铁以300km/h从北京前往上海（相距1318km），动车以200km/h同时从上海前往北京。1.5小时后动车停靠20分钟。计算相遇时间及各自行驶距离。",
    "大模型在手机和边缘计算盒子落地有哪些技术瓶颈？从硬件、量化算法、内存带宽三个维度说明优化路径。",
    "100枚外观相同的金币中只有1枚较轻的假币。用没有砝码的天平，最少称量几次才能保证找出假币？给出分组策略和证明。",
]
MODES = ("off", "low", "medium", "xhigh")
PROPS_FIELDS = ("n_ctx", "n_ctx_train", "n_slots", "total_slots", "model_path",
                "model_alias", "model", "build_info", "version", "chat_format",
                "modalities", "kv_unified", "context_shift")
GENERATION_FIELDS = ("n_ctx", "n_predict", "n_batch", "n_ubatch", "seed",
                     "temperature", "top_k", "top_p", "min_p")


def normalize_base_url(value):
    parsed = urllib.parse.urlsplit(value.rstrip("/"))
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("base URL must use http:// or https://")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain credentials, a query, or a fragment")
    path = parsed.path.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def redact(value, key=""):
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, dict):
        return {k: redact(v, key) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact(v, key) for v in value]
    return value


def headers(api_key):
    result = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        result["Authorization"] = "Bearer " + api_key
    return result


def get_json(base_url, path, api_key, timeout):
    req = urllib.request.Request(base_url + path, headers=headers(api_key))
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def iter_sse(response):
    """Support SSE comments, CRLF, multiline data and a final unseparated frame."""
    data = []
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)


def make_payload(prompt, mode, config, seed, cache_marker=None, max_tokens=None):
    kwargs = {"enable_thinking": mode != "off"}
    if mode != "off":
        kwargs["reasoning_effort"] = mode
    if cache_marker:
        prompt = f"Benchmark marker {cache_marker}; ignore this marker.\n\n{prompt}"
    payload = {"model": config.model, "messages": [{"role": "user", "content": prompt}],
               "stream": True, "stream_options": {"include_usage": True},
               "max_tokens": config.max_tokens if max_tokens is None else max_tokens,
               "temperature": config.temperature, "seed": seed,
               "cache_prompt": config.cache == "warm", "chat_template_kwargs": kwargs}
    if config.budget is not None:
        payload["thinking_budget_tokens"] = 0 if mode == "off" else config.budget
    return payload


def merge_tool_calls(existing, chunks):
    for chunk in chunks:
        target = existing.setdefault(chunk.get("index", 0), {"function": {"name": "", "arguments": ""}})
        for field in ("id", "type"):
            if field in chunk:
                target[field] = chunk[field]
        for field in ("name", "arguments"):
            target["function"][field] += chunk.get("function", {}).get(field, "") or ""


def send_chat_request(req_id, case, mode, config, seed, cache_marker=None, max_tokens=None):
    payload = make_payload(case["prompt"], mode, config, seed, cache_marker, max_tokens)
    started = time.perf_counter()
    result = {"req_id": req_id, "prompt_id": case["id"], "seed": seed,
              "prompt_sha256": hashlib.sha256(payload["messages"][0]["content"].encode()).hexdigest(),
              "cache_marker": cache_marker, "success": False, "complete": False,
              "truncated": False, "stream_done": False, "body_complete": False,
              "finish_reason": None, "ttft_sec": None, "first_content_sec": None,
              "latency_sec": None, "http_status": None, "event_count": 0,
              "content": "", "reasoning_content": "", "tool_calls": [],
              "usage": {}, "timings": {}, "error": None}
    content, reasoning, tools = [], [], {}
    try:
        req = urllib.request.Request(config.base_url + "/v1/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"), headers=headers(config.api_key))
        with urllib.request.urlopen(req, timeout=config.timeout) as response:
            result["http_status"] = response.status
            for data in iter_sse(response):
                if data == "[DONE]":
                    result["stream_done"] = True
                    continue  # Keep reading through EOF before stopping the timer.
                event = json.loads(data)
                result["event_count"] += 1
                if event.get("error"):
                    raise ValueError("server stream error: " + json.dumps(event["error"]))
                for field in ("id", "model", "system_fingerprint"):
                    if field in event:
                        result[field] = event[field]
                for field in ("usage", "timings"):
                    if event.get(field):
                        result[field].update(event[field])
                choices = event.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                thought, text = delta.get("reasoning_content") or "", delta.get("content") or ""
                tool_chunks = delta.get("tool_calls") or []
                now = time.perf_counter() - started
                if (thought or text or tool_chunks) and result["ttft_sec"] is None:
                    result["ttft_sec"] = now
                if text and result["first_content_sec"] is None:
                    result["first_content_sec"] = now
                reasoning.append(thought)
                content.append(text)
                merge_tool_calls(tools, tool_chunks)
                if choice.get("finish_reason") is not None:
                    result["finish_reason"] = choice["finish_reason"]
            result["body_complete"] = True
        if not result["stream_done"] or result["finish_reason"] is None:
            raise ValueError("incomplete SSE response: missing [DONE] or finish_reason")
        result["success"] = True
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        try:
            result["error"] = f"HTTP {exc.code}: " + exc.read().decode("utf-8", errors="replace")
        except Exception:
            result["error"] = f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["latency_sec"] = time.perf_counter() - started
    result["content"], result["reasoning_content"] = "".join(content), "".join(reasoning)
    result["tool_calls"] = [tools[index] for index in sorted(tools)]
    result["content_chars"], result["reasoning_chars"] = len(result["content"]), len(result["reasoning_content"])
    result["truncated"] = result["finish_reason"] == "length"
    result["complete"] = (result["success"] and result["finish_reason"] in ("stop", "tool_calls")
                          and bool(result["content"] or result["tool_calls"]))
    if case.get("expected"):
        result["retrieval_checks"] = {name: value in result["content"] for name, value in case["expected"].items()}
        result["retrieval_passed"] = result["complete"] and all(result["retrieval_checks"].values())
    result["prompt_tokens"] = result["usage"].get("prompt_tokens", result["timings"].get("prompt_n"))
    result["predicted_tokens"] = result["usage"].get("completion_tokens", result["timings"].get("predicted_n"))
    return redact(result, config.api_key)


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def distribution(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "p50": percentile(values, 0.5), "p95": percentile(values, 0.95)}


def summarize(results, wall_sec):
    successful = [r for r in results if r["success"]]
    counts = [r["predicted_tokens"] for r in successful]
    total_tokens = sum(counts) if all(isinstance(v, (int, float)) for v in counts) else None
    reasons = {}
    for result in results:
        reason = result["finish_reason"] or "missing"
        reasons[reason] = reasons.get(reason, 0) + 1
    summary = {"total_requests": len(results), "success_count": len(successful),
               "complete_count": sum(r["complete"] for r in results),
               "truncated_count": sum(r["truncated"] for r in results),
               "error_count": len(results) - len(successful), "finish_reasons": reasons,
               "batch_wall_sec": wall_sec, "total_predicted_tokens": total_tokens,
               "system_throughput_tps": total_tokens / wall_sec if total_tokens is not None and wall_sec > 0 else None}
    for field in ("latency_sec", "ttft_sec", "first_content_sec"):
        summary[field] = distribution([r[field] for r in successful])
    summary["complete_latency_sec"] = distribution([r["latency_sec"] for r in successful if r["complete"]])
    for name, field in (("server_decode_tps", "predicted_per_second"), ("server_prompt_tps", "prompt_per_second")):
        summary[name] = distribution([r["timings"].get(field) for r in successful])
    retrieval = [r for r in results if "retrieval_passed" in r]
    if retrieval:
        summary.update(retrieval_passed=sum(r["retrieval_passed"] for r in retrieval), retrieval_total=len(retrieval))
    return summary


def prompt_schedule(prompt_count, count, seed):
    order = list(range(prompt_count))
    random.Random(seed).shuffle(order)
    return [order[index % prompt_count] for index in range(count)]


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace",
                                timeout=10, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def gpu_snapshot(enabled=True):
    binary = shutil.which("nvidia-smi") if enabled else None
    if not binary:
        return {"available": False}
    return command_output([binary, "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
                           "--format=csv,nounits"])


def metadata(config):
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "gpu": gpu_snapshot(not config.no_gpu_info)}
    try:
        props = get_json(config.base_url, "/props", config.api_key, min(config.timeout, 10))
        result["props"] = {name: props[name] for name in PROPS_FIELDS if name in props}
        defaults = props.get("default_generation_settings", {})
        result["props"]["default_generation_settings"] = {name: defaults[name] for name in GENERATION_FIELDS if name in defaults}
        params = defaults.get("params", {})
        if isinstance(params, dict):
            result["props"]["default_generation_settings"]["params"] = {name: params[name] for name in GENERATION_FIELDS if name in params}
    except Exception as exc:
        result["props_error"] = str(exc)
    if config.server_binary:
        result["server_version"] = command_output([str(Path(config.server_binary).resolve()), "--version"])
    return redact(result, config.api_key)


def run_benchmark_cell(config, cases, mode, concurrency, repeat_index, run_id):
    schedule = prompt_schedule(len(cases), config.requests, config.seed + repeat_index)
    warmups = []
    for index in range(config.warmup):
        warmups.append(send_chat_request(f"warmup-{index}", {"id": "warmup", "prompt": "Reply with OK."},
                                         "off", config, config.seed, uuid.uuid4().hex, max_tokens=8))
    if config.cache == "warm":
        for case_index in dict.fromkeys(schedule):
            warmups.append(send_chat_request(f"prime-{case_index}", cases[case_index], mode, config, config.seed, max_tokens=1))
    results = []
    gpu_before = gpu_snapshot(not config.no_gpu_info)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for index, case_index in enumerate(schedule):
            marker = None
            if config.cache == "cold":
                marker = hashlib.sha256(f"{run_id}:{mode}:{concurrency}:{repeat_index}:{index}".encode()).hexdigest()
            futures.append(executor.submit(send_chat_request, index, cases[case_index], mode,
                                           config, config.seed + index, marker))
        for future in as_completed(futures):
            results.append(future.result())
    wall_sec = time.perf_counter() - started
    results.sort(key=lambda result: result["req_id"])
    return {"mode": mode, "concurrency": concurrency, "repeat": repeat_index + 1,
            "cache": config.cache, "summary": summarize(results, wall_sec), "warmup_results": warmups,
            "results": results, "gpu_before": gpu_before, "gpu_after": gpu_snapshot(not config.no_gpu_info)}


def make_needle_case(word_count, seed):
    """Retrieval smoke test: filler word count is deliberately not a token estimate."""
    generator = random.Random(seed)
    expected = {name: f"{generator.getrandbits(48):012x}" for name in ("beginning", "middle", "end")}
    filler = ["archive"] * word_count
    for fraction, name in ((0.1, "beginning"), (0.5, "middle"), (0.9, "end")):
        filler[min(word_count - 1, int(word_count * fraction))] = f"\nSecret {name}: {expected[name]}.\n"
    prompt = ("Read the archive and report the exact three secret values in it.\n\n" + " ".join(filler)
              + "\n\nReturn the beginning, middle and end secret values, preserving every character.")
    return {"id": f"needle-{word_count}-words", "prompt": prompt, "expected": expected}


def load_cases(config):
    if config.needle_words:
        return [make_needle_case(config.needle_words, config.seed)]
    if not config.prompt_file:
        return [{"id": f"builtin-{index + 1}", "prompt": text} for index, text in enumerate(PROMPTS)]
    path = Path(config.prompt_file)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() != ".json":
        if not text.strip():
            raise ValueError("prompt file is empty")
        return [{"id": path.stem, "prompt": text}]
    cases = json.loads(text)
    if not isinstance(cases, list) or not cases:
        raise ValueError("JSON prompts must be a non-empty list of strings or {id, prompt, expected?} objects")
    result = []
    for index, case in enumerate(cases):
        case = {"prompt": case} if isinstance(case, str) else dict(case)
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            raise ValueError(f"prompt {index + 1} must contain non-empty text")
        case.setdefault("id", f"file-{index + 1}")
        if "expected" in case and (not isinstance(case["expected"], dict) or not all(isinstance(v, str) and v for v in case["expected"].values())):
            raise ValueError("expected must map labels to non-empty strings")
        result.append(case)
    if len({case["id"] for case in result}) != len(result):
        raise ValueError("prompt IDs must be unique")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.getenv("BONSAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080")))
    parser.add_argument("--model", default=os.getenv("BONSAI_API_MODEL", "bonsai"))
    parser.add_argument("--modes", default="off,low,medium,xhigh", help="comma-separated: off,low,medium,xhigh")
    parser.add_argument("--concurrency", default="1,2,4,8", help="comma-separated concurrent request counts")
    parser.add_argument("--requests", type=int, help="same count/cell; multiple of every concurrency (default >=8)")
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured short requests before each cell")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", choices=("cold", "warm"), default="cold")
    parser.add_argument("--max-tokens", type=int, default=2048, help="total reasoning + answer output allowance")
    parser.add_argument("--budget", type=int, help="thinking_budget_tokens: -1 unlimited, 0 off, N cap")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600, help="socket timeout in seconds")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt-file", help="UTF-8 text (one prompt) or JSON prompt list")
    group.add_argument("--needle-words", type=int, help="retrieval filler words; service reports actual tokens")
    parser.add_argument("--server-binary", help="optional executable to query with --version")
    parser.add_argument("--no-gpu-info", action="store_true", help="skip optional nvidia-smi snapshots")
    parser.add_argument("--label", default="", help="configuration label, for example 64k-f16-ub512")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args(argv)
    try:
        args.base_url = normalize_base_url(args.base_url)
        args.modes = list(dict.fromkeys(value.strip().lower() for value in args.modes.split(",")))
        args.concurrency = list(dict.fromkeys(int(value) for value in args.concurrency.split(",")))
        if not args.modes or any(mode not in MODES for mode in args.modes):
            raise ValueError("modes must contain only off, low, medium, xhigh")
        if not args.concurrency or any(value < 1 for value in args.concurrency):
            raise ValueError("concurrency must be positive")
        multiple = math.lcm(*args.concurrency)
        if args.requests is None:
            args.requests = multiple * math.ceil(8 / multiple)
        if args.requests < multiple or args.requests % multiple:
            raise ValueError(f"requests must be a positive multiple of {multiple}")
        if args.repeat < 1 or args.warmup < 0 or args.max_tokens < 1 or args.timeout <= 0:
            raise ValueError("repeat, max-tokens, timeout must be positive; warmup must be >=0")
        if args.budget is not None and args.budget < -1:
            raise ValueError("budget must be >= -1")
        if args.needle_words is not None and args.needle_words < 10:
            raise ValueError("needle-words must be >=10")
        if not math.isfinite(args.temperature) or args.temperature < 0 or not math.isfinite(args.timeout):
            raise ValueError("temperature and timeout must be finite, temperature >=0")
    except ValueError as exc:
        parser.error(str(exc))
    args.api_key = os.getenv("BONSAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    return args


def write_report(path, report, api_key):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(redact(report, api_key), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    try:
        cases = load_cases(args)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"Invalid prompt file: {exc}") from exc
    run_id = uuid.uuid4().hex
    report = {"schema_version": 1, "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
              "config": {key: value for key, value in vars(args).items() if key != "api_key"},
              "notes": ["latency and throughput include reading the entire response body",
                        "length stops are truncated and excluded from completed-answer latency",
                        "cold = unique prefix + cache_prompt=false; warm = exact-prompt priming + cache_prompt=true",
                        "cache mode is requested behavior; inspect server timings for actual reuse",
                        "needle retrieval is a smoke test, not a general long-context quality benchmark"],
              "metadata_before": metadata(args), "prompts": cases, "cells": []}
    print(f"Benchmark: {len(args.modes)} modes x {len(args.concurrency)} concurrency levels x {args.repeat} repeats; "
          f"{args.requests} requests/cell, cache={args.cache}, max_tokens={args.max_tokens}", flush=True)
    print("Output allowance includes reasoning. Check complete_count and truncated_count before comparing answer latency.", flush=True)
    interrupted = False
    try:
        for repeat_index in range(args.repeat):
            for mode in args.modes:
                for concurrency in args.concurrency:
                    print(f"Running mode={mode} concurrency={concurrency} repeat={repeat_index + 1} ...", flush=True)
                    cell = run_benchmark_cell(args, cases, mode, concurrency, repeat_index, run_id)
                    report["cells"].append(cell)
                    write_report(args.output, report, args.api_key)
                    summary = cell["summary"]
                    rate = summary["system_throughput_tps"]
                    rate_text = f"{rate:.2f}" if rate is not None else "unknown"
                    print(f"  complete={summary['complete_count']}/{args.requests}, truncated={summary['truncated_count']}, "
                          f"errors={summary['error_count']}, throughput={rate_text} tok/s, latency_p50={summary['latency_sec']['p50']}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; saving completed cells.", file=sys.stderr)
    report.update(metadata_after=metadata(args), finished_at=datetime.now(timezone.utc).isoformat(), interrupted=interrupted)
    report["aggregates"] = []
    for mode in args.modes:
        for concurrency in args.concurrency:
            cells = [cell for cell in report["cells"] if cell["mode"] == mode and cell["concurrency"] == concurrency]
            if cells:
                report["aggregates"].append({"mode": mode, "concurrency": concurrency,
                    "summary": summarize([r for cell in cells for r in cell["results"]], sum(cell["summary"]["batch_wall_sec"] for cell in cells))})
    write_report(args.output, report, args.api_key)
    print(f"Saved full request outputs and metadata to {Path(args.output).resolve()}", flush=True)
    if interrupted:
        return 130
    failed = any(not result["complete"] or result.get("retrieval_passed") is False for cell in report["cells"] for result in cell["results"])
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
