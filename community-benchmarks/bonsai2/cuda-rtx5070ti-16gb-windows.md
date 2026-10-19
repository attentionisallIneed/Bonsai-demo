# NVIDIA RTX 5070 Ti 16GB - CUDA 13.3 on Windows

Collected 2026-10-19 (Asia/Shanghai). Bonsai 2 27B PQ2_0, Prism
`prism-b10777-e3de715` / `e3de71522`, NVIDIA driver 617.14,
Intel i5-13600KF (14 cores / 20 logical processors), approximately 32GB RAM.
Stock GPU clocks and power settings. ComfyUI was stopped during these runs.

The practical result is a verified 262,144-token single-session configuration,
with 260,088 actual input tokens, using calibrated Q4 KV. The everyday default
remains a single 65,536-token FP16 session. Large contexts reduce decode speed;
they are not a free speed upgrade.

## Model and runtime identity

- Weights: `Ternary-Bonsai-2-27B-PQ2_0.gguf`, 7,206,168,928 bytes.
- Projector: `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`.
- Runtime asset: `llama-prism-b10777-e3de715-bin-win-cuda-13.3-x64.zip`.
- CUDA DLL asset: `cudart-llama-bin-win-cuda-13.3-x64.zip`.
- Both archives passed the SHA256 digests published by GitHub. This release
  includes the quantized K/V short-query CUDA FA repair in Prism PR #355.
- Model/bias SHA256, calibration settings, profile snapshots and the redacted
  actual max-context process command are in
  [launch-configurations.json](rtx5070ti-20261019/launch-configurations.json).
- PTQ1_0, speculative/MTP decoding, overclocking, and contexts beyond the model's
  native 262,144-token limit were not tested.

## llama-bench microbatch sweep

No server was resident during this sweep. Three repetitions per test, full
offload, Flash Attention, FP16 KV, four CPU threads, polling disabled:

```powershell
.\bin\cuda\llama-bench.exe -m .\models\bonsai2-gguf\27B\Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -fa on -p '512,2048' -n 128 -ub '256,512,1024,2048' -b 2048 -t 4 --poll 0 -r 3 -o json
```

| Microbatch | PP512 t/s | PP2048 t/s | TG128 t/s |
| ---: | ---: | ---: | ---: |
| 256 | 1974.41 +/- 2.73 | 1973.68 +/- 0.87 | 85.71 +/- 0.29 |
| 512 | 2016.01 +/- 35.73 | 2029.13 +/- 2.33 | 85.19 +/- 0.05 |
| 1024 | 2020.38 +/- 38.49 | 1988.18 +/- 1.13 | 85.62 +/- 0.36 |
| 2048 | 1982.06 +/- 97.42 | 1964.00 +/- 6.80 | 85.09 +/- 0.15 |

Larger microbatches did not improve PP2048. Balanced/throughput/long retain 512;
max uses 256 to constrain working buffers. The max profile was not compared
against 512 at the same 256K context. The raw three samples per case, build,
backend and model metadata are preserved in
[ubatch-sweep.json](rtx5070ti-20261019/ubatch-sweep.json).

## Actual near-limit context and cache reuse

Start one profile at a time, on an isolated local port:

```powershell
.\scripts\make_kv_bias.ps1
.\run_server.ps1 -Profile balanced -Port 18080 -NoBrowser -NoTunnel
# After stopping the current server, replace balanced with long or max.
```

For the matching context size, with BONSAI_API_KEY set locally:

```powershell
python scripts/benchmark_context.py --context-size 65536 --reserve-tokens 2048 --base-url http://127.0.0.1:18080 --server-binary bin/cuda/llama-server.exe --output performance-results/context-64k-f16.json
python scripts/benchmark_context.py --context-size 131072 --reserve-tokens 2048 --base-url http://127.0.0.1:18080 --server-binary bin/cuda/llama-server.exe --output performance-results/context-128k-q4.json
python scripts/benchmark_context.py --context-size 262144 --reserve-tokens 2048 --base-url http://127.0.0.1:18080 --server-binary bin/cuda/llama-server.exe --output performance-results/context-256k-q4.json
```

The complete chat template was tokenized before inference. Service usage matched
that count exactly. Each first turn retrieved three random values near 10%, 50%
and 90% of a synthetic inventory. The second turn replayed the full history and
returned the same exact JSON map. All six responses finished normally and passed.

| Profile | KV / projector | Actual first input | First prefill | First decode t/s | Follow-up prefill | Follow-up cached tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| balanced 64K | FP16 / GPU | 63,480 | 46.956s | 57.43 | 0.185s | 63,538 |
| long 128K | mean-centered Q4 / CPU | 129,016 | 115.735s | 36.43 | 0.214s | 129,074 |
| max 256K | mean-centered Q4 / CPU | 260,088 | 355.569s | 23.05 | 0.260s | 260,146 |

Only 32 new prompt tokens were evaluated on every follow-up. Follow-up total
response times, including the 59 output tokens, were 1.215s, 1.850s and 2.876s.
GPU total usage after each run was 13,684 / 11,090 / 13,322 MiB, including the
Windows desktop; these before/after observations are not continuous peak telemetry.
Q4 runs explicitly disabled K rotation to match calibration. Their model and bias
hashes were checked before startup. These measurements compare different context
lengths and cache types, not an isolated FP16-versus-Q4 speed A/B.

Evidence: [64K](rtx5070ti-20261019/context-64k-f16.json),
[128K](rtx5070ti-20261019/context-128k-q4.json),
[256K](rtx5070ti-20261019/context-256k-q4.json). Reports include the entire synthetic
prompt, outputs, token counting and timing evidence. These are retrieval/history
checks, not a guarantee of reasoning quality on every long document. Input,
thinking, tool schemas and output must all fit the context; the 2,048-token
reserve in this test should be increased for longer answers.

## API throughput and runtime baseline

The updated runtime's short-prompt API decode median was 84.08 t/s; the previous
b10772 run was 81.92 t/s. Their whole-request throughput was 83.08 versus 81.03
t/s. This small observed difference is not established as a causal upgrade gain:
one cell per build, randomized cold-prefix markers, and 4/4 versus 3/4 length
stops. Raw evidence: [before](rtx5070ti-20261019/old-b10772-balanced64k.json),
[after](rtx5070ti-20261019/new-b10777-balanced64k.json).

For throughput, the 32K profile uses four slots sharing one KV pool. Eight
requests per concurrency cell used the same shuffled question set and seeds,
thinking off, cold cache, and a 384-token output cap:

```powershell
.\run_server.ps1 -Profile throughput -Port 18080 -NoBrowser -NoTunnel
python scripts/benchmark_throughput.py --base-url http://127.0.0.1:18080 --modes off --concurrency 1,2,4 --requests 8 --warmup 1 --repeat 1 --cache cold --max-tokens 384 --server-binary bin/cuda/llama-server.exe --label new-b10777-throughput32k --output performance-results/concurrency-throughput32k.json
```

| Concurrent requests | Aggregate t/s | Per-stream decode median | Request latency p50 |
| ---: | ---: | ---: | ---: |
| 1 | 81.09 | 84.55 | 4.735s |
| 2 | 130.77 | 72.15 | 5.920s |
| 4 | 170.83 | 47.72 | 8.991s |

All 24 requests succeeded at transport level and reached the output cap. These
numbers measure fixed-budget throughput, not completed-answer latency or answer
quality. This is a small eight-request-per-cell run, not a sustained load test.
[Full output](rtx5070ti-20261019/concurrency-throughput32k.json).

## Capability regression

```powershell
python scripts/smoke_local_api.py --base-url http://127.0.0.1:18080 --model bonsai --cases text tools vision --output performance-results/capabilities-throughput32k.json
```

All passed: final text answer, native inventory tool retry/round-trip with the
correct calculated result, and identification of an inline synthetic red image.
[Capability report](rtx5070ti-20261019/capabilities-throughput32k.json).
