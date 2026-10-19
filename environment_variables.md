# Environment variables

Complete reference for the demo's user-configurable environment variables. The [README](README.md#environment-variables) covers the common ones; everything is listed here.

Every script in this repo is driven by environment variables — model selection and server behavior are all configured this way. The reference below covers the demo's **own** user-configurable variables (internal outputs the scripts set for themselves, like `BONSAI_DISPLAY`, `BONSAI_MCP_IDS`, `BONSAI_DEMO_DB`, or `BONSAI_CODE_INTERPRETER_ON`, are not listed). They're read by `setup.sh`, `setup.ps1`, `download_models.sh`, and the `run_*` / `start_*` launchers (Linux, macOS, and Windows). The build scripts take CLI flags rather than env vars (see below), and the llama.cpp/MLX runtimes accept a few env vars of their own that are not listed here — for example, M5 Macs may need `GGML_METAL_TENSOR_DISABLE=1` (see the [FAQ](FAQ.md)).

| Variable | Default | Valid values | Purpose |
|----------|---------|--------------|---------|
| **Model & setup** | | | |
| `BONSAI_FAMILY` | `bonsai2` | `bonsai2`, `ternary`, `bonsai`, `all` | Model family. `bonsai2` = Bonsai 2; `ternary` = earlier Ternary-Bonsai; `bonsai` = 1-bit Bonsai. `all` expands to all three families (setup/download only). |
| `BONSAI_MODEL` | `27B` | `27B`, `8B`, `4B`, `1.7B`, `all` | Model size. `all` expands to all four sizes (setup/download only). |
| `BONSAI_TOKEN` | — | HF read-only token | No longer needed: all model repos are public. Kept for compatibility; if set, it is passed to the HF downloads. |
| `BONSAI_SKIP_GGUF` | unset | `1` | Skip the GGUF download entirely (macOS MLX-only setups, saves disk space). The llama.cpp scripts then point you at the MLX ones instead (see "Running the Model" below). |
| `BONSAI_SKIP_MLX` | unset | `1` | Skip the MLX download (macOS only; MLX is skipped automatically on Intel Macs and non-macOS). |
| `BONSAI_OPENWEBUI` | `1` | `0` | Skip installing Open WebUI during `setup.sh`. (`setup.sh` only installs it — the demo is launched separately with `./scripts/start_openwebui.sh`.) |
| **llama.cpp server** | | | |
| `BONSAI_GGUF` | unset | path to a `.gguf` | Run any GGUF directly with `start_llama_server.sh`, skipping the family/size lookup and the download check (relative paths resolve against the demo dir). Drafters and kv-bias files are picked up from the same folder as the file. |
| `BONSAI_MMPROJ` | unset | path to an mmproj `.gguf` | Vision projector to pair with `BONSAI_GGUF`. Without it a custom model runs text-only. |
| `BONSAI_HOST` | `127.0.0.1` | any bind address | Bind address. For `start_llama_server.sh` this is the llama-server's `--host`. For `start_openwebui.sh` it binds the **Open WebUI** UI instead (its managed llama-server stays on `127.0.0.1`), so a non-loopback value exposes the unauthenticated UI/code interpreter — that requires opt-in via `BONSAI_ALLOW_REMOTE=1` (trusted networks only). |
| `BONSAI_CTX` | RAM-tiered, or selected Windows profile | `0`, or ≤ `262144` | Total context pool. `0`/unset uses the selected Windows profile's pool, or automatic RAM tiers without a profile (never `-c 0`). An explicit number forces it; the Windows launcher accepts 512..262144. |
| `BONSAI_LLAMA_BIN` | unset (search the repo's builds) | directory containing `llama-server` | Use a prebuilt PrismML-fork build elsewhere on disk instead of the one `setup.sh` built. macOS/Linux launchers only. |
| `BONSAI_NGL` | auto-detect | any int; `0` = CPU-only | Override GPU layer offload. Auto-detect keys on installed tooling, so weak iGPUs can be better with `0`. |
| `BONSAI_IMAGE_MAX_TOKENS` | `1024` on Metal/Vulkan/CPU; uncapped on CUDA/ROCm | number; `0` = uncapped | Cap on vision tokens per image (27B). `0` restores full detail (best for OCR / screenshots / small text) but is slower on large images. |
| `BONSAI_MMPROJ_CPU` | unset; enabled by Windows `long`/`max` | `0`, `1` | Keep the 27B vision projector in system RAM instead of VRAM (`--no-mmproj-offload`), freeing ~0.9 GiB for KV/context; slower image prefill. Explicit `0` disables a Windows profile's CPU projector setting. |
| `BONSAI_SPECULATIVE` | `0` | `1` | Enable speculative decoding with the paired DSpark drafter for previous-generation `ternary`/`bonsai` 27B only. Bonsai 2 has no official drafter yet; its family launchers warn and run without speculation. The shell launcher’s explicit `BONSAI_GGUF` custom-model path retains experimental drafter discovery. Previous-generation measurements (CUDA: 1.8-2.4x decode for ternary, 1.4-1.75x for 1-bit, code/math best; not recommended on Apple Silicon — only ternary code/math gains there). Opt-in, server-only. [SPECULATIVE.md](SPECULATIVE.md) |
| `PORT` | `8080` | port number | Port for `start_llama_server.sh`; Windows uses `BONSAI_PORT`. |
| `BONSAI_SPEC_NMAX` | `4` | int | dspark draft n-max override (PowerShell scripts only). |
| `BONSAI_KV4` | `0`; enabled by Windows `long`/`max` | `0`, `1` | 4-bit (Q4_0) KV cache, ~3.5x less KV memory for very long contexts; decode slightly slower than F16. Explicit `0` disables profile quantization. Calibrate with `./scripts/make_kv_bias.sh` or `.\scripts\make_kv_bias.ps1` on Windows; Windows `long`/`max` require a matching bias and model identity metadata while Q4 is enabled. [KV-CACHE.md](KV-CACHE.md) |
| **Windows profiles and local service** | | | |
| `BONSAI_PROFILE` | `balanced` for `run_server.ps1`/BAT; `auto` for `start_llama_server.ps1` | `auto`, `balanced`, `throughput`, `long`, `max` | Select Windows launch defaults from `scripts/server-profiles.json`; explicit `-Profile` wins. `auto` keeps RAM-tiered context. See the [Windows guide](README_PUBLIC_SERVER.md). |
| `BONSAI_PORT` | `8080` | `1`..`65535` | Windows server port. `-Port` or native `--port` overrides it. |
| `BONSAI_PARALLEL` | selected profile, otherwise runtime default | positive integer | Override Windows server slots. `throughput` shares its total pool with `--kv-unified`; more slots do not multiply total capacity. Active speculative decoding still forces one slot unless overridden with native arguments. |
| `BONSAI_API_KEY` | unset | API key | Windows server authentication and benchmark client key. `run_server.ps1` falls back to `ApiKey` in the gitignored `.bonsai-server.local.json`; an environment key wins. Benchmark tools fall back to `OPENAI_API_KEY` only when `BONSAI_API_KEY` is unset. |
| **Benchmark clients (`benchmark_throughput.py`, `benchmark_context.py`)** | | | |
| `BONSAI_BASE_URL` | `OPENAI_BASE_URL`, then `http://127.0.0.1:8080` | HTTP(S) server URL | Benchmark server address; accepts a root URL or a trailing `/v1`. Explicit `--base-url` wins. This does not change the server's bind address or port. |
| `BONSAI_API_MODEL` | `bonsai` | model id | Benchmark request model; explicit `--model` wins. This does not select or load model weights (`BONSAI_MODEL` does that). There is no `OPENAI_MODEL` fallback. |
| `OPENAI_BASE_URL` | unset | HTTP(S) server URL | Benchmark compatibility fallback when `BONSAI_BASE_URL` is unset. |
| `OPENAI_API_KEY` | unset | API key | Benchmark compatibility fallback when `BONSAI_API_KEY` is unset; it does not configure Windows server authentication. |
| **Agentic demo (`start_agent_server.sh`, AGENT-DEMO.md)** | | | |
| `AGENT_REASONING_BUDGET` | `16384` | tokens, `-1` = unlimited | Thinking budget per turn (`--reasoning-budget`); the server force-closes thinking at this count. |
| `AGENT_MIN_P` | `0.05` | `0` to `1` | Sampler min-p for the agent profile; the model card's thinking-mode value. The skateboard recording used `0`. |
| `AGENT_MODEL_ALIAS` | `bonsai2-27b-pq2-v16_2` | any id | Model id the server reports (`--alias`); part of the system prompt Hermes builds. |
| `AGENT_SERVER_SEED` | `42` | integer, empty = none | Server-side sampler seed (`-s`). The demo runner expects it to equal `AGENT_SEED` unless it runs with `AGENT_TRACE=1`. |
| `AGENT_UPSTREAM` | `127.0.0.1:8080` | `host:port` | (`run_agent_demo.sh`) The llama-server to drive; may be on another machine. |
| `AGENT_MODEL` | first id from `/v1/models` | model id | (`run_agent_demo.sh`) Model id to expect and send; the runner aborts if the server does not report it. |
| `AGENT_SEED` | `42` | integer, empty = skip | (`run_agent_demo.sh`) Sampler seed the run expects on the server (or injects with `AGENT_TRACE=1`). |
| `AGENT_EFFORT` | unset (template default, xhigh) | `xhigh`, `medium` | (`run_agent_demo.sh`) Reasoning effort; needs `AGENT_TRACE=1` or the server's `--chat-template-kwargs`. |
| `AGENT_CONFIG` | per mode | path to a Hermes yaml | (`run_agent_demo.sh`) Hermes config to use instead of `hermes-config-round0.yaml` / `hermes-config-feedback.yaml`. |
| `AGENT_CTX` | `131072` | ≤ the server's per-slot context | (`run_agent_demo.sh`) `context_length` written into the Hermes config. |
| `AGENT_WORKSPACE_ROOT` | `<system temp>/bonsai-agent` | absolute path | (`run_agent_demo.sh`) Where the agent's working directory is created; must be outside any git repo. |
| `AGENT_HOME_ROOT` | `agent-runs/` | path | (`run_agent_demo.sh`) Where the run's private `HERMES_HOME` is created. |
| `AGENT_TRACE` | `0` | `0`, `1` | (`run_agent_demo.sh`) `1` records every request/response to `agent-runs/<run>/wire.jsonl` through a local logging proxy and injects `AGENT_SEED`/`AGENT_EFFORT` per request. |
| **MLX server** | | | |
| `BONSAI_BACKEND` | `llama` | `llama`, `mlx` | Which backend `start_openwebui.sh` serves (`mlx` is Apple Silicon-only). It does **not** change `run_llama.sh` / `run_llama.ps1` / `run_mlx.sh` — those pick their backend by which script you invoke. |
| `BONSAI_MLX_VLM` | `1` | `0`, `1` | Set up/use the native mlx-vlm environment for Bonsai 2 and the older ternary 27B. Required for Bonsai 2: setting `0` skips setup and its MLX launchers refuse to run, rather than fall back to an incompatible loader. For the older ternary family, `0` selects text-only mlx_lm. |
| `BONSAI_MLX_VISION` | `0` | `1` | Force MLX vision for a pre-existing MLX server whose implementation isn't known (set it explicitly if you started it with mlx-vlm). |
| **Open WebUI** | | | |
| `BONSAI_ALLOW_REMOTE` | `0` | `1` | Allow binding Open WebUI to a non-loopback `BONSAI_HOST`. Auth is disabled + a code interpreter may be enabled, so this is trusted-networks-only. |
| `BONSAI_LOG` | `1` | `0` | Discard run logs to `/dev/null` instead of writing fresh timestamped logs under `.openwebui/logs/`. |
| `BONSAI_LLAMA_VERBOSE` | `0` | `1` | Run llama-server with `-v` for full request-body debugging (very noisy; for diffing tool-call rounds). |
| `BONSAI_MAX_TOOL_ITERS` | `30` | int; `-1` = unlimited | Cap on the native tool-call loop rounds in Open WebUI (default 30; `-1` unbounded). |
| `BONSAI_CODE_INTERPRETER` | `1` | `0` | Disable the server-side Jupyter code interpreter (falls back to the browser Pyodide — plots still work, but no yfinance/network). |
| `BONSAI_JUPYTER_PORT` | `8888` | port | Port for the Jupyter code interpreter kernel. |
| `BONSAI_BRAVE_TOOLS` | `brave_web_search brave_news_search brave_summarizer` | space-separated tool names | Which Brave MCP tools to expose (~2.9k prompt tokens for the default three; the full set is ~29k). |
| `BRAVE_API_KEY` | — | Brave Search API key | Enables the Brave web-search MCP server (or a gitignored `.brave_key` file; needs `npm i -g @brave/brave-search-mcp-server`). [TOOLS.md](TOOLS.md) |

`all` is only valid for `setup.sh` / `setup.ps1` / `download_models.sh` — the run/server scripts need a concrete family/size. Extra llama-server flags (e.g. `--reasoning-budget N`, `-ub 1024`) pass straight through as trailing arguments to the **standalone llama-server launchers** (`start_llama_server.sh` / `start_llama_server.ps1`); `start_openwebui.sh` forwards its trailing arguments to `open-webui serve` instead.

For Windows profiles, native CLI arguments override `BONSAI_*` environment
variables, which override profile defaults. An unset or `0` `BONSAI_CTX` uses the
selected profile's context rather than RAM tiers; choose `-Profile auto` for RAM
tiers. Explicit `BONSAI_KV4=0` and `BONSAI_MMPROJ_CPU=0` disable those profile
features. Profiles and `BONSAI_PROFILE`, `BONSAI_PORT`, `BONSAI_PARALLEL` server
settings are Windows-only; benchmark variables work on all platforms. The shell
server launchers do not consume `BONSAI_API_KEY`. Neither benchmark tool reads
`.bonsai-server.local.json`, so export its key separately when benchmarking an
authenticated server. Set the alias variables only when their `BONSAI_*`
counterparts are absent; an explicitly empty `BONSAI_*` benchmark value does not
fall back to an `OPENAI_*` alias.

The build scripts take their options as **command-line flags, not environment variables** (e.g. `./scripts/build_cuda_linux.sh [repo_dir] --archs "80;86" --output cuda`, or `.\scripts\build_cuda_windows.ps1 -Archs "80;86;89;90"`) — see the README's "Building from Source" section.

**Platform coverage:** the model/setup vars (`BONSAI_FAMILY`, `BONSAI_MODEL`, `BONSAI_TOKEN`) and the llama.cpp server vars (`BONSAI_HOST`, `BONSAI_CTX`, `BONSAI_NGL`, `BONSAI_IMAGE_MAX_TOKENS`, `BONSAI_MMPROJ_CPU`, `BONSAI_SPECULATIVE`, `BONSAI_SPEC_NMAX`, `BONSAI_KV4`) work on **Linux, macOS, and Windows** (the `.ps1` launchers mirror the `.sh` ones). The **MLX** vars (`BONSAI_BACKEND=mlx`, `BONSAI_MLX_VLM`, `BONSAI_MLX_VISION`, `BONSAI_SKIP_MLX`) and the **Open WebUI** vars (`BONSAI_ALLOW_REMOTE`, `BONSAI_LOG`, `BONSAI_LLAMA_VERBOSE`, `BONSAI_MAX_TOOL_ITERS`, `BONSAI_CODE_INTERPRETER`, `BONSAI_JUPYTER_PORT`, `BONSAI_BRAVE_TOOLS`, `BRAVE_API_KEY`) are **macOS/Linux only** — MLX is Apple Silicon-only, and Open WebUI has no Windows launcher (`start_openwebui.sh` only; `setup.ps1` doesn't install it), so those vars have no effect on Windows.

