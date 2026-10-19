# Windows 本地与公网服务

`run_server.bat` 和 `run_server.ps1` 共用 `scripts/start_llama_server.ps1`。
本地入口默认使用 balanced 配置；标准启动器不指定配置档时仍保留原来的自动内存分档。

## 启动与停止

```powershell
.\run_server.ps1 -Profile balanced
.\run_server.ps1 -Profile throughput
.\run_server.ps1 -Profile long
.\run_server.ps1 -Profile max
.\stop_server.ps1
```

双击 `run_server.bat` 使用 balanced；BAT 也接受相同参数。
停止脚本只处理本仓库目录内的 llama-server 和 FRP 进程，保留其他项目的服务。
`-KeepTunnel` 可在停止模型服务时保留隧道。

`-NoBrowser` 跳过打开浏览器，`-NoTunnel` 不自动启动 FRP。
独立测试可以使用 `-Port 18080 -NoBrowser -NoTunnel`。
发现端口已有服务时，启动器会退出；切换配置档需要先停止原服务。

## 配置档

这些配置以 RTX 5070 Ti 16GB、32GB 系统内存为调优目标。其他硬件应先测显存和响应 timings。
配置来源为 `scripts/server-profiles.json`；模型权重仍使用 Bonsai 2 27B PQ2_0。

| 配置档 | 总上下文池 | 槽位 | KV | 视觉投影器 | 用途 |
| --- | ---: | ---: | --- | --- | --- |
| balanced | 65,536 | 1 | FP16 | GPU | 单人聊天与长文 |
| throughput | 32,768 | 4，共享池 | FP16 | GPU | 多个短请求的总吞吐 |
| long | 131,072 | 1 | 校准 Q4_0 | CPU | 较长输入 |
| max | 262,144 | 1 | 校准 Q4_0 | CPU | 模型原生上下文上限 |

总池不是每个用户独享的配额。throughput 使用 `--kv-unified`，所有活跃请求的 KV 共同占用池容量。
输入、聊天模板、工具定义、思考和正文输出共同消耗上下文。不要把上下文全填成输入。
long/max 需要先更新运行时并校准偏置；容量配置成功不代表所有长文任务质量合格。
Q4 KV 主要节省显存，不能当作速度提升保证；CPU 视觉投影器会让图片编码变慢。

配置档使用模板的 `reasoning_effort=medium`，保留思考能力。UI 的“Medium”是另一种 2,048-token 思考预算。
通过 UI、请求中的 `thinking_budget_tokens` 或原生 `--reasoning-budget` 可进一步限制等待时间。
CUDA 图片默认保持完整细节，没有额外的 1,024-token 图像上限。

## 覆盖参数与预览

优先级为原生命令行参数、`BONSAI_*` 环境变量、配置档默认值。

```powershell
$env:BONSAI_CTX = '49152'
$env:BONSAI_PARALLEL = '1'
.\run_server.ps1 -Profile balanced -ub 1024 --reasoning-budget 2048
.\run_server.ps1 -Profile balanced -DryRun
```

`-DryRun` 输出将传给二进制的参数，不启动模型、浏览器或隧道，密钥会被隐藏。
`BONSAI_PROFILE`、`BONSAI_PORT`、`BONSAI_NGL`、`BONSAI_KV4`、`BONSAI_MMPROJ_CPU`
和 `BONSAI_IMAGE_MAX_TOKENS` 也可覆盖相应设置。设置 `BONSAI_CTX=0` 恢复所选配置档的自动默认值。
普通参数直接透传；如要精确控制思考模板，可传 `--chat-template-kwargs` JSON。

## 本地密钥与 FRP

复制 `.bonsai-server.example.json` 为 `.bonsai-server.local.json`，只在本地填入 API 密钥和隧道偏好。
也可以使用 `BONSAI_API_KEY` 环境变量，其优先级高于本地 JSON。
`.bonsai-server.local.json` 和 `frp/` 均在 `.gitignore` 中；禁止把真实密钥写回脚本或文档。

如需沿用现有公网隧道，在本地 JSON 设置 `EnableTunnel=true`，并保留 `frp/frpc.exe` 与
`frp/frpc.toml`。没有 API 密钥时启动器拒绝自动开启隧道。
隧道进程在后台隐藏运行。公网域名、网关与 TLS 仍由已有的本地 FRP 和云端配置管理。

本地 Web UI 为 `http://localhost:8080`；API base URL 为 `http://127.0.0.1:8080/v1`。
客户端使用本地配置中的相同密钥；如通过公网访问，使用自己的 HTTPS 域名。
浏览器内保存的 UI 设置可能覆盖服务端采样和工具默认值。

## 运行时与长上下文

```powershell
.\stop_server.ps1 -KeepTunnel
.\scripts\download_cuda_bin.ps1 -CudaTag 13.3
.\scripts\make_kv_bias.ps1
.\run_server.ps1 -Profile long
```

下载器使用 `scripts/llama-release.txt` 的固定版本，校验官方 SHA256，并在暂存目录验证后替换运行时。
失败会保留原安装。不要使用 stock llama.cpp 替换 Bonsai 2 的 Prism 二进制。
Windows 校准细节见 [KV-CACHE.md](KV-CACHE.md)。

显存中的其他模型会影响速度和可用上下文。特别是 ComfyUI 与 LLM 同时驻留时，应关注专用显存和系统内存，
避免驱动将模型数据换入共享内存。先确认其他 GPU 任务已释放权重，再比较配置档。

## 可复现测量

本机 64K/128K/256K、微批扫描及并发测量见
[RTX 5070 Ti 实测报告](community-benchmarks/bonsai2/cuda-rtx5070ti-16gb-windows.md)。

压测通过环境变量读取密钥。报告保留完整生成输出，因此只应使用公开或合成输入。

```powershell
$env:BONSAI_BASE_URL = 'http://127.0.0.1:8080'
# BONSAI_API_KEY 只在本地环境中设置。
python scripts/benchmark_throughput.py --modes off --concurrency 1,2,4 --requests 8 --repeat 3 --cache cold --output performance-results/throughput.json
python scripts/benchmark_throughput.py --modes medium --budget 2048 --concurrency 1 --requests 4 --cache warm --output performance-results/thinking.json
python scripts/benchmark_context.py --context-size 131072 --reserve-tokens 2048 --output performance-results/context-128k.json
```

比较完成请求数、截断数、TTFT、首正文时间、p50/p95、服务端 decode 速度和总吞吐。
固定输出预算下的 tok/s 不能替代完整答案质量与耗时；`finish_reason=length` 会被明确计为截断。
冷热缓存分开测试，跨配置比较时保持题集、种子和输出预算一致。
