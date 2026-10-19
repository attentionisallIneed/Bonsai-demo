[CmdletBinding(PositionalBinding=$false)]
param(
    [ValidateSet("auto", "balanced", "throughput", "long", "max")]
    [string]$Profile = "",
    [int]$Port = 0,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ServerArguments
)

$ErrorActionPreference = "Stop"
$ExtraArgs = @($ServerArguments | Where-Object { $null -ne $_ })
if (-not $Profile) { $Profile = if ($env:BONSAI_PROFILE) { $env:BONSAI_PROFILE } else { "auto" } }
$ProfileConfig = $null
if ($Profile -ne "auto") {
    $Profiles = Get-Content -LiteralPath (Join-Path $PSScriptRoot "server-profiles.json") -Raw | ConvertFrom-Json
    $ProfileConfig = $Profiles.PSObject.Properties[$Profile].Value
    if (-not $ProfileConfig) { throw "Unknown BONSAI_PROFILE '$Profile'." }
}


$BonsaiModel  = if ($env:BONSAI_MODEL)  { $env:BONSAI_MODEL.ToUpperInvariant() } else { "27B" }
$BonsaiFamily = if ($env:BONSAI_FAMILY) { $env:BONSAI_FAMILY.ToLowerInvariant() } else { "bonsai2" }

if ($BonsaiModel -notin @("27B", "8B", "4B", "1.7B")) {
    Write-Host "[ERR] Unknown BONSAI_MODEL='$BonsaiModel'. Valid values: 27B, 8B, 4B, 1.7B" -ForegroundColor Red
    exit 1
}
if ($BonsaiFamily -notin @("bonsai2", "bonsai", "ternary")) {
    Write-Host "[ERR] Unknown BONSAI_FAMILY='$BonsaiFamily'. Valid values: bonsai2, bonsai, ternary" -ForegroundColor Red
    exit 1
}
# Bonsai 2 is 27B. Without this, another size walks into a directory that was
# never going to exist and the error blames setup. Mirrors BONSAI2_SIZES in
# scripts/common.sh.
$Bonsai2Sizes = if ($env:BONSAI2_SIZES) { $env:BONSAI2_SIZES -split '\s+' } else { @("27B") }
if ($BonsaiFamily -eq "bonsai2" -and $BonsaiModel -notin $Bonsai2Sizes) {
    Write-Host "[ERR] Bonsai 2 is 27B." -ForegroundColor Red
    Write-Host "      Bonsai 2:     `$env:BONSAI_MODEL='27B'; .\scripts\start_llama_server.ps1" -ForegroundColor Yellow
    Write-Host "      Other sizes:  `$env:BONSAI_FAMILY='ternary'; .\scripts\start_llama_server.ps1" -ForegroundColor Yellow
    exit 1
}

$DemoDir = Split-Path $PSScriptRoot -Parent
Set-Location $DemoDir

# Bind to localhost by default; override with BONSAI_HOST=0.0.0.0 for LAN/remote.
$HostAddress = if ($env:BONSAI_HOST) { $env:BONSAI_HOST } else { "127.0.0.1" }
if ($Port -eq 0) { $Port = if ($env:BONSAI_PORT) { [int]$env:BONSAI_PORT } else { 8080 } }
if ($Port -lt 1 -or $Port -gt 65535) { throw "Port must be between 1 and 65535." }
for ($i = 0; $i -lt $ExtraArgs.Count - 1; $i++) {
    if ($ExtraArgs[$i] -in @("--port")) { $Port = [int]$ExtraArgs[$i + 1] }
}

if (-not $DryRun) { try {
    $null = Invoke-WebRequest -Uri "http://localhost:$Port/health" -TimeoutSec 2
    Write-Host "[WARN] Health endpoint responded on port $Port; llama-server may already be running." -ForegroundColor Yellow
    exit 1
} catch {} }

# Captured unconditionally: the finally block at the end restores it on every
# launch, not just the ones where the KV4 bias block below sets it.
$priorRotDisable = $env:LLAMA_ATTN_ROT_DISABLE
try {

if ($BonsaiFamily -eq "bonsai2") {
    $ModelDir = Join-Path $DemoDir "models\bonsai2-gguf\$BonsaiModel"
    $FamilyDisplay = "Bonsai-2"
} elseif ($BonsaiFamily -eq "ternary") {
    $ModelDir = Join-Path $DemoDir "models\ternary-gguf\$BonsaiModel"

    $FamilyDisplay = "Ternary-Bonsai"
} else {
    $ModelDir = Join-Path $DemoDir "models\gguf\$BonsaiModel"
    $FamilyDisplay = "Bonsai"
}

$Display = "$FamilyDisplay-$BonsaiModel"
# Select exactly the demo quant for the family (a leftover F16 or g64 file
# must never be picked up).
$BinCandidates = @(
    "bin\cuda\llama-server.exe",
    "bin\hip\llama-server.exe",
    "bin\vulkan\llama-server.exe",
    "bin\cpu\llama-server.exe",
    "llama.cpp\build\bin\Release\llama-server.exe",
    "llama.cpp\build\bin\llama-server.exe"
)
$Model = $null
$BinRel = $null
foreach ($cand in $BinCandidates) {
    if (Test-Path (Join-Path $DemoDir $cand)) { $BinRel = $cand; break }
}
$Pq2Ready = $BinRel -notlike "bin\vulkan\*"
if ($BonsaiFamily -eq "bonsai2") {
    # Every Bonsai 2 band needs the fork's kernels, so there is no group-64 fallback.
    $tryPatterns = @("*-PQ2_0.gguf", "*-PTQ1_0.gguf")
} elseif ($BonsaiFamily -eq "ternary") {
    # PQ2_0 where the backend has kernels (see MODEL-FORMATS.md); official group-64
    # otherwise (*g64 on pre-v7 repos, plain *-Q2_0 on newer repos).
    # BONSAI_FORCE_G64=1 skips PQ2_0 regardless of backend.
    $tryPatterns = if ($env:BONSAI_FORCE_G64 -eq "1" -or -not $Pq2Ready) { @("*g64.gguf", "*-Q2_0.gguf") }
                   else { @("*-PQ2_0.gguf", "*g64.gguf", "*-Q2_0.gguf") }
} else {
    $tryPatterns = @("*-Q1_0.gguf")
}
foreach ($qp in $tryPatterns) {
    # plain *-Q2_0.gguf is only trusted with the downloader's .official-q2_0 marker
    # (newer repos); otherwise it is a deprecated legacy leftover v7 refuses to load
    if ($qp -eq "*-Q2_0.gguf" -and -not (Test-Path (Join-Path $ModelDir ".official-q2_0"))) { continue }
    $Model = Get-ChildItem -Path $ModelDir -Filter $qp -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike "*mmproj*" -and $_.Name -notlike "*dspark*" -and $_.Name -notlike "*kv-bias*" } |
        Select-Object -First 1
    if ($Model) { break }
}
if (-not $Model) {
    Write-Host "[ERR] No model file found for $Display in $ModelDir" -ForegroundColor Red
    Write-Host "      Run .\setup.ps1 first." -ForegroundColor Yellow
    exit 1
}

# Vision: use the multimodal projector when present (27B is a VLM).
$Mmproj = Get-ChildItem -Path $ModelDir -Filter *mmproj*.gguf -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($BonsaiModel -eq "27B" -and -not $Mmproj) {
    Write-Host "[WARN] No mmproj file found in $ModelDir - image input disabled." -ForegroundColor Yellow
    Write-Host "       Re-run setup.ps1 to fetch it." -ForegroundColor Yellow
}

if (-not $BinRel) {
    Write-Host "[ERR] llama-server.exe not found. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}

$Bin = Join-Path $DemoDir $BinRel
$BinDir = Split-Path $Bin -Parent
$env:Path = "$BinDir;$env:Path"

# Default context: RAM-tiered cap. BONSAI_CTX=0 or unset both mean "auto" ->
# this tiered default (never -c 0, which uses the model's full training context
# and OOMs constrained machines). Pass an explicit number to override.
$CtxDefault = if ($env:BONSAI_CTX -and $env:BONSAI_CTX -ne "0") { $env:BONSAI_CTX } elseif ($ProfileConfig) { "$($ProfileConfig.context)" } else {
    $MemGB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    if ($MemGB -le 11) { "8192" } elseif ($MemGB -le 23) { "16384" } elseif ($MemGB -le 35) { "32768" } elseif ($MemGB -le 71) { "65536" } elseif ($BonsaiModel -eq "27B") { "131072" } else { "65536" }
}
if ([int]$CtxDefault -lt 512 -or [int]$CtxDefault -gt 262144) { throw "BONSAI_CTX must be 512..262144 (0 = auto)." }

$Ngl = if ($env:BONSAI_NGL) {
    $env:BONSAI_NGL
} elseif ($BinRel -like "bin\cpu\*") {
    "0"
} else {
    "99"
}

Write-Host ""
Write-Host "=== llama.cpp server (GGUF) ==="
Write-Host "  Model:   $($Model.Name)"
Write-Host "  Binary:  $Bin"
$NglNote = if ($env:BONSAI_NGL) { "set via BONSAI_NGL" } else { "auto-detected; override with BONSAI_NGL, 0 = CPU-only" }
Write-Host "  GPU:     -ngl $Ngl ($NglNote)"
Write-Host ""
Write-Host "  Open http://localhost:$Port in your browser to chat."
Write-Host "  API:  http://localhost:$Port/v1/chat/completions"
Write-Host "  Press Ctrl+C to stop."
Write-Host ""

$ChatTemplateKwargs = if ($PSVersionTable.PSEdition -eq 'Desktop') { '{\"enable_thinking\": false}' } else { '{"enable_thinking": false}' }

# Sampling for the 27B path: Bonsai 2 uses the model card's thinking-mode values, the
# earlier families keep the profile they were tested on. Mirrors start_llama_server.sh.
$SamplingArgs = if ($BonsaiFamily -eq "bonsai2") {
    @("--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.05")
} else {
    @("--temp", "0.7", "--top-p", "0.95", "--top-k", "20", "--min-p", "0")
}

$ServerArgs = @(
    "-m", $Model.FullName,
    "--host", $HostAddress,
    "--port", "$Port",
    "-ngl", $Ngl, "-fa", "on",
    "-c", $CtxDefault,
    "--temp", "0.5",
    "--top-p", "0.85",
    "--top-k", "20",
    "--min-p", "0",
    "--reasoning-budget", "0",
    "--reasoning-format", "none",
    "--chat-template-kwargs", $ChatTemplateKwargs
)

# 27B: --jinja enables native OpenAI-style tool calling; --mmproj enables
# image input; sampling matches the 27B reference demo (temp 0.7, top-p 0.95);
# thinking stays enabled (reasoning overrides removed).
# Older sizes keep the exact flag set they were tested with.
if ($BonsaiModel -eq "27B") {
    # Speculative decoding (opt-in, BONSAI_SPECULATIVE=1): pair the target with
    # its dspark drafter for ~1.8-2x decode on code/reasoning. Disables
    # prompt-cache reuse and forces a single slot, so it is off by default and
    # lives on this standalone server, not the agentic Open WebUI path.
    $Ctx = $CtxDefault
    $SpecArgs = @()
    if ($env:BONSAI_SPECULATIVE -eq "1" -and $BonsaiFamily -eq "bonsai2") {
        Write-Host "[WARN] Bonsai 2 27B has no official DSpark drafter released yet; running without speculation." -ForegroundColor Yellow
        Write-Host "       Unset BONSAI_SPECULATIVE to hide this warning. See SPECULATIVE.md for availability." -ForegroundColor Yellow
    } elseif ($env:BONSAI_SPECULATIVE -eq "1") {
        # v7 builds read only converted (arch=dflash) drafters; convert the downloaded
        # bf16 sidecar once with gguf-dspark-to-dflash (see SPECULATIVE.md)
        $Drafter = Get-ChildItem -Path $ModelDir -Filter *dspark-dflash*.gguf -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($Drafter) {
            $Nmax = if ($env:BONSAI_SPEC_NMAX) { $env:BONSAI_SPEC_NMAX } else { "4" }
            $SpecArgs = @("-md", $Drafter.FullName, "--spec-type", "draft-dspark", "--spec-draft-n-max", $Nmax, "-ngld", "999", "-np", "1")
            # dspark re-prefills every request; give the model room to think.
            # An explicit non-zero BONSAI_CTX still wins; unset or 0 (auto) gets
            # this dspark-friendly floor. Explicit profile contexts are preserved.
            if (-not $ProfileConfig -and (-not $env:BONSAI_CTX -or $env:BONSAI_CTX -eq "0")) {
                $Ctx = "$([math]::Max([int]$Ctx, 16384))"
            }
            Write-Host "  Speculative: $($Drafter.Name) (draft-dspark, n-max $Nmax)" -ForegroundColor Green
        } else {
            Write-Host "[WARN] BONSAI_SPECULATIVE=1 but no converted *dspark-dflash*.gguf drafter in $ModelDir - running without speculation." -ForegroundColor Yellow
            Write-Host "       Convert the downloaded bf16 drafter once (see SPECULATIVE.md, section 'Converting the published drafter')." -ForegroundColor Yellow
        }
    }
    # 4-bit KV cache (opt-in, BONSAI_KV4=1): stores the KV cache in Q4_0 to cut
    # KV memory for very long contexts on tight machines (decode is slightly
    # slower than F16 KV). If a mean-centering bias built by
    # scripts/make_kv_bias.sh is present it is applied automatically for
    # better quality. Mirrors start_llama_server.sh.
    $KvArgs = @()
    $UseKv4 = if ($null -ne $env:BONSAI_KV4) { $env:BONSAI_KV4 -eq "1" } else { $ProfileConfig -and $ProfileConfig.kv4 }
    if ($UseKv4) {
        # Known prebuilt CUDA builds contain a quantized-KV out-of-bounds write.
        # Unknown/custom builds and other backends are not identified by this marker.
        $ReleaseMarker = Join-Path $BinDir ".llama_release"
        if (-not $DryRun -and $BinRel -like "bin\cuda\*" -and (Test-Path -LiteralPath $ReleaseMarker)) {
            $Release = Get-Content -LiteralPath $ReleaseMarker -Raw
            if ($Release -match 'prism-b(\d+)' -and [int]$Matches[1] -gt 10709 -and [int]$Matches[1] -lt 10776) {
                throw "CUDA build prism-b$($Matches[1]) has a known quantized-KV memory bug. Run .\scripts\download_cuda_bin.ps1 before using BONSAI_KV4=1 or a long/max profile."
            }
        }
        $KvArgs = @("--cache-type-k", "q4_0", "--cache-type-v", "q4_0")
        $BiasPath = Join-Path $ModelDir ($Model.BaseName + "-kv-bias.gguf")
        $KvBias = Get-Item -LiteralPath $BiasPath -ErrorAction SilentlyContinue
        # A family-level legacy filename cannot identify a particular weight pack.
        if (-not $KvBias) {
            $LegacyBias = Join-Path $ModelDir ($Display + "-kv-bias.gguf")
            $LegacyMetadata = $LegacyBias + ".json"
            if (Test-Path -LiteralPath $LegacyBias -PathType Leaf) {
                if (Test-Path -LiteralPath $LegacyMetadata -PathType Leaf) {
                    $KvBias = Get-Item -LiteralPath $LegacyBias
                } else {
                    Write-Warning "Ignoring legacy KV bias without model metadata: $LegacyBias. Run .\scripts\make_kv_bias.ps1."
                }
            }
        }
        if (-not $KvBias -and $Profile -in @("long", "max") -and -not $DryRun) {
            throw "Profile '$Profile' requires a calibrated KV bias. Run .\scripts\make_kv_bias.ps1 first."
        }
        if ($KvBias) {
            $MetadataPath = $KvBias.FullName + ".json"
            $Recalibrate = "Run .\scripts\make_kv_bias.ps1 to regenerate the bias for these weights."
            if (Test-Path -LiteralPath $MetadataPath -PathType Leaf) {
                try { $BiasMetadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json }
                catch { throw "Cannot read KV bias metadata '$MetadataPath'. $Recalibrate" }
                if ($BiasMetadata -isnot [pscustomobject] -or $BiasMetadata.model_file -isnot [string] -or
                    $BiasMetadata.model_file -cne $Model.Name -or
                    ($BiasMetadata.model_size -isnot [int] -and $BiasMetadata.model_size -isnot [long]) -or
                    $BiasMetadata.model_size -ne $Model.Length -or
                    $BiasMetadata.model_sha256 -isnot [string] -or $BiasMetadata.bias_sha256 -isnot [string] -or
                    $BiasMetadata.model_sha256 -notmatch '^[a-fA-F0-9]{64}$' -or
                    $BiasMetadata.bias_sha256 -notmatch '^[a-fA-F0-9]{64}$' -or
                    $BiasMetadata.k_rot -isnot [bool] -or $BiasMetadata.k_rot) {
                    throw "KV bias metadata does not match the selected model, size, hashes, or unrotated basis. $Recalibrate"
                }
                if ($DryRun) {
                    Write-Host "  KV bias: metadata filename/size checked; SHA256 verification is deferred until launch."
                } else {
                    Write-Host "  Verifying KV bias and model SHA256; reading the model weights can take several seconds."
                    if ((Get-FileHash -LiteralPath $KvBias.FullName -Algorithm SHA256).Hash -ne $BiasMetadata.bias_sha256 -or
                        (Get-FileHash -LiteralPath $Model.FullName -Algorithm SHA256).Hash -ne $BiasMetadata.model_sha256) {
                        throw "KV bias or model SHA256 differs from the calibration metadata. $Recalibrate"
                    }
                }
            } elseif ($Profile -in @("long", "max") -and -not $DryRun) {
                throw "Profile '$Profile' requires model identity metadata '$MetadataPath'. $Recalibrate"
            } else {
                Write-Warning "KV bias has no identity metadata and its hashes have not been verified. $Recalibrate"
            }
            # The bias is calibrated with K-rotation off; inference must match
            # (the loader rejects a mismatch by design). The finally block below
            # restores the prior value (the shell launcher execs in a child
            # process, but PowerShell shares the parent env and would otherwise
            # leave this set for later launches).
            $env:LLAMA_ATTN_ROT_DISABLE = "1"
            $KvArgs += @("--kv-mean-center", $KvBias.FullName)
            Write-Host "  KV cache: q4_0 + mean-centering ($($KvBias.Name))" -ForegroundColor Green
        } else {
            Write-Host "  KV cache: q4_0 (no matching bias; run scripts/make_kv_bias.ps1 for better quality)" -ForegroundColor Yellow
        }
    }
    $ServerArgs = @(
        "-m", $Model.FullName,
        "--host", $HostAddress,
        "--port", "$Port",
        "-ngl", $Ngl, "-fa", "on",
        "-c", $Ctx
    ) + $SamplingArgs + @("--jinja")
    if ($Mmproj) {
        $ServerArgs += @("--mmproj", $Mmproj.FullName)
        # BONSAI_MMPROJ_CPU=1 keeps the vision projector in system RAM instead of
        # VRAM (frees ~0.9 GiB for KV/context; slower image prefill only).
        $MmprojCpu = if ($null -ne $env:BONSAI_MMPROJ_CPU) { $env:BONSAI_MMPROJ_CPU -in @("1", "true", "yes", "on") } else { $ProfileConfig -and $ProfileConfig.mmproj_cpu }
        if ($MmprojCpu) {
            $ServerArgs += "--no-mmproj-offload"
            Write-Host "  Vision:  projector on CPU/RAM (BONSAI_MMPROJ_CPU=1)" -ForegroundColor Green
        }
    }
    if ($SpecArgs.Count -gt 0) { $ServerArgs += $SpecArgs }
    if ($KvArgs.Count -gt 0) { $ServerArgs += $KvArgs }
    # Image-token cap: big images cost minutes of prefill on slower hardware.
    # Capped at 1024 unless running the CUDA/HIP build; override with
    # BONSAI_IMAGE_MAX_TOKENS (a number, or 0 to disable the cap).
    $ImageMaxTokens = if ($env:BONSAI_IMAGE_MAX_TOKENS) {
        if ($env:BONSAI_IMAGE_MAX_TOKENS -eq "0") { $null } else { $env:BONSAI_IMAGE_MAX_TOKENS }
    } elseif ($BinRel -like "bin\cuda\*" -or $BinRel -like "bin\hip\*") {
        $null
    } else {
        "1024"
    }
    if ($ImageMaxTokens) { $ServerArgs += @("--image-max-tokens", $ImageMaxTokens) }
    # Default MCP tool servers for the built-in web UI (user can still edit
    # them in Settings -> MCP Client).
    $WebuiConfig = Join-Path $PSScriptRoot "webui-config.json"
    if (Test-Path $WebuiConfig) { $ServerArgs += @("--webui-config-file", $WebuiConfig) }
}

if ($ProfileConfig) {
    $Parallel = if ($env:BONSAI_PARALLEL) { [int]$env:BONSAI_PARALLEL } else { [int]$ProfileConfig.parallel }
    if ($BonsaiModel -eq "27B" -and $SpecArgs.Count -gt 0) { $Parallel = 1 }
    if ($Parallel -lt 1) { throw "BONSAI_PARALLEL must be positive." }
    $ServerArgs += @("--parallel", "$Parallel", "--cache-ram", "$($ProfileConfig.cache_ram_mib)",
        "-b", "$($ProfileConfig.batch)", "-ub", "$($ProfileConfig.ubatch)",
        "-t", "$($ProfileConfig.threads)", "-tb", "$($ProfileConfig.threads_batch)", "--poll", "0")
    if ($ProfileConfig.kv_unified) { $ServerArgs += "--kv-unified" }
    $EffortJson = '{"reasoning_effort":"' + $ProfileConfig.reasoning_effort + '"}'
    if ($PSVersionTable.PSEdition -eq 'Desktop') { $EffortJson = $EffortJson.Replace('"', '\"') }
    if ($BonsaiModel -eq "27B") { $ServerArgs += @("--chat-template-kwargs", $EffortJson) }
} elseif ($env:BONSAI_PARALLEL) {
    $ServerArgs += @("--parallel", $env:BONSAI_PARALLEL)
}
if ($env:BONSAI_API_KEY) { $ServerArgs += @("--api-key", $env:BONSAI_API_KEY) }

$EffCtx = if ($BonsaiModel -eq "27B") { $Ctx } else { $CtxDefault }
Write-Host "  Context: -c $EffCtx (override with BONSAI_CTX, 0 = auto)"
    if ($DryRun) {
        $SafeArgs = @($ServerArgs + $ExtraArgs)
        for ($i = 0; $i -lt $SafeArgs.Count; $i++) {
            if ($SafeArgs[$i] -in @("--api-key", "--api-key-file")) {
                if ($i + 1 -lt $SafeArgs.Count) { $SafeArgs[$i + 1] = "<redacted>" }
            } elseif ($SafeArgs[$i] -match '^--api-key(?:-file)?=') { $SafeArgs[$i] = "--api-key=<redacted>" }
        }
        @{ binary=$Bin; profile=$Profile; arguments=$SafeArgs } | ConvertTo-Json -Depth 4
        $code = 0
    } else {
        $NativeExtraArgs = @($ExtraArgs)
        if ($PSVersionTable.PSEdition -eq 'Desktop') {
            # Windows PowerShell's legacy native binder removes JSON quotes.
            # Escape valid JSON at the final native boundary; already escaped
            # legacy values remain unchanged rather than being escaped twice.
            for ($i = 0; $i -lt $NativeExtraArgs.Count - 1; $i++) {
                if ($NativeExtraArgs[$i] -eq "--chat-template-kwargs") {
                    try {
                        $null = ConvertFrom-Json -InputObject $NativeExtraArgs[$i + 1] -ErrorAction Stop
                        $NativeExtraArgs[$i + 1] = [regex]::Replace($NativeExtraArgs[$i + 1], '(\\*)"', '${1}${1}\"')
                    } catch {}
                    $i++
                }
            }
        }
        & $Bin @ServerArgs @NativeExtraArgs
        $code = $LASTEXITCODE
    }
} finally {
    # Don't leak the KV4 bias flag into the parent PowerShell session.
    if ($null -eq $priorRotDisable) {
        Remove-Item Env:LLAMA_ATTN_ROT_DISABLE -ErrorAction SilentlyContinue
    } else {
        $env:LLAMA_ATTN_ROT_DISABLE = $priorRotDisable
    }
}
exit $code
