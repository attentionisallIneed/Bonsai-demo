# Build llama.cpp with CUDA on Windows
# Prerequisites: CUDA toolkit (nvcc), CMake, Visual Studio Build Tools (or full VS)
#
# Usage:
#   .\scripts\build_cuda_windows.ps1
#   .\scripts\build_cuda_windows.ps1 -CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
#   .\scripts\build_cuda_windows.ps1 -Archs "80;86;89;90"
param(
    [string]$CudaPath = "",
    [string]$Archs = "",
    [string]$RepoDir = ".\llama.cpp",
    [string]$OutputDir = "cuda"
)

$ErrorActionPreference = "Stop"
# Resolve relative paths before entering the source checkout.
$RepoDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RepoDir)

# ── Clone if needed ──
if (-not (Test-Path $RepoDir)) {
    Write-Host "llama.cpp not found at $RepoDir - cloning from PrismML-Eng ..."
    git clone -b prism https://github.com/PrismML-Eng/llama.cpp.git $RepoDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed (exit $LASTEXITCODE); no binaries were copied." }
}
if (-not (Test-Path -LiteralPath $RepoDir -PathType Container)) { throw "llama.cpp checkout not found: $RepoDir" }

# ── Find CUDA toolkit ──
if (-not $CudaPath) {
    $nvcc = Get-Command nvcc -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($nvcc) {
        $CudaPath = (Split-Path (Split-Path $nvcc.Source))
    } else {
        $defaultPaths = @(
            "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA"
        )
        foreach ($base in $defaultPaths) {
            if (Test-Path $base) {
                $latest = Get-ChildItem $base -Directory | Sort-Object Name -Descending | Select-Object -First 1
                if ($latest) {
                    $CudaPath = $latest.FullName
                    break
                }
            }
        }
    }
}

if (-not $CudaPath -or -not (Test-Path $CudaPath)) {
    Write-Host "[ERR] CUDA toolkit not found." -ForegroundColor Red
    Write-Host "      Install from https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    Write-Host "      Or specify: .\scripts\build_cuda_windows.ps1 -CudaPath 'C:\path\to\cuda'" -ForegroundColor Yellow
    throw "CUDA toolkit not found."
}

$Nvcc = Join-Path $CudaPath "bin\nvcc.exe"
if (-not (Test-Path $Nvcc)) {
    Write-Host "[ERR] nvcc not found at $Nvcc" -ForegroundColor Red
    throw "nvcc not found at $Nvcc"
}

# ── Detect CUDA version ──
$nvccOutput = & $Nvcc --version 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { throw "nvcc --version failed (exit $LASTEXITCODE); no binaries were copied." }
if ($nvccOutput -match 'release (\d+)\.(\d+)') {
    $CudaMajor = [int]$Matches[1]
    $CudaMinor = [int]$Matches[2]
    $CudaVersion = "$CudaMajor.$CudaMinor"
} else {
    Write-Host "[ERR] Could not parse CUDA version from nvcc" -ForegroundColor Red
    throw "Could not parse CUDA version from nvcc."
}

# ── Default architectures ──
if (-not $Archs) {
    if ($CudaMajor -ge 13) {
        $Archs = "80;86;89;90;100;120"
    } elseif ($CudaMajor -eq 12 -and $CudaMinor -ge 8) {
        $Archs = "80;86;89;90;120"
    } else {
        # SM120 first became available in CUDA 12.8. In particular the 12.4
        # and 12.6 toolkits cannot compile a fat binary containing that target.
        # https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/
        $Archs = "80;86;89;90"
    }
}
if (($CudaMajor -lt 12 -or ($CudaMajor -eq 12 -and $CudaMinor -lt 8)) -and
    @($Archs -split ';' | Where-Object { $_ -match '^(100|101|120)([af])?(-real|-virtual)?$' }).Count -gt 0) {
    throw "Blackwell architectures (100/101/120) require CUDA 12.8 or newer; this toolkit is $CudaVersion."
}

$Dest = Join-Path "bin" $OutputDir
$BuildDir = "build-cuda"

Write-Host ""
Write-Host "=== Building llama.cpp with CUDA $CudaVersion (Windows) ===" -ForegroundColor Cyan
Write-Host "  Repo:   $RepoDir"
Write-Host "  CUDA:   $CudaPath (v$CudaVersion)"
Write-Host "  Archs:  $Archs"
Write-Host "  Output: $Dest"
Write-Host ""

# ── Ensure cmake is available ──
$Cmake = Get-Command cmake -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $Cmake) {
    Write-Host "[ERR] CMake not found. Install from https://cmake.org/download/" -ForegroundColor Red
    Write-Host "      Or if you ran setup.ps1, activate the venv first:" -ForegroundColor Yellow
    Write-Host "      & .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    throw "CMake not found."
}

# ── Configure ──
$priorCudacxx = $env:CUDACXX
$pushedLocation = $false
try {
    $env:CUDACXX = $Nvcc
    Push-Location -LiteralPath $RepoDir
    $pushedLocation = $true

    & $Cmake.Source -B $BuildDir `
        -DGGML_CUDA=ON `
        -DCMAKE_CUDA_COMPILER="$Nvcc" `
        -DCMAKE_CUDA_ARCHITECTURES="$Archs" `
        -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed (exit $LASTEXITCODE); no binaries were copied." }

# ── Build ──
# Limit parallelism to avoid OOM during CUDA compilation.
    try {
        $sysInfo = Get-CimInstance Win32_ComputerSystem -ErrorAction Stop
        $memGB = [math]::Floor($sysInfo.TotalPhysicalMemory / 1GB)
        if ($memGB -lt 16) {
            $buildJobs = 2
            Write-Host "  Low RAM (~${memGB} GB) -- using -j $buildJobs" -ForegroundColor Yellow
        } else {
            $buildJobs = [math]::Min(16, $sysInfo.NumberOfLogicalProcessors)
            Write-Host "  Using -j $buildJobs"
        }
    } catch {
        $buildJobs = 4
        Write-Host "  Could not detect RAM -- using -j $buildJobs" -ForegroundColor Yellow
    }

    & $Cmake.Source --build $BuildDir --config Release -j $buildJobs
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed (exit $LASTEXITCODE); no binaries were copied." }
} finally {
    if ($pushedLocation) { Pop-Location }
    $env:CUDACXX = $priorCudacxx
}

# ── Copy binaries ──
Write-Host ""
Write-Host "=== Copying binaries to $Dest ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $Dest -Force | Out-Null

# Ship every llama-*.exe that was built (cli, server, quantize, bench, ...) so
# bin/ is entirely source-built and consistent -- no mixing with prebuilt binaries.
$binDirs = @((Join-Path $RepoDir "$BuildDir\bin\Release"), (Join-Path $RepoDir "$BuildDir\bin"))
$copied = @{}
foreach ($binDir in $binDirs) {
    if (-not (Test-Path $binDir)) { continue }
    foreach ($exe in Get-ChildItem -Path $binDir -Filter "llama-*.exe" -ErrorAction SilentlyContinue) {
        if ($copied.ContainsKey($exe.Name)) { continue }
        Copy-Item $exe.FullName -Destination $Dest -Force
        $copied[$exe.Name] = $true
        Write-Host "  Copied $($exe.Name)"
    }
}
if ($copied.Count -eq 0) { throw "CMake build produced no llama-*.exe binaries." }

# ── Copy DLLs ──
Write-Host ""
Write-Host "=== Copying shared libraries ===" -ForegroundColor Cyan
$dllPatterns = @("llama.dll", "ggml*.dll", "llama-*.dll", "mtmd.dll")
foreach ($pattern in $dllPatterns) {
    $releasePath = Join-Path $RepoDir "$BuildDir\bin\Release"
    $binPath = Join-Path $RepoDir "$BuildDir\bin"
    foreach ($searchPath in @($releasePath, $binPath)) {
        $dlls = Get-ChildItem -Path $searchPath -Filter $pattern -ErrorAction SilentlyContinue
        foreach ($dll in $dlls) {
            Copy-Item $dll.FullName -Destination $Dest -Force
            Write-Host "  Copied $($dll.Name)"
        }
    }
}

# Copy CUDA runtime DLLs from toolkit
$cudaBinDir = Join-Path $CudaPath "bin"
foreach ($cudaDll in @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")) {
    $found = Get-ChildItem -Path $cudaBinDir -Filter $cudaDll -ErrorAction SilentlyContinue
    foreach ($f in $found) {
        Copy-Item $f.FullName -Destination $Dest -Force
        Write-Host "  Copied $($f.Name) (CUDA runtime)"
    }
}

Write-Host ""
Write-Host "Done! CUDA $CudaVersion Windows binaries are in: $Dest" -ForegroundColor Green
Write-Host ""
Write-Host "Run:" -ForegroundColor Cyan
Write-Host "  `$model = (Get-ChildItem models\gguf\*.gguf | Select-Object -First 1).FullName"
Write-Host "  $Dest\llama-cli.exe -m `$model -ngl 99 -c 4096 -p 'Hello'"
