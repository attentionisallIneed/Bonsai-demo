# Shared by setup.ps1's CUDA path and the standalone runtime updater.
# Loading this file defines functions only; it does not download or install anything.

function Get-BonsaiReleaseTag([string]$RepositoryRoot) {
    $tag = (Get-Content -LiteralPath (Join-Path $RepositoryRoot "scripts\llama-release.txt") -Raw).Trim()
    if ($tag -notmatch '^prism-b\d+-[a-f0-9]+$') { throw "Invalid scripts/llama-release.txt" }
    return $tag
}

function Assert-BonsaiRuntimeStopped([string]$RepositoryRoot, [string]$Destination) {
    $prefixes = @($RepositoryRoot, $Destination) | ForEach-Object {
        [IO.Path]::GetFullPath($_).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    }
    # Process paths, rather than a port probe, also catch CLI/calibration processes.
    foreach ($process in @(Get-Process -Name "llama-*" -ErrorAction SilentlyContinue)) {
        try { $processPath = $process.Path } catch { $processPath = $null }
        if (-not $processPath) {
            throw "Cannot inspect llama process PID $($process.Id). Stop it before updating the runtime."
        }
        foreach ($prefix in $prefixes) {
            if ($processPath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Runtime is in use by $($process.Name) (PID $($process.Id)). Stop it before updating."
            }
        }
    }
}

function Get-BonsaiReleaseAssets([string]$ReleaseTag, [string[]]$Names) {
    $headers = @{ "User-Agent" = "Bonsai-demo-runtime-updater"; "Accept" = "application/vnd.github+json" }
    # Use an existing credential in memory only; never log or persist it.
    $token = if ($env:GH_TOKEN) { $env:GH_TOKEN } elseif ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } else { $null }
    if ($token) { $headers.Authorization = "Bearer $token" }
    $endpoint = "repos/PrismML-Eng/llama.cpp/releases/tags/$ReleaseTag"
    try {
        $release = Invoke-RestMethod -Uri "https://api.github.com/$endpoint" -Headers $headers
    } catch {
        $status = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
        $gh = if ($status -in @(403, 429)) { Get-Command gh -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1 } else { $null }
        if (-not $gh) { throw }
        Write-Host 'GitHub API is rate limited; retrying with the existing gh session ...' -ForegroundColor Yellow
        $json = & $gh.Source api $endpoint 2>$null | Out-String
        if ($LASTEXITCODE -ne 0) { throw "GitHub release metadata fallback failed (gh exit $LASTEXITCODE). Existing runtime was preserved." }
        try { $release = $json | ConvertFrom-Json } catch { throw "gh returned invalid release metadata. Existing runtime was preserved." }
    } finally {
        $token = $null
        $headers.Remove('Authorization')
    }
    if ($release.tag_name -ne $ReleaseTag) { throw "GitHub returned an unexpected release tag." }
    foreach ($name in $Names) {
        $asset = @($release.assets | Where-Object { $_.name -eq $name })
        if ($asset.Count -ne 1) { throw "Release $ReleaseTag does not contain exactly one $name" }
        if ($asset[0].size -le 0) { throw "Release asset $name is empty." }
        # GitHub publishes SHA256 digests for these assets. Fail closed when an API
        # response omits one instead of silently installing an unverified archive.
        if ($asset[0].digest -notmatch '^sha256:[a-fA-F0-9]{64}$') {
            throw "No SHA256 digest published for $name. Existing runtime was preserved."
        }
        $expectedUrl = "https://github.com/PrismML-Eng/llama.cpp/releases/download/$ReleaseTag/$name"
        if ($asset[0].browser_download_url -ne $expectedUrl) { throw "Unexpected download URL for $name" }
        $asset[0]
    }
}

function Test-BonsaiAsset([string]$Path, $Asset) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -ne $Asset.size) { return $false }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Asset.digest.Substring(7)
}

function Receive-BonsaiDownload([string]$Uri, [string]$Path) {
    $curl = Get-Command curl.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($curl) {
        & $curl.Source --fail --location --retry 3 --retry-all-errors --retry-delay 2 --connect-timeout 20 --speed-limit 1024 --speed-time 30 --continue-at - --output $Path $Uri
        if ($LASTEXITCODE -eq 33) {
            # Some proxies do not support Range. Retry from byte zero in that case.
            & $curl.Source --fail --location --retry 3 --retry-all-errors --retry-delay 2 --connect-timeout 20 --speed-limit 1024 --speed-time 30 --output $Path $Uri
        }
        if ($LASTEXITCODE -ne 0) { throw "Download failed ($LASTEXITCODE); partial data is retained for retry: $Path" }
        return
    }
    # Windows PowerShell 5.1 fallback, including HTTP Range resume without curl.
    $offset = if (Test-Path -LiteralPath $Path) { (Get-Item -LiteralPath $Path).Length } else { 0 }
    $request = [Net.HttpWebRequest]::Create($Uri)
    $request.UserAgent = "Bonsai-demo-runtime-updater"
    if ($offset -gt 0) { $request.AddRange($offset) }
    $response = $null; $inputStream = $null; $outputStream = $null
    try {
        $response = $request.GetResponse()
        $append = $offset -gt 0 -and [int]$response.StatusCode -eq 206
        if ($append -and $response.Headers['Content-Range'] -notmatch "^bytes $offset-") {
            throw "Server returned an incorrect resumed byte range."
        }
        $mode = if ($append) { [IO.FileMode]::Append } else { [IO.FileMode]::Create }
        $outputStream = [IO.File]::Open($Path, $mode, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $inputStream = $response.GetResponseStream()
        $inputStream.CopyTo($outputStream)
    } finally {
        if ($outputStream) { $outputStream.Dispose() }
        if ($inputStream) { $inputStream.Dispose() }
        if ($response) { $response.Dispose() }
    }
}

function Save-BonsaiAsset($Asset, [string]$CacheDirectory) {
    $archive = Join-Path $CacheDirectory $Asset.name
    if (Test-BonsaiAsset $archive $Asset) { return $archive }
    $partial = "$archive.part"
    if ((Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial).Length -ge $Asset.size) {
        if (Test-BonsaiAsset $partial $Asset) {
            Move-Item -LiteralPath $partial -Destination $archive -Force
            return $archive
        }
        Remove-Item -LiteralPath $partial -Force
    }
    Write-Host "Downloading $($Asset.name) (resumable) ..." -ForegroundColor Cyan
    Receive-BonsaiDownload $Asset.browser_download_url $partial
    if (-not (Test-BonsaiAsset $partial $Asset)) {
        Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
        throw "Size or SHA256 mismatch for $($Asset.name). Existing runtime was preserved."
    }
    Move-Item -LiteralPath $partial -Destination $archive -Force
    return $archive
}

function Expand-BonsaiRuntimeArchive([string]$Archive, [string]$Destination) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $prefix = [IO.Path]::GetFullPath($Destination).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $zip = [IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        foreach ($entry in $zip.Entries) {
            $target = [IO.Path]::GetFullPath((Join-Path $Destination $entry.FullName))
            if (-not $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Unsafe archive entry: $($entry.FullName)"
            }
        }
    } finally { $zip.Dispose() }
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
}

function Test-BonsaiCudaFiles([string]$Directory) {
    foreach ($file in @('llama-server.exe', 'llama-cli.exe', 'llama-kv-mean-center.exe', 'ggml-cuda.dll')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Directory $file) -PathType Leaf)) { return $false }
    }
    return @((Get-ChildItem -LiteralPath $Directory -Filter 'cudart64_*.dll' -File)).Count -gt 0 -and
        @((Get-ChildItem -LiteralPath $Directory -Filter 'cublas64_*.dll' -File)).Count -gt 0 -and
        @((Get-ChildItem -LiteralPath $Directory -Filter 'cublasLt64_*.dll' -File)).Count -gt 0
}

function Test-BonsaiRuntimeSmoke([string]$Directory, [string]$ReleaseTag) {
    $priorPath = $env:Path
    try {
        $env:Path = "$Directory;$env:Path"
        $output = & (Join-Path $Directory 'llama-server.exe') --version 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0 -or $output -notmatch [regex]::Escape(($ReleaseTag -split '-')[-1])) {
            throw "Staged llama-server failed --version or did not report the pinned commit: $output"
        }
    } finally { $env:Path = $priorPath }
}

function Remove-BonsaiTemporaryDirectory([string]$Path, [string]$Parent, [string]$LeafPrefix) {
    # Only generated sibling staging/backup directories may be recursively removed.
    $resolved = [IO.Path]::GetFullPath($Path)
    $expectedParent = [IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    if ((Split-Path $resolved -Parent) -ne $expectedParent -or
        -not (Split-Path $resolved -Leaf).StartsWith($LeafPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a directory outside the runtime staging area: $resolved"
    }
    if (Test-Path -LiteralPath $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}

function Invoke-BonsaiCudaInstall {
    [CmdletBinding()]
    param([string]$RepositoryRoot, [ValidateSet('12.4', '13.3')][string]$CudaTag,
        [switch]$Force, [switch]$DryRun)
    $RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
    $destination = Join-Path $RepositoryRoot 'bin\cuda'
    $tag = Get-BonsaiReleaseTag $RepositoryRoot
    $stamp = "$tag cuda-$CudaTag"
    $stampPath = Join-Path $destination '.llama_release'
    if (-not $Force -and (Test-Path -LiteralPath $stampPath) -and
        (Get-Content -LiteralPath $stampPath -Raw).Trim() -eq $stamp -and (Test-BonsaiCudaFiles $destination)) {
        Write-Host "CUDA runtime already installed ($stamp)." -ForegroundColor Green
        return
    }
    $names = @("llama-$tag-bin-win-cuda-$CudaTag-x64.zip", "cudart-llama-bin-win-cuda-$CudaTag-x64.zip")
    if ($DryRun) {
        [pscustomobject]@{ ReleaseTag = $tag; CudaTag = $CudaTag; Destination = $destination; Assets = $names }
        return
    }
    Assert-BonsaiRuntimeStopped $RepositoryRoot $destination
    $assets = @(Get-BonsaiReleaseAssets $tag $names)
    $cache = Join-Path $RepositoryRoot '.cache\llama-downloads'
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    $archives = @($assets | ForEach-Object { Save-BonsaiAsset $_ $cache })
    $parent = Split-Path $destination -Parent
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $suffix = [guid]::NewGuid().ToString('N')
    $staging = Join-Path $parent "cuda.staging-$suffix"
    $backup = Join-Path $parent "cuda.backup-$suffix"
    # Siblings guarantee directory moves stay on the destination volume.
    New-Item -ItemType Directory -Path $staging | Out-Null
    try {
        foreach ($archive in $archives) { Expand-BonsaiRuntimeArchive $archive $staging }
        if (-not (Test-BonsaiCudaFiles $staging)) { throw "Staged CUDA runtime is incomplete. Existing runtime was preserved." }
        Test-BonsaiRuntimeSmoke $staging $tag
        [IO.File]::WriteAllText((Join-Path $staging '.llama_release'), $stamp, [Text.UTF8Encoding]::new($false))
        $record = [ordered]@{ release_tag = $tag; cuda_tag = $CudaTag; installed_utc = [DateTime]::UtcNow.ToString('o')
            assets = @($assets | ForEach-Object { @{ name = $_.name; size = $_.size; digest = $_.digest } }) }
        [IO.File]::WriteAllText((Join-Path $staging '.llama_release.json'), ($record | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
        Assert-BonsaiRuntimeStopped $RepositoryRoot $destination
        if (Test-Path -LiteralPath $destination) { Move-Item -LiteralPath $destination -Destination $backup }
        try { Move-Item -LiteralPath $staging -Destination $destination } catch {
            if (Test-Path -LiteralPath $backup) { Move-Item -LiteralPath $backup -Destination $destination }
            throw
        }
        Write-Host "CUDA runtime installed: $stamp" -ForegroundColor Green
        # A cleanup failure must not roll back the verified new install.
        try { Remove-BonsaiTemporaryDirectory $backup $parent 'cuda.backup-' } catch {
            Write-Warning "The old runtime backup could not be removed: $backup"
        }
    } finally { Remove-BonsaiTemporaryDirectory $staging $parent 'cuda.staging-' }
}
