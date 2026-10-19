# Model selection and atomic bias generation. No work is done merely by loading this file.

function Resolve-BonsaiCalibrationModel([string]$RepositoryRoot, [string]$ModelPath, [string]$BinaryPath) {
    if (-not $ModelPath) { $ModelPath = $env:BONSAI_GGUF }
    if ($ModelPath) {
        if (-not [IO.Path]::IsPathRooted($ModelPath)) { $ModelPath = Join-Path $RepositoryRoot $ModelPath }
        if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) { throw "Model not found: $ModelPath" }
        return Get-Item -LiteralPath $ModelPath
    }
    $family = if ($env:BONSAI_FAMILY) { $env:BONSAI_FAMILY.ToLowerInvariant() } else { 'bonsai2' }
    $size = if ($env:BONSAI_MODEL) { $env:BONSAI_MODEL.ToUpperInvariant() } else { '27B' }
    if ($family -notin @('bonsai2', 'ternary', 'bonsai') -or $size -notin @('27B', '8B', '4B', '1.7B')) {
        throw "Choose a concrete BONSAI_FAMILY (bonsai2/ternary/bonsai) and BONSAI_MODEL (27B/8B/4B/1.7B)."
    }
    if ($family -eq 'bonsai2' -and $size -ne '27B') { throw "Bonsai 2 is 27B." }
    $directory = switch ($family) { 'bonsai2' { 'bonsai2-gguf' }; 'ternary' { 'ternary-gguf' }; 'bonsai' { 'gguf' } }
    $modelDirectory = Join-Path $RepositoryRoot "models\$directory\$size"
    $patterns = switch ($family) {
        'bonsai2' { @('*-PQ2_0.gguf', '*-PTQ1_0.gguf') }
        'bonsai' { @('*-Q1_0.gguf') }
        'ternary' {
            if ($env:BONSAI_FORCE_G64 -eq '1' -or $BinaryPath -match '[\\/]bin[\\/](vulkan|sycl)[\\/]') {
                @('*g64.gguf', '*-Q2_0.gguf')
            } else { @('*-PQ2_0.gguf', '*g64.gguf', '*-Q2_0.gguf') }
        }
    }
    foreach ($pattern in $patterns) {
        if ($pattern -eq '*-Q2_0.gguf' -and -not (Test-Path -LiteralPath (Join-Path $modelDirectory '.official-q2_0'))) { continue }
        $model = Get-ChildItem -LiteralPath $modelDirectory -Filter $pattern -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notmatch 'mmproj|dspark|kv-bias' } | Sort-Object Name | Select-Object -First 1
        if ($model) { return $model }
    }
    throw "No supported model weights in $modelDirectory. Run setup.ps1 or specify -ModelPath."
}

function Read-BonsaiGgufString([IO.BinaryReader]$Reader) {
    $length = $Reader.ReadUInt64()
    if ($length -gt 1048576) { throw "Invalid GGUF string length." }
    $data = $Reader.ReadBytes([int]$length)
    if ($data.Length -ne $length) { throw "Truncated GGUF string." }
    return [Text.Encoding]::UTF8.GetString($data)
}

function Read-BonsaiGgufValue([IO.BinaryReader]$Reader, [uint32]$Type, [int]$Depth = 0) {
    if ($Depth -gt 4) { throw "Invalid nested GGUF metadata." }
    switch ($Type) {
        0 { return $Reader.ReadByte() }; 1 { return $Reader.ReadSByte() }
        2 { return $Reader.ReadUInt16() }; 3 { return $Reader.ReadInt16() }
        4 { return $Reader.ReadUInt32() }; 5 { return $Reader.ReadInt32() }
        6 { return $Reader.ReadSingle() }; 7 { return $Reader.ReadBoolean() }
        8 { return Read-BonsaiGgufString $Reader }
        9 {
            $elementType = $Reader.ReadUInt32(); $count = $Reader.ReadUInt64()
            if ($count -gt 1048576) { throw "Invalid GGUF metadata array length." }
            for ($i = 0; $i -lt $count; $i++) { $null = Read-BonsaiGgufValue $Reader $elementType ($Depth + 1) }
            return $null
        }
        10 { return $Reader.ReadUInt64() }; 11 { return $Reader.ReadInt64() }; 12 { return $Reader.ReadDouble() }
        default { throw "Unknown GGUF metadata value type $Type" }
    }
}

function Test-BonsaiBiasFile([string]$Path) {
    # Check the metadata basis and tensor data bounds before publishing a bias.
    # This also rejects a successful tool exit that left a truncated output file.
    $reader = [IO.BinaryReader]::new([IO.File]::OpenRead($Path))
    try {
        if ($reader.ReadUInt32() -ne 0x46554747 -or $reader.ReadUInt32() -notin @(2, 3)) { throw "Invalid bias GGUF header." }
        $tensorCount = $reader.ReadUInt64(); $metadataCount = $reader.ReadUInt64()
        if ($tensorCount -eq 0 -or $tensorCount -gt 1024 -or $metadataCount -gt 4096) { throw "Invalid bias GGUF counts." }
        $metadata = @{}
        for ($i = 0; $i -lt $metadataCount; $i++) {
            $key = Read-BonsaiGgufString $reader
            $metadata[$key] = Read-BonsaiGgufValue $reader $reader.ReadUInt32()
        }
        if ($metadata['general.type'] -ne 'kv-mean-center' -or
            -not $metadata.ContainsKey('kv_mean_center.k_rot') -or $metadata['kv_mean_center.k_rot'] -ne $false) {
            throw "Bias GGUF must identify an unrotated kv-mean-center calibration."
        }
        [uint64]$dataEnd = 0
        for ($i = 0; $i -lt $tensorCount; $i++) {
            $name = Read-BonsaiGgufString $reader
            $dimensions = $reader.ReadUInt32()
            if ($name -notmatch '^kv_bar\.blk\.\d+\.k$' -or $dimensions -lt 1 -or $dimensions -gt 4) { throw "Invalid bias tensor." }
            [uint64]$elements = 1
            for ($d = 0; $d -lt $dimensions; $d++) {
                $dimension = $reader.ReadUInt64()
                if ($dimension -eq 0 -or $dimension -gt 1048576) { throw "Invalid bias tensor shape." }
                $elements *= $dimension
                if ($elements -gt 16777216) { throw "Bias tensor is unexpectedly large." }
            }
            if ($reader.ReadUInt32() -ne 0) { throw "Bias tensors must use F32 storage." }
            $offset = $reader.ReadUInt64()
            if ($offset -gt 67108864) { throw "Invalid bias tensor offset." }
            $dataEnd = [Math]::Max($dataEnd, $offset + $elements * 4)
        }
        [uint64]$alignment = if ($metadata.ContainsKey('general.alignment')) { $metadata['general.alignment'] } else { 32 }
        if ($alignment -lt 1 -or $alignment -gt 4096) { throw "Invalid bias GGUF alignment." }
        $dataStart = [uint64]([Math]::Ceiling($reader.BaseStream.Position / [double]$alignment) * $alignment)
        if ($reader.BaseStream.Length -lt ($dataStart + $dataEnd)) { throw "Truncated bias GGUF tensor data." }
    } finally { $reader.Dispose() }
}

function Invoke-BonsaiKvCalibration {
    [CmdletBinding()]
    param([string]$RepositoryRoot, [string]$CorpusPath, [string]$ModelPath,
        [string]$BinaryPath, [string]$OutputPath, [int]$ContextSize = 512,
        [int]$Chunks = -1, [int]$GpuLayers = 99, [switch]$DryRun)
    if ($Chunks -eq 0 -or $Chunks -lt -1) { throw "Chunks must be -1 (all) or a positive number." }
    if (-not $BinaryPath) {
        foreach ($relative in @('bin\cuda', 'bin\hip', 'bin\vulkan', 'bin\cpu', 'llama.cpp\build\bin\Release', 'llama.cpp\build\bin')) {
            $candidate = Join-Path $RepositoryRoot "$relative\llama-kv-mean-center.exe"
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { $BinaryPath = $candidate; break }
        }
    }
    if (-not $BinaryPath -or -not (Test-Path -LiteralPath $BinaryPath -PathType Leaf)) {
        throw "llama-kv-mean-center.exe not found. Run scripts/download_cuda_bin.ps1 or setup.ps1 first."
    }
    $BinaryPath = (Get-Item -LiteralPath $BinaryPath).FullName
    $model = Resolve-BonsaiCalibrationModel $RepositoryRoot $ModelPath $BinaryPath
    if (-not $CorpusPath) { $CorpusPath = Join-Path $RepositoryRoot 'scripts\kv-calibration-corpus.txt' }
    if (-not (Test-Path -LiteralPath $CorpusPath -PathType Leaf)) { throw "Calibration corpus not found: $CorpusPath" }
    $corpus = Get-Item -LiteralPath $CorpusPath
    if ($corpus.Length -eq 0) { throw "Calibration corpus is empty." }
    if (-not $OutputPath) { $OutputPath = Join-Path $model.DirectoryName "$($model.BaseName)-kv-bias.gguf" }
    $OutputPath = [IO.Path]::GetFullPath($OutputPath)
    if ([IO.Path]::GetExtension($OutputPath) -ne '.gguf' -or
        $OutputPath -eq $model.FullName -or $OutputPath -eq $corpus.FullName) {
        throw "OutputPath must be a separate .gguf bias file."
    }
    $arguments = @('-m', $model.FullName, '-f', $corpus.FullName, '-ngl', "$GpuLayers",
        '-c', "$ContextSize", '--chunks', "$Chunks", '--cache-type-k', 'q4_0', '--cache-type-v', 'q4_0', '-fa', 'on')
    if ($DryRun) {
        [pscustomobject]@{ Model = $model.FullName; Binary = $BinaryPath; Corpus = $corpus.FullName
            Output = $OutputPath; RotationDisable = '1'; Arguments = $arguments + @('-o', $OutputPath) }
        return
    }
    if ($BinaryPath -match '[\\/]bin[\\/]cuda[\\/]') {
        $stampPath = Join-Path (Split-Path $BinaryPath -Parent) '.llama_release'
        $stamp = if (Test-Path -LiteralPath $stampPath) { (Get-Content -LiteralPath $stampPath -Raw).Trim() } else { '' }
        if ($stamp -notmatch '^prism-b(\d+)-' -or [int]$Matches[1] -lt 10777) {
            throw "Update the CUDA runtime with scripts/download_cuda_bin.ps1 before Q4 calibration (requires Prism b10777 or newer)."
        }
    }
    $parent = Split-Path $OutputPath -Parent
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) { throw "Output directory does not exist: $parent" }
    $temporary = Join-Path $parent ('.kv-calibration-' + [guid]::NewGuid().ToString('N') + '.tmp')
    $temporaryMetadata = "$temporary.json"
    $priorRotation = $env:LLAMA_ATTN_ROT_DISABLE; $priorPath = $env:Path
    try {
        $modelHash = (Get-FileHash -LiteralPath $model.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $modelLength = $model.Length; $modelWriteTime = $model.LastWriteTimeUtc
        $env:LLAMA_ATTN_ROT_DISABLE = '1'
        $env:Path = "$(Split-Path $BinaryPath -Parent);$env:Path"
        Write-Host "Calibrating $($model.Name), Q4 K/V, K rotation disabled ..." -ForegroundColor Cyan
        $global:LASTEXITCODE = 0
        & $BinaryPath @arguments '-o' $temporary | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "llama-kv-mean-center failed (exit $LASTEXITCODE). Existing bias was preserved." }
        Test-BonsaiBiasFile $temporary
        $after = Get-Item -LiteralPath $model.FullName
        if ($after.Length -ne $modelLength -or $after.LastWriteTimeUtc -ne $modelWriteTime) {
            throw "Model changed during calibration; refusing to publish the bias."
        }
        $record = [ordered]@{ schema_version = 1; model_file = $model.Name; model_size = $modelLength
            model_sha256 = $modelHash; model_last_write_utc = $modelWriteTime.ToString('o')
            bias_sha256 = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant()
            k_rot = $false; rotation_disable = '1'; cache_type_k = 'q4_0'; cache_type_v = 'q4_0'
            context_size = $ContextSize; chunks = $Chunks; corpus_sha256 = (Get-FileHash -LiteralPath $corpus.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            created_utc = [DateTime]::UtcNow.ToString('o') }
        [IO.File]::WriteAllText($temporaryMetadata, ($record | ConvertTo-Json -Depth 3), [Text.UTF8Encoding]::new($false))
        if (Test-Path -LiteralPath $OutputPath) { [IO.File]::Replace($temporary, $OutputPath, $null) }
        else { [IO.File]::Move($temporary, $OutputPath) }
        $metadataPath = "$OutputPath.json"
        if (Test-Path -LiteralPath $metadataPath) { [IO.File]::Replace($temporaryMetadata, $metadataPath, $null) }
        else { [IO.File]::Move($temporaryMetadata, $metadataPath) }
        Write-Host "Bias written: $OutputPath" -ForegroundColor Green
        Write-Host 'Use: $env:BONSAI_KV4="1"; .\scripts\start_llama_server.ps1'
    } finally {
        $env:LLAMA_ATTN_ROT_DISABLE = $priorRotation; $env:Path = $priorPath
        foreach ($path in @($temporary, $temporaryMetadata)) {
            if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
        }
    }
}
