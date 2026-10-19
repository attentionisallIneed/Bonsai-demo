# Offline regression tests for atomic runtime upgrades and KV calibration.
# Run with Windows PowerShell 5.1 or pwsh; no Pester, network, model or GPU needed.
$ErrorActionPreference = 'Stop'
$repository = Split-Path $PSScriptRoot -Parent
. (Join-Path $repository 'scripts\runtime_helpers.ps1')
. (Join-Path $repository 'scripts\kv_bias_helpers.ps1')
$productionDownloader = ${function:Receive-BonsaiDownload}

function Assert-True($Condition, [string]$Message) {
    if (-not $Condition) { throw "Assertion failed: $Message" }
}
function Assert-Throws([scriptblock]$Action, [string]$Pattern) {
    $message = $null
    try { & $Action } catch { $message = $_.Exception.Message }
    Assert-True ($message -and $message -match $Pattern) "Expected '$Pattern'; got '$message'"
}
function Write-GgufString([IO.BinaryWriter]$Writer, [string]$Value) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
    $Writer.Write([uint64]$bytes.Length); $Writer.Write($bytes)
}
function Write-FakeBias([string]$Path, [bool]$Rotated) {
    $writer = [IO.BinaryWriter]::new([IO.File]::Create($Path))
    try {
        $writer.Write([uint32]0x46554747); $writer.Write([uint32]3)
        $writer.Write([uint64]1); $writer.Write([uint64]2)
        Write-GgufString $writer 'general.type'; $writer.Write([uint32]8)
        Write-GgufString $writer 'kv-mean-center'
        Write-GgufString $writer 'kv_mean_center.k_rot'; $writer.Write([uint32]7); $writer.Write($Rotated)
        Write-GgufString $writer 'kv_bar.blk.0.k'; $writer.Write([uint32]2)
        $writer.Write([uint64]4); $writer.Write([uint64]2); $writer.Write([uint32]0); $writer.Write([uint64]0)
        while ($writer.BaseStream.Position % 32) { $writer.Write([byte]0) }
        for ($i = 0; $i -lt 8; $i++) { $writer.Write([single]0.25) }
    } finally { $writer.Dispose() }
}

$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('bonsai-runtime-test-' + [guid]::NewGuid().ToString('N'))
$savedEnvironment = @{}
foreach ($name in @('BONSAI_GGUF', 'BONSAI_FAMILY', 'BONSAI_MODEL', 'BONSAI_FORCE_G64', 'LLAMA_ATTN_ROT_DISABLE',
    'FAKE_BIAS_FIXTURE', 'FAKE_KV_MODE', 'FAKE_KV_LOG', 'FAKE_GITHUB_METADATA', 'GH_TOKEN', 'GITHUB_TOKEN',
    'FAKE_CURL_LOG', 'FAKE_CURL_FAIL_RANGE')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name)
}
try {
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    foreach ($path in @('scripts', 'bin\cuda', 'fixture\binary', 'fixture\cudart', 'models\bonsai2-gguf\27B')) {
        New-Item -ItemType Directory -Path (Join-Path $temporaryRoot $path) -Force | Out-Null
    }
    $tag = 'prism-b10777-e3de715'
    Set-Content -LiteralPath (Join-Path $temporaryRoot 'scripts\llama-release.txt') -Value $tag
    $destination = Join-Path $temporaryRoot 'bin\cuda'
    $sentinel = Join-Path $destination 'old-working-runtime'
    Set-Content -LiteralPath $sentinel -Value 'preserve-me'
    Set-Content -LiteralPath (Join-Path $destination '.llama_release') -Value 'old-version'
    $fixture = Join-Path $temporaryRoot 'fixture'
    foreach ($name in @('llama-server.exe', 'llama-cli.exe', 'llama-kv-mean-center.exe', 'ggml-cuda.dll')) {
        Set-Content -LiteralPath (Join-Path $fixture "binary\$name") -Value 'binary fixture'
    }
    foreach ($name in @('cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll')) {
        Set-Content -LiteralPath (Join-Path $fixture "cudart\$name") -Value 'DLL fixture'
    }
    $script:metadata = [pscustomobject]@{ tag_name = $tag; assets = @() }
    $assetNames = @("llama-$tag-bin-win-cuda-13.3-x64.zip", 'cudart-llama-bin-win-cuda-13.3-x64.zip')
    for ($i = 0; $i -lt 2; $i++) {
        $folder = if ($i -eq 0) { 'binary' } else { 'cudart' }
        $zipPath = Join-Path $fixture $assetNames[$i]
        Compress-Archive -Path (Join-Path $fixture "$folder\*") -DestinationPath $zipPath
        $script:metadata.assets += [pscustomobject]@{ name = $assetNames[$i]; size = (Get-Item -LiteralPath $zipPath).Length
            digest = 'sha256:' + (Get-FileHash -LiteralPath $zipPath).Hash.ToLowerInvariant()
            browser_download_url = "https://github.com/PrismML-Eng/llama.cpp/releases/download/$tag/$($assetNames[$i])" }
    }
    $script:apiCalls = 0; $script:downloadCalls = 0; $script:corruptDownload = $false
    $script:smokeFails = $false; $script:activateAfterSmoke = $false; $script:activeProcess = $false
    # Replace network and native execution only; exercise the real validation,
    # archive extraction, filesystem moves, cleanup and rollback code.
    function Invoke-RestMethod { param($Uri, $Headers); $script:apiCalls++; return $script:metadata }
    function Receive-BonsaiDownload([string]$Uri, [string]$Path) {
        $script:downloadCalls++
        $source = Join-Path $fixture (($Uri -split '/')[-1])
        if ($script:corruptDownload) { [IO.File]::WriteAllBytes($Path, [byte[]]::new((Get-Item -LiteralPath $source).Length)) }
        else { Copy-Item -LiteralPath $source -Destination $Path }
    }
    function Test-BonsaiRuntimeSmoke([string]$Directory, [string]$ReleaseTag) {
        if ($script:smokeFails) { throw 'simulated missing dependent DLL' }
        if ($script:activateAfterSmoke) { $script:activeProcess = $true }
    }
    function Get-Process {
        param($Name, $ErrorAction)
        if ($script:activeProcess) { [pscustomobject]@{ Path = (Join-Path $destination 'llama-server.exe'); Id = 123; Name = 'llama-server' } }
    }
    $script:corruptDownload = $true
    Assert-Throws { Invoke-BonsaiCudaInstall $temporaryRoot '13.3' } 'SHA256 mismatch'
    Assert-True (Test-Path -LiteralPath $sentinel) 'Corrupt download preserved the working runtime'
    $script:corruptDownload = $false; $script:smokeFails = $true
    Assert-Throws { Invoke-BonsaiCudaInstall $temporaryRoot '13.3' } 'missing dependent DLL'
    Assert-True (Test-Path -LiteralPath $sentinel) 'Failed smoke test preserved the working runtime'
    Assert-True (@(Get-ChildItem (Split-Path $destination) -Directory -Filter 'cuda.staging-*').Count -eq 0) 'Failed staging was cleaned up'
    $script:smokeFails = $false; $script:activateAfterSmoke = $true
    Assert-Throws { Invoke-BonsaiCudaInstall $temporaryRoot '13.3' } 'in use'
    Assert-True (Test-Path -LiteralPath $sentinel) 'A process starting during download prevented replacement'
    $script:activeProcess = $false; $script:activateAfterSmoke = $false
    $script:failPromotion = $true
    function Move-Item {
        param($LiteralPath, $Destination, [switch]$Force)
        if ($script:failPromotion -and $LiteralPath -match 'cuda\.staging-' -and $Destination -eq $destination) {
            throw 'simulated directory promotion failure'
        }
        Microsoft.PowerShell.Management\Move-Item -LiteralPath $LiteralPath -Destination $Destination -Force:$Force
    }
    Assert-Throws { Invoke-BonsaiCudaInstall $temporaryRoot '13.3' } 'promotion failure'
    Assert-True (Test-Path -LiteralPath $sentinel) 'Failed promotion rolled back the previous directory'
    Remove-Item Function:\Move-Item
    Invoke-BonsaiCudaInstall $temporaryRoot '13.3'
    Assert-True (-not (Test-Path -LiteralPath $sentinel)) 'Verified runtime replaced the old directory'
    Assert-True (Test-BonsaiCudaFiles $destination) 'Both the executable and CUDA DLL packages were installed'
    Assert-True ((Get-Content -LiteralPath (Join-Path $destination '.llama_release') -Raw) -eq "$tag cuda-13.3") 'Release stamp matches'
    $record = Get-Content -LiteralPath (Join-Path $destination '.llama_release.json') -Raw | ConvertFrom-Json
    Assert-True ($record.assets.Count -eq 2 -and $record.assets[0].digest -match '^sha256:') 'Installed hashes are recorded'
    $apiBefore = $script:apiCalls; $downloadBefore = $script:downloadCalls
    Invoke-BonsaiCudaInstall $temporaryRoot '13.3'
    Assert-True ($script:apiCalls -eq $apiBefore -and $script:downloadCalls -eq $downloadBefore) 'Current complete runtime is an offline no-op'
    $oldDigest = $script:metadata.assets[0].digest; $script:metadata.assets[0].digest = $null
    Assert-Throws { Get-BonsaiReleaseAssets $tag $assetNames } 'No SHA256 digest'
    $script:metadata.assets[0].digest = $oldDigest
    # Public GitHub API rate limits should reuse an existing gh session without
    # extracting credentials; an environment token must stay in request headers.
    if (-not ('BonsaiTestApiResponse' -as [type])) {
        Add-Type -TypeDefinition @'
public sealed class BonsaiTestApiResponse : System.Net.WebResponse {
    public System.Net.HttpStatusCode StatusCode { get; private set; }
    public BonsaiTestApiResponse(int status) { StatusCode = (System.Net.HttpStatusCode)status; }
}
'@
    }
    $env:FAKE_GITHUB_METADATA = Join-Path $fixture 'release.json'
    [IO.File]::WriteAllText($env:FAKE_GITHUB_METADATA, ($script:metadata | ConvertTo-Json -Depth 5))
    $fakeGh = Join-Path $fixture 'fake-gh.ps1'
    [IO.File]::WriteAllText($fakeGh, @'
if ($args[0] -ne 'api' -or $args[1] -notmatch '^repos/PrismML-Eng/llama.cpp/releases/tags/prism-b') { exit 9 }
[IO.File]::ReadAllText($env:FAKE_GITHUB_METADATA)
$global:LASTEXITCODE = 0
'@)
    function Invoke-RestMethod {
        param($Uri, $Headers)
        $script:lastHeaders = $Headers.Clone()
        if ($script:rateLimited) {
            throw [Net.WebException]::new('rate limited', $null, [Net.WebExceptionStatus]::ProtocolError, [BonsaiTestApiResponse]::new(403))
        }
        return $script:metadata
    }
    function Get-Command {
        param($Name, $CommandType, $ErrorAction)
        # Duplicate PATH entries must still select one executable.
        @([pscustomobject]@{ Source = $fakeGh }, [pscustomobject]@{ Source = 'missing-second-gh.exe' })
    }
    $env:GH_TOKEN = 'test-only-token'; $script:rateLimited = $false
    $null = Get-BonsaiReleaseAssets $tag $assetNames
    Assert-True ($script:lastHeaders.Authorization -eq 'Bearer test-only-token') 'Existing token is passed in memory as a header'
    $script:rateLimited = $true
    $fallbackAssets = @(Get-BonsaiReleaseAssets $tag $assetNames)
    Assert-True ($fallbackAssets.Count -eq 2 -and $fallbackAssets[0].digest -eq $oldDigest) 'Rate-limit fallback validated gh metadata'
    $script:lastHeaders.Clear(); $script:rateLimited = $false
    Remove-Item Function:\Get-Command
    Set-Item Function:\Receive-BonsaiDownload -Value $productionDownloader
    $fakeCurl = Join-Path $fixture 'fake-curl.ps1'
    $env:FAKE_CURL_LOG = Join-Path $fixture 'curl-calls.txt'
    [IO.File]::WriteAllText($fakeCurl, @'
$arguments = @($args)
[IO.File]::AppendAllText($env:FAKE_CURL_LOG, (($arguments | ConvertTo-Json -Compress) + [Environment]::NewLine))
if ($env:FAKE_CURL_FAIL_RANGE -eq '1' -and $arguments -contains '--continue-at') { exit 33 }
$outputIndex = [array]::IndexOf($arguments, '--output')
$output = $arguments[$outputIndex + 1]
if ($arguments -contains '--continue-at') { [IO.File]::AppendAllText($output, 'tail') }
else { [IO.File]::WriteAllText($output, 'prefixtail') }
$global:LASTEXITCODE = 0
'@)
    function Get-Command {
        param($Name, $CommandType, $ErrorAction)
        @([pscustomobject]@{ Source = $fakeCurl }, [pscustomobject]@{ Source = 'missing-second-curl.exe' })
    }
    $curlPartial = Join-Path $fixture 'curl-download.part'
    [IO.File]::WriteAllText($curlPartial, 'prefix')
    Receive-BonsaiDownload 'https://example.invalid/fixture' $curlPartial
    Assert-True ([IO.File]::ReadAllText($curlPartial) -eq 'prefixtail') 'Duplicate curl PATH candidates still resumed with a single command'
    [IO.File]::WriteAllText($curlPartial, 'prefix'); $env:FAKE_CURL_FAIL_RANGE = '1'
    Receive-BonsaiDownload 'https://example.invalid/fixture' $curlPartial
    Assert-True ([IO.File]::ReadAllText($curlPartial) -eq 'prefixtail') 'An unsupported Range response retried from byte zero'
    Assert-True (@(Get-Content -LiteralPath $env:FAKE_CURL_LOG).Count -eq 3) 'Range fallback issued exactly one additional download'
    Remove-Item Function:\Get-Command
    $unsafeZip = Join-Path $fixture 'unsafe.zip'
    $zip = [IO.Compression.ZipFile]::Open($unsafeZip, [IO.Compression.ZipArchiveMode]::Create)
    try { $null = $zip.CreateEntry('../escaped.txt') } finally { $zip.Dispose() }
    Assert-Throws { Expand-BonsaiRuntimeArchive $unsafeZip $destination } 'Unsafe archive entry'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $temporaryRoot 'bin\escaped.txt'))) 'Archive traversal did not escape staging'
    Assert-Throws { Remove-BonsaiTemporaryDirectory $temporaryRoot (Split-Path $temporaryRoot) 'cuda.staging-' } 'Refusing to remove'
    Write-Host 'PASS: runtime digest, smoke, process race, rollback, stamps, no-op, API fallback and archive boundary tests'

    # Calibration exercises a fake tool that writes real GGUF bias fixtures.
    $env:BONSAI_GGUF = ''; $env:BONSAI_FAMILY = 'bonsai2'; $env:BONSAI_MODEL = '27B'; $env:BONSAI_FORCE_G64 = ''
    $modelDirectory = Join-Path $temporaryRoot 'models\bonsai2-gguf\27B'
    $pq2 = Join-Path $modelDirectory 'Ternary-Bonsai-2-27B-PQ2_0.gguf'
    $ptq1 = Join-Path $modelDirectory 'Ternary-Bonsai-2-27B-PTQ1_0.gguf'
    Set-Content -LiteralPath $pq2 -Value 'PQ2 weights'; Set-Content -LiteralPath $ptq1 -Value 'PTQ1 weights'
    $corpus = Join-Path $temporaryRoot 'corpus.txt'; Set-Content -LiteralPath $corpus -Value 'Synthetic fixture corpus'
    $validBias = Join-Path $fixture 'valid-bias.gguf'; Write-FakeBias $validBias $false
    Test-BonsaiBiasFile $validBias
    $rotatedBias = Join-Path $fixture 'rotated-bias.gguf'; Write-FakeBias $rotatedBias $true
    Assert-Throws { Test-BonsaiBiasFile $rotatedBias } 'unrotated'
    $truncatedBias = Join-Path $fixture 'truncated-bias.gguf'
    $bytes = [IO.File]::ReadAllBytes($validBias); [IO.File]::WriteAllBytes($truncatedBias, $bytes[0..($bytes.Length-17)])
    Assert-Throws { Test-BonsaiBiasFile $truncatedBias } 'Truncated'
    $tool = Join-Path $fixture 'fake-calibrator.ps1'
    $fakeToolSource = @'
$arguments = @($args)
$outIndex = [array]::IndexOf($arguments, '-o')
$output = $arguments[$outIndex + 1]
[IO.File]::WriteAllText($env:FAKE_KV_LOG, (@{ rotation = $env:LLAMA_ATTN_ROT_DISABLE; args = $arguments } | ConvertTo-Json -Depth 4))
if ($env:FAKE_KV_MODE -eq 'fail') { [IO.File]::WriteAllText($output, 'partial'); exit 7 }
Copy-Item -LiteralPath $env:FAKE_BIAS_FIXTURE -Destination $output
$global:LASTEXITCODE = 0
'@
    [IO.File]::WriteAllText($tool, $fakeToolSource)
    $env:FAKE_BIAS_FIXTURE = $validBias; $env:FAKE_KV_LOG = Join-Path $fixture 'tool-log.json'
    $env:LLAMA_ATTN_ROT_DISABLE = 'prior-value'; $pathBefore = $env:Path
    $plan = Invoke-BonsaiKvCalibration -RepositoryRoot $temporaryRoot -BinaryPath $tool -CorpusPath $corpus -DryRun
    Assert-True ($plan.Model -eq $pq2 -and $plan.Output -eq "$($pq2.Substring(0,$pq2.Length-5))-kv-bias.gguf") 'PQ2 is preferred with an exact model filename'
    Invoke-BonsaiKvCalibration -RepositoryRoot $temporaryRoot -BinaryPath $tool -CorpusPath $corpus
    $outputBias = $plan.Output; $biasHash = (Get-FileHash -LiteralPath $outputBias).Hash
    $calibrationLog = Get-Content -LiteralPath $env:FAKE_KV_LOG -Raw | ConvertFrom-Json
    Assert-True ($calibrationLog.rotation -eq '1' -and $calibrationLog.args -contains 'q4_0') 'Calibration uses matching Q4 and rotation settings'
    Assert-True ($env:LLAMA_ATTN_ROT_DISABLE -eq 'prior-value' -and $env:Path -eq $pathBefore) 'Successful calibration restored the environment'
    $biasRecord = Get-Content -LiteralPath "$outputBias.json" -Raw | ConvertFrom-Json
    Assert-True ($biasRecord.model_file -eq (Split-Path $pq2 -Leaf) -and $biasRecord.bias_sha256 -eq $biasHash.ToLowerInvariant()) 'Metadata identifies the model and bias'
    $env:FAKE_KV_MODE = 'fail'
    Assert-Throws { Invoke-BonsaiKvCalibration -RepositoryRoot $temporaryRoot -BinaryPath $tool -CorpusPath $corpus } 'exit 7'
    Assert-True ((Get-FileHash -LiteralPath $outputBias).Hash -eq $biasHash) 'Failed calibration preserved the old bias'
    $env:FAKE_KV_MODE = ''; $env:FAKE_BIAS_FIXTURE = $rotatedBias
    Assert-Throws { Invoke-BonsaiKvCalibration -RepositoryRoot $temporaryRoot -BinaryPath $tool -CorpusPath $corpus } 'unrotated'
    Assert-True ((Get-FileHash -LiteralPath $outputBias).Hash -eq $biasHash) 'Mismatched rotation did not replace the bias'
    Assert-True (@(Get-ChildItem -LiteralPath $modelDirectory -Filter '.kv-calibration-*').Count -eq 0) 'Failed calibration left no discoverable temporary bias'
    Assert-True ($env:LLAMA_ATTN_ROT_DISABLE -eq 'prior-value' -and $env:Path -eq $pathBefore) 'Failed calibration restored the environment'
    Remove-Item -LiteralPath $pq2
    $plan = Invoke-BonsaiKvCalibration -RepositoryRoot $temporaryRoot -BinaryPath $tool -CorpusPath $corpus -DryRun
    Assert-True ($plan.Model -eq $ptq1) 'PTQ1 fallback matches the launcher'
    Write-Host 'PASS: GGUF bounds/basis, exact model selection, atomic calibration failure and environment restoration tests'
} finally {
    foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name]) }
    $resolved = [IO.Path]::GetFullPath($temporaryRoot)
    if ((Split-Path $resolved -Leaf) -like 'bonsai-runtime-test-*' -and
        (Split-Path $resolved -Parent).TrimEnd('\') -eq ([IO.Path]::GetTempPath()).TrimEnd('\')) {
        if (Test-Path -LiteralPath $resolved) { Microsoft.PowerShell.Management\Remove-Item -LiteralPath $resolved -Recurse -Force }
    }
}
