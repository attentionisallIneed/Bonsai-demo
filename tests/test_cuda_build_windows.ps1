# Offline build-script regression tests using tiny native-process fixtures.
# No CUDA compiler, GPU, network or Visual Studio installation is required.
$ErrorActionPreference = 'Stop'
$buildScript = Join-Path (Split-Path $PSScriptRoot -Parent) 'scripts\build_cuda_windows.ps1'
$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ('bonsai-cuda-build-test-' + [guid]::NewGuid().ToString('N'))
$shellPath = (Get-Process -Id $PID).Path
$windowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'

function Assert-True($Condition, [string]$Message) {
    if (-not $Condition) { throw "Assertion failed: $Message" }
}

try {
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    $nativeSource = Join-Path $temporaryRoot 'ToolStub.cs'
    [IO.File]::WriteAllText($nativeSource, @'
using System;
using System.Diagnostics;
using System.IO;
public static class ToolStub {
    private static bool Flag(string name) { return Environment.GetEnvironmentVariable(name) == "1"; }
    private static void Log(string kind, string[] args) {
        File.AppendAllText(Environment.GetEnvironmentVariable("CUDA_BUILD_TEST_LOG"),
            kind + "\t" + String.Join("\t", args) + Environment.NewLine);
    }
    public static int Main(string[] args) {
        string name = Path.GetFileNameWithoutExtension(Process.GetCurrentProcess().MainModule.FileName).ToLowerInvariant();
        if (name == "nvcc") {
            Log("nvcc", args);
            Console.WriteLine("Cuda compilation tools, release " + Environment.GetEnvironmentVariable("CUDA_BUILD_TEST_VERSION") + ", V0.0.0");
            return Flag("CUDA_BUILD_TEST_FAIL_NVCC") ? 16 : 0;
        }
        if (name == "git") {
            Log("clone", args);
            string repo = args[args.Length - 1];
            Directory.CreateDirectory(repo);
            if (Flag("CUDA_BUILD_TEST_FAIL_CLONE")) {
                string stale = Path.Combine(repo, "build-cuda", "bin", "Release");
                Directory.CreateDirectory(stale);
                File.WriteAllText(Path.Combine(stale, "llama-server.exe"), "stale clone output");
                return 17;
            }
            File.WriteAllText(Path.Combine(repo, "CMakeLists.txt"), "# fixture checkout");
            return 0;
        }
        bool building = args.Length > 0 && args[0] == "--build";
        Log(building ? "build" : "configure", args);
        if (!building) { return Flag("CUDA_BUILD_TEST_FAIL_CONFIGURE") ? 18 : 0; }
        if (Flag("CUDA_BUILD_TEST_FAIL_BUILD")) { return 19; }
        if (!Flag("CUDA_BUILD_TEST_NO_OUTPUT")) {
            string output = Path.Combine(Directory.GetCurrentDirectory(), "build-cuda", "bin", "Release");
            Directory.CreateDirectory(output);
            File.WriteAllText(Path.Combine(output, "llama-server.exe"), "fresh server");
            File.WriteAllText(Path.Combine(output, "llama-cli.exe"), "fresh cli");
            File.WriteAllText(Path.Combine(output, "ggml-cuda.dll"), "fresh CUDA backend");
        }
        return 0;
    }
}
'@)
    # Windows PowerShell's bundled compiler emits a .NET Framework executable
    # that both PowerShell editions can launch as a real native command.
    $compilerScript = Join-Path $temporaryRoot 'compile-fixture.ps1'
    [IO.File]::WriteAllText($compilerScript, @'
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition ([IO.File]::ReadAllText($args[0])) -OutputAssembly $args[1] -OutputType ConsoleApplication
'@)
    $toolStub = Join-Path $temporaryRoot 'tool-stub.exe'
    & $windowsPowerShell -NoProfile -ExecutionPolicy Bypass -File $compilerScript $nativeSource $toolStub
    Assert-True ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $toolStub)) 'Native test fixture compiled'
    $tools = Join-Path $temporaryRoot 'tools'
    New-Item -ItemType Directory -Path $tools | Out-Null
    foreach ($name in @('git.exe', 'cmake.exe')) { Copy-Item -LiteralPath $toolStub -Destination (Join-Path $tools $name) }
    $cudaPath = Join-Path $temporaryRoot 'CUDA toolkit'
    New-Item -ItemType Directory -Path (Join-Path $cudaPath 'bin') -Force | Out-Null
    Copy-Item -LiteralPath $toolStub -Destination (Join-Path $cudaPath 'bin\nvcc.exe')
    $runner = Join-Path $temporaryRoot 'run-build.ps1'
    [IO.File]::WriteAllText($runner, @'
$ErrorActionPreference = 'Stop'
$configuration = Get-Content -LiteralPath $args[0] -Raw | ConvertFrom-Json
Set-Location -LiteralPath $configuration.directory
$env:Path = $configuration.tools + ';' + $env:Path
$env:CUDACXX = 'previous compiler value'
$env:CUDA_BUILD_TEST_LOG = $configuration.log
$env:CUDA_BUILD_TEST_VERSION = $configuration.version
$env:CUDA_BUILD_TEST_FAIL_CLONE = $configuration.fail_clone
$env:CUDA_BUILD_TEST_FAIL_NVCC = $configuration.fail_nvcc
$env:CUDA_BUILD_TEST_FAIL_CONFIGURE = $configuration.fail_configure
$env:CUDA_BUILD_TEST_FAIL_BUILD = $configuration.fail_build
$env:CUDA_BUILD_TEST_NO_OUTPUT = $configuration.no_output
$before = (Get-Location).Path
$message = ''; $failed = $false
try {
    & $configuration.script -CudaPath $configuration.cuda_path -Archs $configuration.archs -RepoDir '.\source checkout' -OutputDir 'test-cuda'
} catch {
    $message = $_.Exception.Message
    $failed = $true
} finally {
    @{ cwd_before = $before; cwd_after = (Get-Location).Path; cudacxx = $env:CUDACXX; message = $message } |
        ConvertTo-Json | Set-Content -LiteralPath $configuration.result
}
if ($failed) { exit 1 }
'@)

    function Run-BuildCase {
        param([string]$Name, [string]$Version = '13.3', [string]$Archs = '',
            [switch]$FailClone, [switch]$FailNvcc, [switch]$FailConfigure, [switch]$FailBuild,
            [switch]$NoOutput, [switch]$SeedStale, [switch]$MissingCheckout)
        $directory = Join-Path $temporaryRoot $Name
        New-Item -ItemType Directory -Path $directory | Out-Null
        if (-not $MissingCheckout) {
            New-Item -ItemType Directory -Path (Join-Path $directory 'source checkout') | Out-Null
        }
        $installed = Join-Path $directory 'bin\test-cuda'
        New-Item -ItemType Directory -Path $installed -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $installed 'llama-server.exe') -Value 'keep installed server'
        if ($SeedStale) {
            $stale = Join-Path $directory 'source checkout\build-cuda\bin\Release'
            New-Item -ItemType Directory -Path $stale -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $stale 'llama-server.exe') -Value 'stale server must not be copied'
        }
        $configuration = [ordered]@{ directory = $directory; script = $buildScript; cuda_path = $cudaPath; tools = $tools
            version = $Version; archs = $Archs; log = (Join-Path $directory 'native-calls.txt'); result = (Join-Path $directory 'result.json')
            fail_clone = [string][int][bool]$FailClone; fail_nvcc = [string][int][bool]$FailNvcc
            fail_configure = [string][int][bool]$FailConfigure; fail_build = [string][int][bool]$FailBuild; no_output = [string][int][bool]$NoOutput }
        $configurationPath = Join-Path $directory 'configuration.json'
        $configuration | ConvertTo-Json | Set-Content -LiteralPath $configurationPath
        $output = & $shellPath -NoProfile -ExecutionPolicy Bypass -File $runner $configurationPath 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
        Assert-True (Test-Path -LiteralPath $configuration.result) "$Name reported restoration state: $output"
        $result = Get-Content -LiteralPath $configuration.result -Raw | ConvertFrom-Json
        Assert-True ($result.cwd_before -eq $result.cwd_after -and $result.cwd_after -eq $directory) "$Name restored the current location"
        Assert-True ($result.cudacxx -eq 'previous compiler value') "$Name restored CUDACXX"
        $calls = if (Test-Path -LiteralPath $configuration.log) { @(Get-Content -LiteralPath $configuration.log) } else { @() }
        return [pscustomobject]@{ ExitCode = $exitCode; Calls = $calls; Message = $result.message
            Installed = ((Get-Content -LiteralPath (Join-Path $installed 'llama-server.exe') -Raw).Trim()); Output = $output }
    }

    foreach ($version in @('12.4', '12.6', '12.8', '13.3')) {
        $case = Run-BuildCase -Name "success-$version" -Version $version
        Assert-True ($case.ExitCode -eq 0 -and $case.Installed -eq 'fresh server') "CUDA $version succeeds and copies the built output: $($case.Output)"
        $expected = if ($version -eq '13.3') { '80;86;89;90;100;120' } elseif ($version -eq '12.8') { '80;86;89;90;120' } else { '80;86;89;90' }
        Assert-True (@($case.Calls | Where-Object { $_ -match ('-DCMAKE_CUDA_ARCHITECTURES=' + [regex]::Escape($expected) + '(\t|$)') }).Count -eq 1) "CUDA $version default architecture set"
    }
    $case = Run-BuildCase -Name 'custom-archs' -Version '12.4' -Archs '86-real;89-virtual'
    Assert-True ($case.ExitCode -eq 0 -and @($case.Calls | Where-Object { $_ -match '86-real;89-virtual' }).Count -eq 1) 'Explicit -Archs is forwarded unchanged'
    $case = Run-BuildCase -Name 'reject-blackwell' -Version '12.6' -Archs '80;120' -SeedStale
    Assert-True ($case.ExitCode -ne 0 -and $case.Message -match 'require CUDA 12.8' -and $case.Installed -eq 'keep installed server') 'Unsupported explicit SM120 is rejected before copying'
    $case = Run-BuildCase -Name 'clone-failure' -FailClone -MissingCheckout
    Assert-True ($case.ExitCode -ne 0 -and $case.Message -match 'git clone failed.*17' -and $case.Calls.Count -eq 1 -and $case.Installed -eq 'keep installed server') "Clone native exit 17 aborts before configuring or copying: $($case | ConvertTo-Json -Depth 4)"
    $case = Run-BuildCase -Name 'clone-success' -MissingCheckout
    Assert-True ($case.ExitCode -eq 0 -and $case.Installed -eq 'fresh server' -and $case.Calls[0] -match '^clone\t') 'Successful clone continues normally'
    $case = Run-BuildCase -Name 'nvcc-failure' -FailNvcc -SeedStale
    Assert-True ($case.ExitCode -ne 0 -and $case.Message -match 'nvcc --version failed.*16' -and $case.Calls.Count -eq 1 -and $case.Installed -eq 'keep installed server') 'Nonzero nvcc exit rejects even parseable version output'
    $case = Run-BuildCase -Name 'configure-failure' -FailConfigure -SeedStale
    Assert-True ($case.ExitCode -ne 0 -and $case.Message -match 'configure failed.*18' -and @($case.Calls | Where-Object { $_ -match '^build\t' }).Count -eq 0 -and $case.Installed -eq 'keep installed server') 'Configure native exit 18 prevents build and stale artifact copying'
    $case = Run-BuildCase -Name 'build-failure' -FailBuild -SeedStale
    Assert-True ($case.ExitCode -ne 0 -and $case.Message -match 'build failed.*19' -and $case.Installed -eq 'keep installed server') 'Build native exit 19 prevents stale artifact copying'
    $case = Run-BuildCase -Name 'no-output' -NoOutput
    Assert-True ($case.ExitCode -ne 0 -and $case.Message -match 'produced no llama-' -and $case.Installed -eq 'keep installed server') 'A successful build with no artifacts cannot report success'
    Write-Host 'PASS: CUDA 12.4/12.6/12.8/13.3 defaults, explicit architectures, native failures, stale outputs, cwd and CUDACXX restoration'
} finally {
    $resolved = [IO.Path]::GetFullPath($temporaryRoot)
    if ((Split-Path $resolved -Leaf) -like 'bonsai-cuda-build-test-*' -and
        (Split-Path $resolved -Parent).TrimEnd('\') -eq ([IO.Path]::GetTempPath()).TrimEnd('\')) {
        if (Test-Path -LiteralPath $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
    }
}

# GitHub Actions checks LASTEXITCODE after dot-sourcing a PowerShell step. The
# final child intentionally fails; only clear its status after every assertion
# and cleanup have succeeded, so a real fixture failure still fails the step.
$global:LASTEXITCODE = 0
