# One entry point for double-click and PowerShell launches. Keep secrets local.
[CmdletBinding(PositionalBinding=$false)]
param(
    [ValidateSet("auto", "balanced", "throughput", "long", "max")]
    [string]$Profile = "",
    [int]$Port = 0,
    [switch]$NoBrowser,
    [switch]$NoTunnel,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ServerArguments
)
$ErrorActionPreference = "Stop"
$ExtraArgs = @($ServerArguments | Where-Object { $null -ne $_ })
$LocalConfig = $null
$ConfigPath = Join-Path $PSScriptRoot ".bonsai-server.local.json"
if (Test-Path -LiteralPath $ConfigPath) { $LocalConfig = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json }
if (-not $Profile) { $Profile = if ($env:BONSAI_PROFILE) { $env:BONSAI_PROFILE } else { "balanced" } }
if (-not $Port) { $Port = if ($env:BONSAI_PORT) { [int]$env:BONSAI_PORT } else { 8080 } }
for ($i = 0; $i -lt $ExtraArgs.Count - 1; $i++) {
    if ($ExtraArgs[$i] -eq "--port") { $Port = [int]$ExtraArgs[$i + 1] }
}
if ($Port -lt 1 -or $Port -gt 65535) { throw "Port must be between 1 and 65535." }
$PriorKey = $env:BONSAI_API_KEY
if (-not $env:BONSAI_API_KEY -and $LocalConfig.ApiKey) { $env:BONSAI_API_KEY = $LocalConfig.ApiKey }
$BrowserJob = $null
try {
    $FrpExe = Join-Path $PSScriptRoot "frp\frpc.exe"
    $FrpConfig = Join-Path $PSScriptRoot "frp\frpc.toml"
    $EnableTunnel = -not $NoTunnel -and $LocalConfig.EnableTunnel -and (Test-Path -LiteralPath $FrpExe) -and (Test-Path -LiteralPath $FrpConfig)
    if ($EnableTunnel -and -not $env:BONSAI_API_KEY) { throw "Set BONSAI_API_KEY or ApiKey in .bonsai-server.local.json before enabling the tunnel." }
    if (-not $DryRun) {
        $Headers = @{}
        if ($env:BONSAI_API_KEY) { $Headers.Authorization = "Bearer $env:BONSAI_API_KEY" }
        $Existing = $false
        try { $null = Invoke-WebRequest "http://127.0.0.1:$Port/health" -Headers $Headers -TimeoutSec 2 -UseBasicParsing; $Existing = $true } catch {}
        if ($Existing) {
            Write-Host "A server is already running on port $Port. Stop it before changing profiles."
            exit 1
        }
        if ($EnableTunnel -and -not (Get-Process -Name frpc -ErrorAction SilentlyContinue)) {
            Start-Process -FilePath $FrpExe -ArgumentList @("-c", ('"' + $FrpConfig + '"')) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
        }
        $OpenBrowser = -not $NoBrowser -and ($null -eq $LocalConfig -or $LocalConfig.OpenBrowser -ne $false)
        if ($OpenBrowser) {
            $BrowserJob = Start-Job -ArgumentList $Port,$env:BONSAI_API_KEY -ScriptBlock {
                param($ServerPort, $Key)
                $Headers = @{}
                if ($Key) { $Headers.Authorization = "Bearer $Key" }
                for ($i = 0; $i -lt 90; $i++) {
                    try {
                        $null = Invoke-WebRequest "http://127.0.0.1:$ServerPort/health" -Headers $Headers -TimeoutSec 1 -UseBasicParsing
                        Start-Process "http://localhost:$ServerPort"
                        break
                    } catch { Start-Sleep -Seconds 1 }
                }
            }
        }
    }
    & (Join-Path $PSScriptRoot "scripts\start_llama_server.ps1") -Profile $Profile -Port $Port -DryRun:$DryRun @ExtraArgs
    $Code = $LASTEXITCODE
} finally {
    if ($BrowserJob) { Stop-Job $BrowserJob -ErrorAction SilentlyContinue; Remove-Job $BrowserJob -Force -ErrorAction SilentlyContinue }
    if ($null -eq $PriorKey) { Remove-Item Env:BONSAI_API_KEY -ErrorAction SilentlyContinue } else { $env:BONSAI_API_KEY = $PriorKey }
}
exit $Code
