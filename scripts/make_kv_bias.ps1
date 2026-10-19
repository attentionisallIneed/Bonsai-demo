<#
.SYNOPSIS
Calibrate a model-specific Q4 K-cache bias with the pinned Prism runtime.
.EXAMPLE
.\scripts\make_kv_bias.ps1
.EXAMPLE
.\scripts\make_kv_bias.ps1 .\my-corpus.txt -Chunks 4
.EXAMPLE
.\scripts\make_kv_bias.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$CorpusPath,
    [string]$ModelPath,
    [string]$BinaryPath,
    [string]$OutputPath,
    [ValidateRange(32, 32768)][int]$ContextSize = 512,
    [ValidateRange(-1, 2147483647)][int]$Chunks = -1,
    [ValidateRange(0, 999)][int]$GpuLayers = 99,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
if (-not $PSBoundParameters.ContainsKey('GpuLayers') -and $env:BONSAI_NGL) {
    $parsedLayers = 0
    if (-not [int]::TryParse($env:BONSAI_NGL, [ref]$parsedLayers) -or $parsedLayers -lt 0 -or $parsedLayers -gt 999) {
        throw "BONSAI_NGL must be between 0 and 999."
    }
    $PSBoundParameters['GpuLayers'] = $parsedLayers
}
. (Join-Path $PSScriptRoot 'kv_bias_helpers.ps1')
Invoke-BonsaiKvCalibration -RepositoryRoot (Split-Path $PSScriptRoot -Parent) @PSBoundParameters
