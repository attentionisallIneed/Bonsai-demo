<#
.SYNOPSIS
Safely install the pinned Prism CUDA runtime, without downloading models or Python.
.EXAMPLE
.\scripts\download_cuda_bin.ps1 13.3
.EXAMPLE
.\scripts\download_cuda_bin.ps1 -CudaTag 12.4 -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][ValidateSet("12.4", "13.3")][string]$CudaTag = "13.3",
    [switch]$Force,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "runtime_helpers.ps1")
Invoke-BonsaiCudaInstall -RepositoryRoot (Split-Path $PSScriptRoot -Parent) -CudaTag $CudaTag -Force:$Force -DryRun:$DryRun
