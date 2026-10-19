param([switch]$KeepTunnel)
$ErrorActionPreference = 'Stop'
$RootPrefix = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\') + '\'
$Names = @('llama-server.exe')
if (-not $KeepTunnel) { $Names += 'frpc.exe' }
foreach ($Name in $Names) {
    foreach ($ProcessInfo in @(Get-CimInstance Win32_Process -Filter "Name = '$Name'")) {
        if ($ProcessInfo.ExecutablePath -and $ProcessInfo.ExecutablePath.StartsWith($RootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            Stop-Process -Id $ProcessInfo.ProcessId -ErrorAction Stop
            Write-Host "Stopped $Name (PID $($ProcessInfo.ProcessId))."
        }
    }
}
