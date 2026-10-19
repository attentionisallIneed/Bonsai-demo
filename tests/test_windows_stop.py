"""Stop only this checkout's services; all process operations are mocked."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHELL = shutil.which('pwsh') or shutil.which('powershell')


@unittest.skipUnless(SHELL and os.name == 'nt', 'Windows PowerShell is required')
class StopTests(unittest.TestCase):
    def test_project_boundary_and_keep_tunnel(self):
        with tempfile.TemporaryDirectory(prefix='bonsai stop ') as folder:
            # Hosted Windows runners expose an 8.3 TEMP path (RUNNER~1), while
            # PowerShell expands PSScriptRoot to its long form.
            folder = str(Path(folder).resolve())
            shutil.copy2(ROOT / 'stop_server.ps1', Path(folder) / 'stop_server.ps1')
            environment = dict(os.environ, BONSAI_TEST_ROOT=folder)
            for keep, expected in ((False, [100, 102]), (True, [100])):
                command = r'''
$global:Stopped = @()
function Get-CimInstance {
    param($ClassName, $Filter)
    if ($Filter -eq "Name = 'llama-server.exe'") {
        [pscustomobject]@{ExecutablePath=($env:BONSAI_TEST_ROOT + '\bin\cuda\llama-server.exe');ProcessId=100}
        [pscustomobject]@{ExecutablePath=($env:BONSAI_TEST_ROOT + '-other\bin\llama-server.exe');ProcessId=101}
        [pscustomobject]@{ExecutablePath=$null;ProcessId=104}
    } elseif ($Filter -eq "Name = 'frpc.exe'") {
        [pscustomobject]@{ExecutablePath=($env:BONSAI_TEST_ROOT + '\frp\frpc.exe');ProcessId=102}
        [pscustomobject]@{ExecutablePath='C:\elsewhere\frpc.exe';ProcessId=103}
    }
}
function Stop-Process { param($Id, $ErrorAction) $global:Stopped += $Id }
& (Join-Path $env:BONSAI_TEST_ROOT 'stop_server.ps1') KEEP_SWITCH
Write-Output ('RESULT=' + (ConvertTo-Json -InputObject @($global:Stopped) -Compress))
'''.replace('KEEP_SWITCH', '-KeepTunnel' if keep else '')
                result = subprocess.run([SHELL, '-NoProfile', '-Command', command], env=environment,
                                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                line = next(line for line in result.stdout.splitlines() if line.startswith('RESULT='))
                self.assertEqual(json.loads(line[7:]), expected)


if __name__ == '__main__':
    unittest.main()
