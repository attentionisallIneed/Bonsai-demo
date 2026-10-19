"""Exercise the real PowerShell argument boundary without a model or GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHELL = shutil.which('pwsh') or shutil.which('powershell')


def option(arguments, *names):
    values = [arguments[i + 1] for i, arg in enumerate(arguments[:-1]) if arg in names]
    return values[-1] if values else None


@unittest.skipUnless(SHELL, 'PowerShell is required')
class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='bonsai launcher ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'scripts').mkdir()
        for name in ('run_server.ps1', 'run_server.bat'):
            shutil.copy2(ROOT / name, self.root / name)
        for name in ('start_llama_server.ps1', 'server-profiles.json'):
            shutil.copy2(ROOT / 'scripts' / name, self.root / 'scripts' / name)
        models = self.root / 'models' / 'bonsai2-gguf' / '27B'
        models.mkdir(parents=True)
        for name in ('Ternary-Bonsai-2-27B-PQ2_0.gguf', 'Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf'):
            (models / name).touch()
        binary = self.root / 'bin' / 'cuda'
        binary.mkdir(parents=True)
        (binary / 'llama-server.exe').touch()
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('BONSAI_', 'LLAMA_ARG_'))}

    def run_launcher(self, *args, script='run_server.ps1', environment=None, success=True):
        env = dict(self.env, **(environment or {}))
        completed = subprocess.run([SHELL, '-NoProfile', '-File', str(self.root / script), '-DryRun', *args],
                                   cwd=self.root, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=25)
        if not success:
            self.assertNotEqual(completed.returncode, 0)
            return completed
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        start = completed.stdout.index('{\n')
        return json.loads(completed.stdout[start:]), completed.stdout

    def test_default_is_single_64k_and_preserves_local_auth_without_printing_it(self):
        (self.root / '.bonsai-server.local.json').write_text(json.dumps({'ApiKey': 'local-test-secret'}))
        report, output = self.run_launcher()
        self.assertEqual(report['profile'], 'balanced')
        self.assertEqual(option(report['arguments'], '-c'), '65536')
        self.assertEqual(option(report['arguments'], '--parallel'), '1')
        self.assertNotIn(None, report['arguments'])
        self.assertEqual(option(report['arguments'], '--api-key'), '<redacted>')
        self.assertNotIn('local-test-secret', output)
        self.assertIn('bonsai launcher ', report['binary'])

    def test_cli_overrides_environment_which_overrides_profile(self):
        report, _ = self.run_launcher('--parallel', '3', '-ub', '1024', '-c', '98304',
                                      environment={'BONSAI_CTX': '131072', 'BONSAI_PARALLEL': '2'})
        self.assertEqual(option(report['arguments'], '--parallel', '-np'), '3')
        self.assertEqual(option(report['arguments'], '-ub'), '1024')
        self.assertEqual(option(report['arguments'], '-c'), '98304')
        self.assertIn('131072', report['arguments'])

    def test_throughput_shares_pool_and_port_is_forwarded(self):
        report, _ = self.run_launcher('-Profile', 'throughput', '--port', '18081')
        self.assertEqual(option(report['arguments'], '--parallel'), '4')
        self.assertEqual(option(report['arguments'], '--port'), '18081')
        self.assertIn('--kv-unified', report['arguments'])

    def test_long_profile_selects_quantized_cache_and_cpu_projector(self):
        report, _ = self.run_launcher('-Profile', 'long')
        self.assertEqual(option(report['arguments'], '-c'), '131072')
        self.assertEqual(option(report['arguments'], '--cache-type-k'), 'q4_0')
        self.assertIn('--no-mmproj-offload', report['arguments'])

    def test_environment_can_disable_profile_quantization_and_projector_offload(self):
        report, _ = self.run_launcher('-Profile', 'long', environment={'BONSAI_KV4': '0', 'BONSAI_MMPROJ_CPU': '0'})
        self.assertNotIn('--cache-type-k', report['arguments'])
        self.assertNotIn('--no-mmproj-offload', report['arguments'])

    def test_explicit_keys_are_redacted_and_invalid_context_fails(self):
        report, output = self.run_launcher('--api-key', 'cli-test-secret', '--api-key=embedded-secret')
        self.assertNotIn('cli-test-secret', output)
        self.assertNotIn('embedded-secret', output)
        self.run_launcher(environment={'BONSAI_CTX': '-1'}, success=False)

    def test_double_click_wrapper_uses_same_configuration_and_passes_arguments(self):
        if os.name != 'nt':
            self.skipTest('cmd.exe is required')
        completed = subprocess.run(['cmd.exe', '/c', str(self.root / 'run_server.bat'), '-DryRun',
                                    '-Profile', 'throughput', '-ub', '1024'], cwd=self.root,
                                   env=self.env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=25)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout[completed.stdout.index('{\n'):])
        self.assertEqual(report['profile'], 'throughput')
        self.assertEqual(option(report['arguments'], '-ub'), '1024')


if __name__ == '__main__':
    unittest.main()
