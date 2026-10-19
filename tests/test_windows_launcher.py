"""Exercise the real PowerShell argument boundary without a model or GPU."""
import base64
import hashlib
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
        # Let each edition discover its own modules. A Python child of pwsh can
        # otherwise pass PS7's Utility module into Windows PowerShell 5.1.
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(('BONSAI_', 'LLAMA_ARG_')) and k.lower() != 'psmodulepath'}

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

    def compile_native_stub(self):
        """A real Windows executable exposes the native argv boundary without a GPU."""
        compiler = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
        if os.name != 'nt' or not compiler.exists():
            self.skipTest('The Windows .NET C# compiler is required for the native argv stub')
        source = self.root / 'argv-stub.cs'
        source.write_text('''using System;
using System.Text;
class ArgvStub {
    static int Main(string[] args) {
        foreach (string arg in args)
            Console.WriteLine("ARG:" + Convert.ToBase64String(Encoding.UTF8.GetBytes(arg)));
        return 0;
    }
}
''')
        binary = self.root / 'bin/cuda/llama-server.exe'
        binary.unlink()
        completed = subprocess.run([str(compiler), '/nologo', '/target:exe', '/out:' + str(binary), str(source)],
                                   capture_output=True, text=True, timeout=25)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return binary

    def write_bias_fixture(self, *, legacy=False, metadata=True):
        model = self.root / 'models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf'
        model.write_bytes(b'model A weights')
        name = 'Bonsai-2-27B-kv-bias.gguf' if legacy else model.stem + '-kv-bias.gguf'
        bias = model.with_name(name)
        bias.write_bytes(b'bias fixture A')
        sidecar = bias.with_name(bias.name + '.json')
        record = {'model_file': model.name, 'model_size': model.stat().st_size,
                  'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
                  'bias_sha256': hashlib.sha256(bias.read_bytes()).hexdigest(), 'k_rot': False}
        if metadata:
            sidecar.write_text(json.dumps(record))
        return model, bias, sidecar, record

    def run_native_launcher(self, *args, shell=None, environment=None, success=True):
        # No HTTP calls, browser, real model, or GPU process are involved.
        def quote(value):
            return "'" + str(value).replace("'", "''") + "'"
        command = ("function Invoke-WebRequest { throw 'HTTP disabled in launcher tests' }; & " +
                   quote(self.root / 'scripts/start_llama_server.ps1') + ' ' +
                   ' '.join(str(value) if value in ('-Profile', '-Port', '-DryRun') else quote(value)
                            for value in args))
        completed = subprocess.run([shell or SHELL, '-NoProfile', '-Command', command], cwd=self.root,
                                   env=dict(self.env, **(environment or {})), capture_output=True,
                                   text=True, encoding='utf-8', errors='replace', timeout=25)
        if not success:
            self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            return completed
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        arguments = [base64.b64decode(line[4:]).decode('utf-8') for line in completed.stdout.splitlines()
                     if line.startswith('ARG:')]
        self.assertTrue(arguments, completed.stdout + completed.stderr)
        self.assertNotIn('-Profile', arguments, 'Launcher options must bind before the native argument boundary')
        return arguments

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

    def test_speculation_keeps_explicit_profile_contexts(self):
        models = self.root / 'models/ternary-gguf/27B'
        models.mkdir(parents=True)
        for name in ('Ternary-Bonsai-27B-PQ2_0.gguf', 'Ternary-Bonsai-27B-dspark-dflash.gguf'):
            (models / name).touch()
        profiles = json.loads((self.root / 'scripts/server-profiles.json').read_text())
        for profile, settings in profiles.items():
            with self.subTest(profile=profile):
                report, _ = self.run_launcher('-Profile', profile,
                                             environment={'BONSAI_FAMILY': 'ternary', 'BONSAI_SPECULATIVE': '1'})
                self.assertEqual(option(report['arguments'], '-c'), str(settings['context']))
                self.assertEqual(option(report['arguments'], '-np', '--parallel'), '1')

    def test_preparation_errors_restore_rotation_in_both_powershell_editions(self):
        (self.root / 'models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0-kv-bias.gguf').touch()
        for shell in dict.fromkeys(filter(None, (shutil.which('powershell'), shutil.which('pwsh')))):
            for previous in ('', 'previous-value'):
                with self.subTest(shell=shell, previous=previous):
                    script = str(self.root / 'scripts/start_llama_server.ps1').replace("'", "''")
                    command = ("try { & '" + script + "' -Profile long -DryRun } "
                               "catch { Write-Host 'EXPECTED_PREPARATION_ERROR' }; "
                               "Write-Host ('RESTORED_ROTATION=' + $env:LLAMA_ATTN_ROT_DISABLE)")
                    completed = subprocess.run([shell, '-NoProfile', '-Command', command], cwd=self.root,
                                               env=dict(self.env, BONSAI_PARALLEL='invalid',
                                                        LLAMA_ATTN_ROT_DISABLE=previous), capture_output=True,
                                               text=True, encoding='utf-8', errors='replace', timeout=25)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    self.assertIn('EXPECTED_PREPARATION_ERROR', completed.stdout)
                    self.assertIn('RESTORED_ROTATION=' + previous + '\n', completed.stdout)

    def test_known_unsafe_cuda_quantized_kv_builds_fail_before_native_launch(self):
        self.compile_native_stub()
        marker = self.root / 'bin/cuda/.llama_release'
        for build, unsafe in ((10709, False), (10710, True), (10772, True), (10775, True), (10776, False)):
            with self.subTest(build=build):
                marker.write_text(f'prism-b{build}-test\n')
                result = self.run_native_launcher('-Profile', 'balanced', environment={'BONSAI_KV4': '1'},
                                                  success=not unsafe)
                if unsafe:
                    self.assertIn('download_cuda_bin.ps1', result.stdout + result.stderr)
                    self.assertNotIn('ARG:', result.stdout)
        marker.write_text('prism-b10772-test\n')
        report, _ = self.run_launcher('-Profile', 'long')
        self.assertEqual(option(report['arguments'], '--cache-type-k'), 'q4_0')
        self.run_native_launcher('-Profile', 'balanced', environment={'BONSAI_KV4': '0'})
        marker.write_text('custom-build-with-unknown-version\n')
        self.run_native_launcher('-Profile', 'balanced', environment={'BONSAI_KV4': '1'})

    def test_cuda_kv_guard_does_not_reject_other_or_unidentified_backends(self):
        binary = self.compile_native_stub()
        (self.root / 'bin/cuda/.llama_release').write_text('prism-b10772-test\n')
        for destination in ('bin/cpu', 'llama.cpp/build/bin/Release'):
            with self.subTest(destination=destination):
                backend = self.root / destination
                backend.mkdir(parents=True)
                replacement = backend / 'llama-server.exe'
                shutil.move(str(binary), str(replacement))
                self.run_native_launcher('-Profile', 'balanced', environment={'BONSAI_KV4': '1'})
                shutil.move(str(replacement), str(binary))
        (self.root / 'bin/cuda/.llama_release').unlink()
        self.run_native_launcher('-Profile', 'balanced', environment={'BONSAI_KV4': '1'})

    def test_bias_identity_hashes_are_verified_before_native_launch(self):
        self.compile_native_stub()
        model, bias, sidecar, record = self.write_bias_fixture()
        for profile in ('long', 'max'):
            with self.subTest(profile=profile):
                arguments = self.run_native_launcher('-Profile', profile)
                self.assertEqual(option(arguments, '--kv-mean-center'), str(bias))
        # A same-name, same-size replacement must fail the full model hash check.
        model.write_bytes(b'model B weights')
        report, output = self.run_launcher('-Profile', 'long')
        self.assertEqual(option(report['arguments'], '--kv-mean-center'), str(bias))
        self.assertIn('SHA256 verification is deferred', output)
        failed = self.run_native_launcher('-Profile', 'long', success=False)
        self.assertIn('make_kv_bias.ps1', failed.stdout + failed.stderr)
        self.assertIn('SHA256', failed.stdout + failed.stderr)
        self.assertNotIn('ARG:', failed.stdout)
        model.write_bytes(b'model A weights')
        bias.write_bytes(b'bias fixture B')
        failed = self.run_native_launcher('-Profile', 'long', success=False)
        self.assertIn('SHA256', failed.stdout + failed.stderr)
        self.assertNotIn('ARG:', failed.stdout)

    def test_dry_run_rejects_mismatched_or_malformed_bias_metadata(self):
        model, bias, sidecar, record = self.write_bias_fixture()
        invalid = ({'model_file': 'other-model.gguf'}, {'model_size': model.stat().st_size + 1},
                   {'model_sha256': 'invalid'}, {'bias_sha256': ['0' * 64]},
                   {'k_rot': True}, {'k_rot': 'false'})
        for changes in invalid:
            with self.subTest(changes=changes):
                sidecar.write_text(json.dumps(dict(record, **changes)))
                failed = self.run_launcher('-Profile', 'long', success=False)
                self.assertIn('make_kv_bias.ps1', failed.stdout + failed.stderr)
        sidecar.write_text('{broken JSON')
        failed = self.run_launcher('-Profile', 'long', success=False)
        self.assertIn('Cannot read KV bias metadata', failed.stdout + failed.stderr)

    def test_legacy_named_bias_requires_matching_sidecar_and_hashes(self):
        self.compile_native_stub()
        model, bias, sidecar, record = self.write_bias_fixture(legacy=True)
        arguments = self.run_native_launcher('-Profile', 'long')
        self.assertEqual(option(arguments, '--kv-mean-center'), str(bias))
        sidecar.write_text(json.dumps(dict(record, model_file='another-weight-pack.gguf')))
        failed = self.run_launcher('-Profile', 'auto', environment={'BONSAI_KV4': '1'}, success=False)
        self.assertIn('make_kv_bias.ps1', failed.stdout + failed.stderr)
        sidecar.unlink()
        report, output = self.run_launcher('-Profile', 'auto', environment={'BONSAI_KV4': '1'})
        self.assertNotIn('--kv-mean-center', report['arguments'])
        self.assertIn('Ignoring legacy KV bias without model metadata', output)

    def test_unidentified_exact_bias_warns_but_long_profiles_require_metadata(self):
        self.compile_native_stub()
        model, bias, sidecar, record = self.write_bias_fixture(metadata=False)
        report, output = self.run_launcher('-Profile', 'auto', environment={'BONSAI_KV4': '1'})
        self.assertEqual(option(report['arguments'], '--kv-mean-center'), str(bias))
        self.assertIn('hashes have not been verified', output)
        arguments = self.run_native_launcher('-Profile', 'auto', environment={'BONSAI_KV4': '1'})
        self.assertEqual(option(arguments, '--kv-mean-center'), str(bias))
        for profile in ('long', 'max'):
            with self.subTest(profile=profile):
                failed = self.run_native_launcher('-Profile', profile, success=False)
                self.assertIn('requires model identity metadata', failed.stdout + failed.stderr)
                self.assertNotIn('ARG:', failed.stdout)

    def test_template_json_crosses_the_real_native_boundary_in_both_editions(self):
        self.compile_native_stub()
        value = json.dumps({'reasoning_effort': 'high', 'message': 'say "hi" at C:\\models\\bonsai'})
        for shell in dict.fromkeys(filter(None, (shutil.which('powershell'), shutil.which('pwsh')))):
            with self.subTest(shell=shell):
                arguments = self.run_native_launcher('-Profile', 'balanced', '--chat-template-kwargs', value, shell=shell)
                self.assertEqual(json.loads(option(arguments, '--chat-template-kwargs')), json.loads(value))
        desktop = shutil.which('powershell')
        if desktop:
            # Preserve callers who already use legacy escaped JSON on Windows PowerShell.
            escaped = '{\\"reasoning_effort\\":\\"high\\"}'
            arguments = self.run_native_launcher('-Profile', 'balanced', '--chat-template-kwargs', escaped, shell=desktop)
            self.assertEqual(json.loads(option(arguments, '--chat-template-kwargs')), {'reasoning_effort': 'high'})

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
