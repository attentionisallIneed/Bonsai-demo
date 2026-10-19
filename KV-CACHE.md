# 4-bit KV cache (experimental)

⚠️ **Experimental, llama.cpp backend only.** A memory tool, not a speed tool: decode is slightly slower than the default FP16 KV cache.

`BONSAI_KV4=1` stores the KV cache in Q4_0 (4-bit) instead of FP16, cutting KV memory roughly **3.5x**: from 64 KiB per token to about 18 KiB per token on the 27B, so a 100K-token context needs about **1.8 GiB instead of 6.3 GiB**. The 27B's hybrid attention already keeps the cache small, so reach for this only at very long contexts on tight machines.

```bash
BONSAI_KV4=1 ./scripts/start_llama_server.sh
```

Under the hood this passes `--cache-type-k q4_0 --cache-type-v q4_0` (quantized KV requires flash attention, which the scripts already enable).

## Better quality: the mean-centering bias

4-bit quantization of the K cache loses a little accuracy on channels whose activations have a nonzero mean. A small **model-specific calibration bias** fixes most of that at zero decode-time cost (one subtract when the cache is written). Build it once:

```bash
./scripts/make_kv_bias.sh
BONSAI_KV4=1 ./scripts/start_llama_server.sh
```

The script runs `llama-kv-mean-center` (included in the prebuilt binaries) over a calibration text and writes `<Model>-kv-bias.gguf` next to the model weights. The server picks the bias up automatically whenever `BONSAI_KV4=1` is set; without a bias, the 4-bit cache still runs, just with slightly lower quality.

### Windows (PowerShell)

Update the runtime before enabling CUDA quantized KV. The pinned
`prism-b10777-e3de715` includes the [short-query quantized K/V Flash Attention
fix](https://github.com/PrismML-Eng/llama.cpp/pull/355). The updater checks both
release archives against GitHub's published SHA256 digests and stages a complete
runtime before replacing the current one. Stop this repository's running llama
processes first; a failed download or validation keeps the installed runtime.

```powershell
.\scripts\download_cuda_bin.ps1 -CudaTag 13.3
.\scripts\make_kv_bias.ps1
$env:BONSAI_KV4 = "1"
.\scripts\start_llama_server.ps1
```

Use `-CudaTag 12.4` for a driver that does not support CUDA 13.3. Downloads resume
from `.cache/llama-downloads` on retry. The release pin is shared with setup in
`scripts/llama-release.txt`.

The Windows calibration script prefers the same PQ2_0 weights as the launcher,
then PTQ1_0 when PQ2_0 is absent. It respects `BONSAI_FAMILY`, `BONSAI_MODEL` and
`BONSAI_GGUF`, or accepts explicit `-ModelPath` and `-BinaryPath` arguments.
The default output includes the exact model filename, for example
`Ternary-Bonsai-2-27B-PQ2_0-kv-bias.gguf`, plus a `.json` file recording model and
bias SHA256 hashes. An existing bias is replaced only after a successful tool
exit and validation of the GGUF's tensor bounds and rotation metadata.

```powershell
.\scripts\make_kv_bias.ps1 .\my-corpus.txt -Chunks 4
.\scripts\make_kv_bias.ps1 -DryRun
```

The built-in UTF-8 corpus covers English, Chinese, code, mathematics and
structured records, and requires no network. A user corpus must tokenize to at
least `-ContextSize` tokens (512 by default). Calibration explicitly sets
`LLAMA_ATTN_ROT_DISABLE=1` with Q4_0 K/V and Flash Attention; inference must use
that same rotation state, which the server launcher handles when loading a bias.
The script restores the previous environment on success and failure.

Notes:

- **Calibration does not need much data.** The script ships a tiny built-in synthetic corpus; for best results pass your own text file as the first argument, representative of your workload:

  ```bash
  ./scripts/make_kv_bias.sh my_corpus.txt
  ```

- **The bias is model-specific.** Re-run the script after switching `BONSAI_FAMILY` / `BONSAI_MODEL`.
- Calibration and inference must agree on the K-rotation state; the script and server handle the matching flags automatically, and the loader refuses a mismatched bias by design.

Full background and the manual command flow: [PrismML-Eng/llama.cpp#54](https://github.com/PrismML-Eng/llama.cpp/issues/54).
