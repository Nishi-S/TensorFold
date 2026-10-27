# Independent GLM/Mia port experiments

This local fork starts at TensorFold `8c36359d533debdbb8f53a4229b0bb9362c9c783`.
It targets two GB10s, one request, MTP, vision and FP8 KV with
`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit` revision
`57edefd2f5d9b371c8345883304d5af68b52fa24`.
The reference recipe is Mia revision `bf5f94523a8251029cedae86bf61f8ce37fac218`.
These experiments are unqualified until the device and model checks below run.
Current host/compiler evidence is in [the catch-up receipt](glm-mia-catchup-validation.md).

## Implemented catch-up paths

| Native setting | Implementation | Mia source |
| --- | --- | --- |
| `TF_GLM_DENSE_FORMAT=affine4`, `TF_GLM_DENSE_BITS=4` | Group-64 affine Q4, BF16 scale/bias, 16-candidate MSE clipping search; load, tile, compact and runtime dispatch | 0005 |
| `TF_GLM_HEAD_FORMAT=fp8`, `TF_GLM_HEAD_BITS=8` | Weight-only E4M3FN head, FP32 absmax/448 scales per 128 values, BF16 activations, FP32 64-value product groups | 0002/0005 |
| `TF_EXL3_DECODE_LOADS=nc1`, `nc2`, `nc4` | 16-byte noncoherent trellis loads and 1/2/4-step register rings | 0047 |
| `TF_EXL3_DECODE_FUSE=1`, `2`, `3` | Down fusion, gate/up fusion, or both, preserving ordered split sums | 0016 |
| `TF_GLM_L2PF=1`/`bulk`, `lines`, `touch` | Bounded side-stream weight prefetch before FFN gather | 0046 |
| `TF_GLM_Q4_PAIR_DECODE=1` | Optional packed BF16 conversion for native symmetric Q4 only | Native experiment |

All paths are off by default; the original symmetric formats remain the default.
Use the same flags on every rank. The rank handshake checks formats, load mode,
fusion mask and L2 mode/budgets before allocating weights. This fork's handshake
layout is 13: do not mix its ranks with older binaries.

`TF_EXL3_DECODE_LOADS=baseline` selects the original loads; `scalar` retains the
first-stage one-step scalar prefetch. The legacy `TF_EXL3_DECODE_PREFETCH=1`
alias selects scalar when no new load selector is present. Explicit load mode
wins. The new vector and fusion paths are restricted to loader-verified mcg
four-bit experts and 1–16 rows; mixed widths and larger calls retain the native
path. Down fusion applies to per-slot output with one K split; weighted-combine
calls retain their original down epilogue. Gate/up uses scratch-owned counters
cleared on the consuming stream, fenced partial writes and ordered split reads.

L2 prefetch currently covers the **f site only**, warming the next layer input
or final head prefix. It does not yet port Mia's a/o sites, shared-expert reorder
or MTP-specific prefetch. `1` means bulk, as in Mia; on pre-SM90 chips bulk falls
back to line hints. Bulk chunks and touch loads stay within allocation bounds.
`TF_GLM_L2PF_MB` and `_ROWS` accept integers 1–64 (defaults 8 MiB and 64 rows).
Prompts and graph capture skip side-stream prefetch. Events join outstanding
reads to the main stream, and errors/teardown drain before freeing weights.

The affine quantizer uses deterministic scalar FP32 error accumulation; near
MSE ties it may select a different range than PyTorch's parallel reduction.
The CUDA affine formula is `qdot*scale + sum(x)*bias` in grouped FP32 arithmetic;
nonfinite classes follow that factored formula, not algebraic reassociation of
infinities. This is not a claim of byte-identical whole-model output to Mia.
FP8 keeps Mia's per-128 scale storage but scales each 64-value product group.
Both new formats retain shape-selected K slices across prompt/decode. Matrices
needing split K use at most 64-row windows and ordered reduction; unsplit large
calls use 128-row blocks. Mia's single-kernel sequential prompt folding is not
ported, so this correctness-first prompt implementation may have extra launches.

Native already compresses dense weights after loading and uses partials plus an
ordered reduction instead of thread-block clusters. Mia 0103's no-cluster policy
is therefore already represented architecturally; its speedup is not claimed.
This patch covers the priority paths above, not all 123 Python recipe patches.
The specialized FP8 `kv_b` path and per-shape dense launch tuning remain outside
this patch; they must not be inferred from the FP8 head setting.

Issue [#2](https://github.com/Nishi-S/TensorFold/issues/2) reports a reproducible
SM121 fusion byte-equality failure with nvcc 13.3.73. The wrapper now defaults
`TF_EXL3_DECODE_FUSE=0`; fusion requires an explicit override until requalified.
This mitigates the reported path; it does not establish its numerical root cause
or whole-model correctness. The standalone fusion tests still exercise masks
1/2/3 with exact comparisons and report the failing configuration and stage.

To explicitly select the combined profile (with fusion disabled), on each rank run your native command
through the wrapper (it only sets environment variables, then executes it):

```sh
TF_GLM_RANK=1 TF_GLM_WORLD=2 NCCL_COMM_ID=192.0.2.1:29583 \
  tools/glm_mia_profile.sh zig-out/native/bin/tensorfold-native serve MODEL_DIR \
  --context 8192 --parallel 1 --kv-cache fp8 --vision --name local-model
```

Use rank 0 on the other host, your actual model path and NCCL link address, and
retain your existing network settings. Individual variables set before the
wrapper override its defaults. This profile selects unqualified implementations;
GPU validation is deferred by request and is not a prerequisite for creating the
patch. Quantization is performed on the host at load time, so the MSE search adds
CPU work and host scratch memory relative to native absmax packing.

## Build and device checks

Use `.zig-version` (0.17.0) and a CUDA compiler supporting the target SM.
Follow [the CUDA recipe](glm-5.3-flash-cuda.md) for rank placement and networking.
On the test hosts, compile both ranks from the same source and CUDA assets:

```sh
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
zig-out/bin/tf-cuda-test exl3 none
zig-out/bin/tf-cuda-test glm-mm-fixed
zig-out/bin/tf-cuda-test glm-q4-pair
zig-out/bin/tf-cuda-test glm-affine
zig-out/bin/tf-cuda-test glm-fp8
zig-out/bin/tf-cuda-test glm-l2-prefetch
zig build kernel-contract -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
```

`exl3 none` deliberately skips the external Python oracle receipt; record that
skip. Full qualification additionally requires the reference fixture directory,
the new replacement contracts, all affected compile limits, nonfinite/guard
checks and real model output checks. Compile success alone does not prove them.
The existing `glm-mm-bench` compares BF16 products, not these Q4 alternatives.

## Model receipt at 8K

Compare original native, affine Q4 alone, FP8 head alone, vector loads alone,
fusion alone, L2 alone, and the combined profile. Keep quantization fixed when
attributing a timing change to load scheduling or fusion.
Keep the build, checkpoint, chat template, rank placement, cache mode, sampling
and output length fixed. Use `--context 8192 --parallel 1 --kv-cache fp8 --vision`
and `--prompt-cache-gib 0` for cold-cache controls. `TF_KV_PAGES=0` explicitly
keeps FP8 KV on whole windows; native refuses FP8 with explicit paged KV.
Keep MTP enabled and check startup logs for the actual loaded configuration.

For each configuration, restart both ranks and record exact launch arguments,
environment settings, build identity, driver/CUDA versions and checkpoint
revision. Run the existing clients against rank 0 after model loading:

```sh
python3 tools/bench_concurrent.py http://127.0.0.1:8080 local-model \
  --levels 1 --alone --serial --tokens 256 --reps 3 --temperatures 0 \
  --label baseline --output baseline-equality.json
python3 tools/bench_openai.py http://127.0.0.1:8080 local-model \
  --tokens 256 --reps 3 --temperatures 0 \
  --label baseline --output baseline-decode.json
```

Replace labels and output paths for each candidate. Preserve every attempt.
The equality client checks same-engine drafted/plain token hashes; compare full
`token_sha256` across configurations as well. Missing hashes, incomplete streams
or failed comparisons are not passes. Inspect token differences before rates.
Alternate baseline/candidate runs; report wall-time decode and TTFT separately.
Do not add PDL kernel durations: overlapping execution and dependency waits can
make that sum misleading. Retain MTP acceptance and peak memory where available.

An 8K context setting is not an 8K prompt workload. For cold prefill, retain the
actual token arrays and request bodies at each measured length. Check available
prompt budget including template, image tokens and reply before sending them.
After 8K passes, expand to longer admitted windows in stages.

Also exercise image input, tool invocation and tool-result continuation,
thinking-off effort headers, `reasoning_effort=high`, resume, cancellation and
reset. Use the standard `--chat-template` override where needed; do not infer
these feature checks passed from a text-only benchmark.

## Attribution and arithmetic limits

The native implementation and history retain TensorFold's license and notices.
Mia 0016/0047 motivate the EXL3 scheduling experiment; the implementation uses
native trellis geometry and the existing shared decoder. Mia 0047 credits Jay
Leaton's Apache-2.0 work, and 0103 credits BadAd84. Any future imported source
must retain its applicable attribution and notices.

The host-only Q4 packing proof can run without CUDA:

```sh
c++ -std=c++17 -O2 tools/zig/cuda_host/glm_q4_pair.cpp -o /tmp/glm-q4-pair-host
/tmp/glm-q4-pair-host
```

This exhausts nibble pairs and their packed positions. It does not execute the
BF16 CUDA instruction or prove whole-GEMM equivalence on a GPU.

The affine/FP8 changes deliberately introduce different weight approximations.
Retain teacher-forced histories, top-token margins and independent FP64 checks
when device testing resumes. Do not report a changed quantizer or output token
sequence as an isolated kernel speedup. No device or served performance result
is claimed by the catch-up patch.
