# GLM/Mia execution-path follow-up

This draft extends `d24bc24955740632dad8ae52986d41c26c55c61b` for the two-GB10,
TP2, GLM EXL3 workload described in issues #2/#3 and PR #1. The reference is Mia
recipe `bf5f94523a8251029cedae86bf61f8ce37fac218`, TensorFold v0.6.0 plus its patches.
The checkpoint revision remains `57edefd2f5d9b371c8345883304d5af68b52fa24`.

The request reports packing about 234.7 s, including 228.1 s affine quantization,
and little TTFT benefit from the previous 127-row experiment. These are supplied
measurements, not measurements of this follow-up. At investigation time the GitHub
connector and public API returned no PR comments/reviews, so the original log
attachments could not be independently inspected. Earlier PR receipts remain
historical. No GPU or model execution was performed for this change.

The implementation contract is explicit: remove redundant work, retain native
ownership and rank agreement, preserve the default path, and expose arithmetic
changes as separate opt-in modes. CUDA compilation and host tests do not establish
GPU equality, model fidelity, or a speedup. The patch remains Draft.

## Implementation and arithmetic policy

All flags below default to `0`. Set each value on every rank. The handshake layout
is now **14**; do not mix old and new binaries. Existing profile format/load choices
remain, and the wrapper does not silently enable these new routes.

| Flag | Runtime change | Arithmetic and limitations |
| --- | --- | --- |
| `TF_GLM_AFFINE_GPU` | Quantize groups/candidates on GPU; tile directly without full affine-weight host readback | Each candidate retains native ordered FP32 scoring, BF16 conversion and first minimum. Intended packed-byte replacement of native scalar, not PyTorch's parallel reduction. Device equality pending. |
| `TF_GLM_AFFINE_XS` | Reuse compatible HC/norm group sums; otherwise compute once per call | Checked whole-row provenance and matching K/stride. Removes output-CTA repetition. SwiGLU sums have another order, so use the native-order fallback. |
| `TF_GLM_AFFINE_SHAPES` | Q4 SHAPE_SK/DEC_TILES policy and corresponding kernel configurations | Changes split/reduction order. Includes shape-only XS scratch and larger partial allocation. Separate numerical qualification required. |
| `TF_GLM_AFFINE_PREFILL` | BF16 reconstructed-weight MMA, continuous FP32 K accumulation, one grid | Distinct prompt arithmetic: BF16 fused reconstruction differs from grouped affine scaling. Includes Mia 0102 tile selection. |
| `TF_GLM_MOE_DIRECT` | Compact routed picks, direct eight-slot output, shared output read by combine; whole prompt expert pass | Intended exact replacement. Removes 9-to-8-to-9 copies, put_slot and 1024-row adapter segmentation. Full-prompt scratch increases; internal unused common-output scratch remains allocated. |
| `TF_GLM_DECODE_GRAPHS` | Cached verify and MTP graph replay using native rank-coordinated graph cache | Uploads, state preparation and hidden copies stay outside capture. Bounded 16-entry cache; drain on eviction/free. See eligibility below. |
| `TF_GLM_HC_MMA` | Prompt HC Tensor Core path, 64-row blocks | BF16 operands, FP32 dot accumulation. Norm square sums retain native order. Numerical mode differs from scalar HC and is not a byte-identical Mia port. |
| `TF_GLM_HC_DEC` | Decode HC with 288 threads, three output partitions and up to four-row weight reuse | Intended byte replacement of native decode ordering. |
| `TF_GLM_KDA_WY` | Prompt-only 32-row blocked delta-rule solve and recurrent-state update | Native prep/out retained; finite FP32 order changes. Nonfinite heads use native serial recurrence. Decode remains serial. |

GPU packing takes precedence over CPU pruning when both are selected; the GPU
packer evaluates its native candidates without the CPU pruning optimization.
The affine shape path can be selected without the standalone XS flag and still
owns sufficient fallback XS scratch. Dedicated prefill supersedes the earlier
127-row packed-window experiment for eligible affine prompt calls.

Graph replay covers ordinary single-sequence verify and internal MTP, including
text generation when vision is enabled. Shared verify windows, paged KV, external
drafters, profiling and windows overlapping image insertion use the existing eager
path. Graph mode and `TF_GLM_PROF=1` are rejected together. Side-stream L2 prefetch
is omitted on graph-eligible calls, including warmup/cache fallback; compare graph
and L2 effects separately. Rank-agreed cache decisions and capture-failure behavior
reuse the existing native graph cache. Host RoCE counters do not count kernels
inside a replay as fresh host launches.

WY uses absolute prompt offsets. Native snapshots/resume already require 2048-token
chunk boundaries, which are multiples of 32; this change does not relax that gate.
Arbitrary mid-block snapshots are not newly supported. Prompt/decode differences,
state continuation, retained prefixes and teacher-forced logits still need device
and model checks. The implementation is a native WY algorithm, not a claim that
its instruction layout or performance equals Mia's kernel.

Mia's compared default keeps `kv_b` in BF16. Specialized FP8 `kv_b` is therefore
not required to match this particular baseline; its absence must not be cited as
the demonstrated cause of this performance gap. FP8 head and FP8 KV cache are
separate features. Broader L2 a/o/MTP prefetch, all-mode graph coverage and a full
123-patch equivalence audit remain outside the qualification established here.

## Required device receipts

Use real rebuilt assets for both ranks; the host semantic build uses unrelated
placeholder images and is not a runnable GPU artifact. Rebuild the original
commands in `glm-mia-fork.md`, plus run `tf-cuda-test glm-affine-quant`, `glm-ref`,
`glm-exl3`, `glm-affine`, and the full kernel-contract runner. The original
`exl3 none` FP32 fusion failure remains independently tracked in issue #2; this
execution work does not claim to fix it. Keep fusion off for the first comparison.

Run the existing profile on the new SHA with all nine flags explicitly zero as
baseline E0. Then enable one route at a time, and finally the compatible combined
profile. Record actual settings and checkpoint/template/binary hashes. Do not
infer a setting was active from its presence in the recipe.

1. GPU packing: compare complete packed words/scales/biases against scalar on
   device fixtures, then real-checkpoint hashes where instrumentation is available.
   Save startup phase times and total readiness latency for both ranks, with timing
   on/off controls. Report memory peaks and errors, not just affine wall time.
2. XS and direct MoE: require byte comparisons against old kernels, including
   nonfinite data, strided rows, tails and guards. Test whole 1024/1862-row MoE
   routing with both rotation layouts. Confirm copy/launch counts in a bounded trace.
3. HC decode: exact old-path bytes. HC MMA, Q4 shapes/reconstructed prefill and WY:
   independent FP64 errors, worst columns/state elements and unchanged nonfinite
   rules; teacher-forced logits/top-token margins, not only sampled output text.
4. Graphs: verify widths, KDA parity, attention bucket/dense-sparse boundaries,
   ring rebasing, multiple sequence lifetimes, cancellation/reset, eviction,
   peer failure and MTP first/subsequent calls. Compare eager/graph token hashes;
   verify captures/replays actually occur. Include graph-off/L2-on and
   graph-on/L2-omitted controls rather than attributing their difference to one knob.
5. WY: validate final recurrent state as well as immediate outputs, 31/32/33-row
   and multi-block seams, zero decay, nonfinite serial fallback and supported
   2048-aligned retained/resumed paths. Same-engine drafted/plain equality remains
   required; it does not replace fidelity checks for the new prompt arithmetic.
6. Served workload: retain #3's actual 73/66/1862-token requests, MTP off/on,
   256 output-token cap, cache zero, at least three alternating repeats per cell.
   Save actual token counts/hashes, TTFT, server decode time, accepted/proposed
   MTP counts, memory, temperature and clocks. Add longer admitted cold prompts
   separately. Retest images, tools, structured output, prompt_logprobs and concurrency.

Exactness failures block enabling the affected replacement. Arithmetic-mode
qualification requires the independent/model evidence above; no tolerance is
widened to obtain a speedup. Compare distributions, retain failed runs, and do not
claim complete Mia parity from compiler or host-only results.

## Host/compiler receipt

All changed CUDA units compile with nvcc 13.0.88 for SM80/86/89/120/121. The
128-row affine prefill kernel has 8-byte spill stores/loads on SM80/86/89 and no
spill on SM120/121; there is no blanket zero-spill claim. Device contracts and
model execution remain pending. Zig 0.17.0 host/semantic validation completed:

```sh
zig build test test-cuda-host test-kernel-contract test-golden fuzz-cortheon \
  compile-cuda-tests compile-modelopt-startup test-glm-prompt-settings-host \
  test-glm-mm-policy-host test-glm-tp3-host \
  -Dfatbins=/workspace/scratch/mia-compile-only-fatbins \
  -Dcontract-controls=/workspace/scratch/mia-compile-only-fatbins/contract_controls.fatbin \
  -j2 --summary all
```

The final run reports **451/451 steps succeeded, 627/629 tests passed, two skipped**;
unchanged targets also reuse earlier cached results. Golden comparison is 26/26
equal. These counts are not a fresh execution count of every cached suite.
The compiled images match **310 symbol/SM resource records**, with zero mismatches
or launch-limit failures. Lean has 428 existing findings on both base and head,
with zero new findings. Initial host mock-fixture failures were repaired before
the final successful run. Device contracts, nvcc 13.3.73 qualification, real packed
checkpoint equality, two-rank graph behavior and model performance were not run.
Scratch logs/assets are local evidence, not deliverable GPU artifacts.
