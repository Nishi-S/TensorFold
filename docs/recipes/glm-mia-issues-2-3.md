# GLM/Mia follow-up: issues 2 and 3

This follow-up uses the reports in [issue #2](https://github.com/Nishi-S/TensorFold/issues/2)
and [issue #3](https://github.com/Nishi-S/TensorFold/issues/3). Their device results
were supplied by the reporter on commit `c2021f408b70a2f51463f1c722056d209c33848e`,
with two GB10s, SM121 and nvcc 13.3.73. They are not device reruns of this follow-up.

## Corrected failure scope

The first failing original comparison is all-four-bit mcg, **FP32 act_mode=1**,
16 rows, unweighted, gate/up-only mask 2, baseline loads, repetition 0.
The preceding GLM BF16 act_mode=0 comparisons passed. Row 10 appears in the first
failing fixture but not in the earlier 1/3/8-row fixtures: this does not establish
a special defect at the 16-row boundary. The printed first four locations and
512 differing words do not establish that every difference is contiguous.

The reporter also observed matching FUSE3/FUSE0 outputs and token hashes for all
18 corresponding GLM requests, nine matching MTP-on/off pairs per native
configuration, and nine functional checks per configuration. This is useful GLM
coverage, not qualification of the failing generic FP32 path or all workloads.

The new targeted test preserves the original random fixture and checks gate/up
partials before down overwrites the shared scratch, followed by FP16 down inputs.
It calls the actual implementation through a compile-time stop-after-gate/up
entrypoint. A partial mismatch points to projection/publication; equal partials
with unequal down inputs point to the epilogue. Exact byte comparisons and the
full-chain regression remain. No production CUDA arithmetic is changed.

Rebuild the real CUDA assets and run on the reporting machine:

```sh
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -j2
zig-out/bin/tf-cuda-test exl3 none
CUDA_LAUNCH_BLOCKING=1 zig-out/bin/tf-cuda-test exl3 none
```

Retain the first differing words and `EXL3 fusion failure:` configuration line.
The `none` argument still skips the external Python oracle. The default wrapper
keeps fusion disabled; explicit masks remain available for controlled comparison.
The numerical root cause remains open pending the intermediate-value result.

## Performance evidence and next measurements

Issue #3 reports about 9–16% lower MTP decode throughput than Mia on its three
workloads; long-input TTFT was 1.771 s versus 1.122 s, while short-input TTFT
improved. FUSE3 did not demonstrate a throughput benefit over FUSE0. Whole-host
used memory improved by about 5.2–5.4 GiB per node; these samples are neither
model-only CUDA allocations nor startup peaks.

The reported full-profile startup took 554.1 s, including about 44–45 s of
checkpoint digest reads. Earlier startup comparisons overlapped compilation.
The report has not isolated the serial MSE quantizer's share of total startup.
Measure clean startup with compilation finished, identical cached checkpoint and
settings, and no competing service. Record both ranks and distinguish checkpoint
loading/digest work from post-load packing.

For bounded prompt/decode attribution, enable the existing `TF_GLM_PROF=1` on
both ranks for a fixed cold 1,862-token request and a separate short-input
256-output-token request. Retain the actual input tokens, profile spans, wall
latency, MTP acceptance and output hashes. Repeat without profiling as a control.
Do not add overlapping/nested spans or PDL kernel durations to claim wall time.
Use the report's prompt-cache-zero controls; reusing an 8K context limit alone
does not reproduce the long-input workload.


## Opt-in implementation changes

`TF_GLM_AFFINE_PRUNE=1` skips the remaining terms of a losing MSE candidate once
its ordered nonnegative FP32 loss prefix reaches the current best loss. Candidate
zero is fully evaluated, strict first-tie selection is retained, and groups above
the conservative finite bound keep the original overflow/error behavior. The
original scalar packer remains the default; this is not a different quantizer.

`TF_GLM_PACKED_PROMPT_BATCH=1` lets split-K affine/FP8 calls batch up to 127 rows
per window, using two BM64 row blocks for a full window. The K slices and ordered
reduction stay unchanged, but tail bucket choices can differ. A full 1,862-row
split call dispatches 30 launches instead of 60; this is a host dispatch count,
not measured latency improvement. Symmetric formats and unsplit large calls keep
their original policies. Both new flags default off and are included in the
rank handshake. Set the same values on every rank.

The scratch planner also accounts for the vocabulary head and shared-expert down
projection. The old shape list omitted the vocabulary head; at high TP its
split-K scratch can exceed the other shapes' maximum. This is a correctness fix,
independent of enabling the batching experiment. The default TP2 allocation does
not increase; high-TP allocations can increase to cover the actual head.

`TF_GLM_PACK_TIMING=1` logs startup-only phase wall times, including readback,
affine/symmetric/FP8 quantization, upload/tiling and compaction. It is disabled by
default. Enabled measurements synchronize BF16-only tiling to attribute that
phase; compare an uninstrumented startup too. Packing totals do not include all
checkpoint reads or digest computation and must not be presented as total startup.

For an explicit experiment, pass these variables before the usual wrapper on
both ranks (keep the existing model, network and serving arguments):

```sh
TF_GLM_AFFINE_PRUNE=1 TF_GLM_PACKED_PROMPT_BATCH=1 TF_GLM_PACK_TIMING=1 \
  tools/glm_mia_profile.sh COMMAND ARGUMENTS
```

Start with each switch separately, then combine them, checking loaded weight
bytes or output hashes and the same cold workload. Fusion remains independently
controlled and defaults off in the wrapper.


## Host-only validation of this follow-up

The affine packer was compared with a frozen copy of the original full scalar
search. Six unit tests cover exact grids, constants/first ties, clipping,
randomized packed-byte equality, BF16 neighbors around the pruning cutoff and
original overflow/error behavior. A synthetic 1,048,576-weight x86_64 ReleaseFast
benchmark alternated old/new order for five repeats; all packed words, scales
and biases matched. Medians were 186.647 ms original and 104.572 ms pruned
(1.785x for this host microbenchmark). This is not an ARM64, checkpoint, whole
startup or GPU speed result. Reproduce on either host architecture with:

```sh
zig run -O ReleaseFast zig/src/families/glm/cuda_affine_pack_bench.zig -lc
```

The launch recorder checks the disabled policy and the enabled 127-row seams,
input/output offsets, split count, kernel bucket, reduction length and scratch
bound. Planner coverage includes every rank at TP1 through TP16, both EXL3 and
non-EXL3 geometry. Default TP2's existing 8,323,072-byte budget is unchanged;
TP16 EXL3 rank zero needs 9,883,648 bytes once the actual head is accounted for.
Device row-invariance fixtures now include 257 rows, but were not executed here.
The remaining device checks and the numerical root cause in #2 remain open.

On the combined follow-up tree based on `93f979d`, Zig 0.17.0 passed the
`compile-cuda-tests compile-modelopt-startup test-glm-prompt-settings-host
test-glm-mm-policy-host test-glm-tp3-host test-kernel-contract` graph:
38/38 steps and 123/123 tests. The compile-only assets contain real unchanged Mia
CUDA images and unrelated/control placeholders, so the resulting runner is not
a runnable GPU qualification artifact. The phase accumulator's standalone host
test also passed. No CUDA source changed in this follow-up.

The complete `zig build test test-cuda-host test-kernel-contract test-golden
fuzz-cortheon -j2 --summary all` graph passed 433/433 steps: 4,715 tests passed,
25 skipped (4,740 total). This includes the actual TP1–16 planner regression.
The command used umask 022 and local socket access for the host server tests.
Its log is `mia-followup-full.log`; counts overlap the focused tests above and
must not be added.

The lean comparison against `93f979d` found 428 inherited findings and zero new
findings (4,397 baseline files, 4,400 current files). Changed Zig formatting and
diff whitespace checks passed. Local logs are in `mia-followup-integrated.log`,
`mia-affine-pruning-test-final.log` and `mia-affine-pruning-bench-final.log` under
the scratch directory; these logs are not included in the PR.
