# GLM/Mia catch-up patch validation, 2026-10-27

This receipt covers the expanded independent fork on TensorFold
`8c36359d533debdbb8f53a4229b0bb9362c9c783`, branch `work/glm-mia-port`.
The Mia reference is `bf5f94523a8251029cedae86bf61f8ce37fac218`.
See [the implementation and remaining differences](glm-mia-fork.md).
The [first-stage receipt](glm-mia-validation.md) is historical and does not
qualify these later changes.

The aggregate artifact is `glm-mia-catchup.patch` with a
separate `.sha256` file. It includes the first-stage changes: apply it to a clean
checkout of the base above, not on top of `glm-mia-stage1.patch`.

```sh
git apply --check /path/to/glm-mia-catchup.patch
git apply /path/to/glm-mia-catchup.patch
```

The artifact's applicability is checked against the base using a temporary Git
index, without altering the working tree or the user's index contents.

PR integration: rebased onto `b30d5b9d36786ff778b4bb7a7e1c4b5d26de126e`.
The original counts below describe the pre-rebase snapshot; they are not a claim
that the entire updated main branch was requalified.

## Implemented scope

- Host affine Q4 MSE packing, BF16 scale/bias, CUDA tiling and dense dispatch.
- E4M3FN head packing, FP32 scales per 128 weights and scaled FP32 product groups
  of 64 values; explicit storage format independent of symmetric int8.
- EXL3 scalar and 16-byte nc1/nc2/nc4 prefetch, down and gate/up epilogue fusion.
- Allocation-bounded L2 bulk/lines/touch prefetch on a joined side stream at the
  f site; prompt/capture exclusion and teardown drain.
- Explicit rank-agreed settings, format-aware weight compaction and an opt-in
  launch wrapper at `tools/glm_mia_profile.sh`.

All new paths remain opt-in. This is a priority-path port, not a port of all 123
Mia patches. The specialized FP8 kv_b path, per-shape launch tuning, L2 a/o/MTP
sites and shared-expert reorder remain outside this patch.

## Compiler and host evidence

Zig 0.17.0 and nvcc 13.0.88 are from SHA256-checked official distributions.
The changed `glm_mm`, `exl3_experts` and `glm_l2_prefetch` translation units
compile for SM80, SM86, SM89, SM120 and SM121. Actual normalized launch records
are checked into `zig/src/core/contract/`; later compiler versions remain
pending. Register counts alone do not establish speed.

The final host cubin-parser audit checks **265 symbol/SM records**, including all
registered legacy symbols in the three changed images: zero mismatches and zero
launch-limit failures. All three compilation logs have no nonzero spill reports.
Recursive CUDA include hashes match their compilation snapshots. The final EXL3
macro spelling change produces identical SASS to the prior expanded version on
all five SMs. Evidence is retained in `mia-final-compile-evidence.json`,
`mia-final-source-matches.json` and `mia-all-resource-check-final.log`.

The integrated CUDA-runner, GLM startup and dispatch/settings semantic build
passed 17/17 steps and 12/12 tests in Debug mode. It used real fatbins for the
three changed translation units and empty unrelated/control placeholders to
force semantic and link analysis. That executable is not a runnable device
qualification artifact. The final current-tree rerun also passed 17/17 steps,
reusing those successful host tests from cache; its log is
`mia-semantic-current.log`. Commands and exit codes for all
final checks are retained in `mia-final-checks-rerun.json`.

Host-only checks include affine and FP8 quantization, all 256 FP8 decoder codes,
4,096 symmetric Q4 pair placements, 66,560 bounded L2 range cases, format dispatch,
EXL3 mode/fusion policy and rank agreement. These do not execute GPU instructions.

The final combined host suite exited 0: **425/425 build steps succeeded**.
Its summary reports **259/260 tests passed, 1 skipped**, with other successful
targets reused from cache. This is the invocation's reported count, not a claim
that only 260 tests exist. The earlier expanded run reported 3022/3033 tests
passed, 10 skipped and the one source-token failure corrected below. Counts from
these runs must not be added: their cached targets overlap.

Some image checks report absent libjpeg-turbo 3.1.4.1 and libwebp 1.6.0. Not every
skip is named in the logs; skips remain skips, not passes.
The final log is `mia-full-suite-final.log`.

Plain optimized `zig build -j2 --summary all` exited 0, **25/25 steps succeeded**;
log: `mia-plain-build-final.log`. This host build does not
contain a complete runnable CUDA asset set. It warns that `libtfgrammar.so` is
absent; required tool calls need the matching release library or the documented
grammar build before later serving checks.

The full host command is:

```sh
umask 022
zig build \
  test test-cuda-host test-kernel-contract test-golden fuzz-cortheon \
  -j2 --summary all
```

Local socket access is enabled for network tests. The command-local umask
preserves the permission bits expected by the unchanged request-log tests.
The first expanded-suite attempt failed one source-token check: fused EXL3
exports used token concatenation, so the source checker could not find their
complete names. The macro now takes the complete export name; its expansion and
kernel semantics are unchanged. The checker and its expectations were preserved.
The failed log is retained as `mia-full-suite.log`.

Lean comparison: 428 inherited findings, 428 in the candidate, zero new.
Changed source modules remain at most 600 lines; Zig formatting and
`git diff --check` pass. The shell profile passes syntax checking and checks of
its default settings and explicit per-variable overrides.

## Deferred device evidence

GPU execution and model validation are deferred at the user's request. The new
standalone tests and registered operator contracts are prepared for independent
FP64 references, nonfinite behavior, input/output guards, repeated calls,
row/prompt invariance, split-K ordering, fusion-counter reuse and L2 lifecycle.
None of those GPU gates is recorded as passed.

The affine CUDA reference is the explicitly factored grouped FP32 formula;
nonfinite behavior must be evaluated against that formula. The scalar CPU MSE
search can choose a different range from PyTorch near reduction ties. Affine Q4
and FP8 introduce different weight approximations, so no whole-model token
identity with Mia is claimed.

The default EXL3 symbol's SM121 SASS remains identical to the base. The expanded
`glm_mm` translation unit changes scheduling/addressing in the default symmetric
Q4 symbol, despite an unchanged instruction count. The first-stage identical-SASS
observation for that symbol must not be carried forward to this patch.

No two-GB10 serving, checkpoint output equality, MTP acceptance, vision, FP8 KV
quality, throughput or latency result is claimed. The recipe contains the later
8K-first comparison procedure and device commands.
