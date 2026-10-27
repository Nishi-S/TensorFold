# GLM/Mia first-stage validation, 2026-10-27

Historical first-stage receipt, before the subsequent format/fusion/L2 catch-up patch.
It does not qualify the current expanded implementation.

Scope: the opt-in EXL3 one-step prefetch and symmetric Q4 pair-conversion
experiments described in [the fork recipe](glm-mia-fork.md). Base commit:
`8c36359d533debdbb8f53a4229b0bb9362c9c783`. These are local source changes.

The validation host has no NVIDIA GPU or target checkpoint. No device contract,
MTP/plain equality, vision, TP2 serving, FP8 KV quality or performance pass is
claimed. Affine MSE Q4 and FP8 head were not implemented at this first stage;
the later catch-up patch adds them, with its own separate receipt.

## Compiler observations

Zig 0.17.0 and NVIDIA nvcc 13.0.88 were obtained from official distributions;
archive SHA256 values were checked. Both changed CUDA translation units compile
for sm80, sm86, sm89, sm120 and sm121 without register spills. Launch records use
the repository cubin parser, which removes the driver-reserved 1 KiB shared
memory from sm120/121 ELF totals.

| Candidate | SM80/86/89 registers per thread | SM120/121 registers per thread | Static shared bytes | Dynamic shared bytes |
| --- | --- | --- | --- | --- |
| `tf_exl3_grouped_1_pf1` | 164 | 127 | 32832 | 0 |
| `glm_mm4_pair_16` | 92 | 49 | 0 | 12672 |

Both launches use 128 threads. The Q4 kernel declares a 128-thread launch bound;
the EXL3 kernel has no explicit bound. Later compiler versions require their own
receipts, recorded as pending rather than reusing 13.0 evidence.

On sm121, baseline and candidate builds have identical SASS for the original
`tf_exl3_grouped_1` and `glm_mm4_16` symbols. The new Q4 symbol changes 32
integer-to-float and 16 BF16 packing instructions into 16 packed BF16 additions;
both keep eight HMMA and eight FFMA instructions in the inspected static listing.
Static instruction count is 657 versus 609; registers are 50 versus 49. On the
older SMs, registers increase from 80 to 92. None of these counts establishes
runtime speed or device output equality.

## Checks

- Host Q4 packing proof: 4096 placements passed. Actual CUDA BF16 instructions
  and GEMM arithmetic still require device execution.
- Focused GLM matmul dispatch host tests: 4/4 passed.
- EXL3 policy test: 1/1 passed.
- Lean checker: 428 findings on the baseline, 428 on the candidate, zero new.
- `git diff --check`: passed.
- CUDA contract-runner semantic/link build: 8/8 steps passed. This used real
  candidate `exl3_experts` and `glm_mm` fatbins, with empty unrelated/control
  placeholders to force analysis of the new paths without building every
  unrelated kernel. This executable is not a runnable qualification artifact.
- Plain `zig build -j2 --summary all`: 25/25 steps passed. This host build has no
  complete CUDA asset set and cannot qualify GPU inference. It warns that
  `libtfgrammar.so` is absent; restore the matching release library or follow
  the repository grammar build instructions before testing required tool calls.
- Combined host suite (`test test-cuda-host test-kernel-contract test-golden
  fuzz-cortheon compile-cuda-tests`, `-j2 --summary all`, with local socket
  access): exit 1, 431/434 build steps succeeded, 3210/3219 tests passed,
  8 skipped and 1 failed. Golden, fuzz, kernel-contract host and CUDA-runner
  compile targets succeeded. This is not recorded as an all-green suite.
- The sole test failure was `request_log`'s publicly readable-file refusal
  case: inherited `umask=0077` turned its requested mode 0644 into 0600, so the
  logger correctly accepted it. Retrying only that unchanged test with
  command-local `umask 022` passed 1/1 (exit 0). No production code or test
  expectation was relaxed. The original suite was not rerun in full.

The log explicitly reports missing libjpeg-turbo 3.1.4.1 and libwebp 1.6.0 for
some skipped image cases. It does not identify every skipped case by name, so
all eight skips are retained without assigning them all to those codecs.

The compile-only command was `zig build compile-cuda-tests -Dfatbins=DIR
-Dcontract-controls=DIR/contract_controls.fatbin -j2 --summary all`, with DIR
holding the explicitly described compile-only inputs. For an executable device
test, use the full `-Dnvcc` build in the fork recipe instead.

The initial concurrent edit/build iteration found a Zig 0.17 array-repetition
syntax error in the new contract fixture; it was corrected to `@splat`.
Review also corrected the EXL3 nonfinite fixture to retain finite control rows.
Initial socket tests encountered sandbox restrictions; the completed combined
run above allowed local sockets. Interrupted preliminary builds are not passes.

## Device coverage prepared, not executed

The CUDA contract runner registers both candidates against their original
symbols, with independent FP64 references, nonfinite inputs, input/output guards,
five-repeat checks, shape edges and accumulation stress. Planted controls omit
the final input/tile. The standalone EXL3 check covers per-slot and reduced
outputs, mixed widths, and the 17-row fallback. `glm-q4-pair` covers strided and
offset inputs, output guards, BF16/FP32 outputs, split K, PDL on/off, five repeats,
and row counts crossing the 16-row dispatch boundary.

Actual device `SUMMARY` lines, full token hashes, served timings and memory
receipts remain required before enabling these settings by default. Use the
current configuration matrix and 8K-first procedure in the fork recipe.
