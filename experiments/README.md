# Optimization variants

Kept as a record of what was tried so the same ground is not re-covered. **None of these is the
shipped miner** — that is `../qpow_hip.cpp`. They are single-file benchmark kernels without the
pool client. Build them **from the repository root** with `-I.`:

    hipcc -I. --offload-arch=gfx1100 -O3 -mcode-object-version=4 experiments/rFAST.cpp -o /tmp/v

`base.cpp` is the control that the others are diffed against.

| family | files | size vs base |
| --- | --- | --- |
| `r*` — reduction / carry-chain rewrites | `rA rB rC rD rE rADE rALL rFAST rFAST2` | 24-43 changed lines |
| `v*` — small single-site variations | `vA vB vC vR` | 2-22 changed lines |
| `s*` — sparse first-round forms | `sB sC sBC s1 s2` | 33-187 changed lines |
| `p*`, `nc` — permutation restructuring / no-carry forms | `pA pB pC nc` | 172-174 changed lines |

Most carry no annotation beyond the diff; they were cut during a single optimization session and
only the conclusions below were written down. Treat an unannotated variant as "tried, did not
ship" and re-measure before believing anything about it.

## The one unfinished thread: the B/C divergence

`sBC` measured **152.1 MH/s against a 136.0 MH/s control — about +12 %** — but computes the wrong
hash. The failure is in the `B`/`C` sub-expressions of the linear layer. Four hypotheses have been
falsified:

1. **Bad hardware or driver** — the failure is bit-exact reproducible (20 mismatches, 5 of 5 runs),
   and the *same* kernel computes the `base` form correctly and the `u128` form incorrectly.
   Hardware faults do not do that.
2. **Compiler version** — identical divergence under clang 17 and clang 19.
3. **Multi-term `__uint128_t` expressions** — `tests/minimal.cpp` passes GPU==CPU on 2/3/4-term and
   compound forms.
4. **Single-expression vs compound `+=` form** — rewriting B and C as `acc += ...` (the shape that
   works in variant A) still fails: `sB`, `sC` and `sBC` are all 0/200.

It is real and reproducible via `tests/gputest.cpp`, but the trigger is unidentified. The next step
is **instrumentation, not another hypothesis**: dump the GPU's actual `sum[]`/`o[]` values for one
input and find the first lane that differs.

Worth +12 % if anyone solves it.
