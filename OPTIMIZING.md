# Optimizing qpow-hip

Everything measured while getting this kernel from 96 MH/s to 176.9 MH/s on an RX 7900 XTX, written
for whoever picks the work up next. It is organised around one question: **what is still worth
trying, and what is already known not to be?**

Read [experiments/README.md](experiments/README.md) too — it covers the one unresolved anomaly.

## Where it stands

| | |
| --- | ---: |
| Shipped | **176.9 MH/s** (bench), 175.7 live on the pool |
| Baseline both other miners hit on this card | 96 MH/s |
| Gain so far | **+84 %** |
| VALU ops/hash | ~23,100 |
| of which conditionals | **6,284** (`v_cndmask` 2,707 + `v_cmp_lt_u64` 2,157 + the rest) |
| `v_mad_u64_u32` | 2,582 |
| VGPRs / occupancy / spill | 102 / 12 waves per SIMD / 8 |

An independent design study predicted ~218 MH/s for a fully tuned kernel, so there is plausibly
another ~20 % available. For context, an RTX 5060 does 265 MH/s — RDNA3 emulates the 64-bit integer
multiplies this algorithm is built on, and no amount of tuning closes that gap.

## How to measure anything

Never trust an instruction count or an occupancy number on its own — on this kernel both have
pointed the wrong way (see "Occupancy is not the metric"). Judge on an alternated bench: run control
and variant several times each, interleaved, and compare medians. Differences under ~1 % are noise.

    ./qpow_hip benchsparse 30            # hashrate
    ./qpow_hip verify < big_vectors.txt  # 8000/8000 - MUST pass
    ./qpow_hip sparsecheck               # 1,048,576 nonces, sparse path == straightforward path
    ./qpow_hip sharecheck 262144         # submitted nonces re-hash correctly

## THE correctness trap — read before you optimise anything

A "lazy" `gf_add` that folds the carry but skips canonicalisation:

    u64 s = a + b; if (s < a) s -= P; return s;   // WRONG

runs at **190 MH/s and passes all 8,000 known-answer vectors** — and is silently incorrect. When an
input is non-canonical (>= P), the fold `s -= P` (which is `+ (2^32-1)` mod 2^64) can itself
overflow, dropping a second fold. `tests/diff.cpp` demonstrates it with x = {2^64-1, 1, 2, 3}:
results differ by exactly 2^32-1. **The KAT passed by luck of the data.**

For a miner a wrong hash is a rejected share, so this cost a whole session and a tempting +40 %.
The lesson generalises: **validate a field-arithmetic change against exact `% P` arithmetic over
adversarial inputs BEFORE you benchmark it.** `tests/fieldtest.cpp`, `tests/ft2.cpp` and
`tests/ft3.cpp` are the harnesses for that; the KAT alone is not sufficient.

### Why the shipped non-canonical reduce is nevertheless safe

`gf_red` deliberately drops its final canonicalising `if (r >= P) r -= P`, so its output is congruent
mod P but **not** canonical. That was worth +20.6 % and it is sound, for a reason the lazy add did
not have: every consumer either accumulates in `__uint128_t` (exact for any u64 input), or performs
exactly one conditional subtract — and `2^64 - P = 2^32 - 1 < P`, so one subtract always lands in
range. The lazy add failed because its *fold* could itself overflow; here the folds are untouched and
only the cosmetic final subtract is gone. Validated over 690 adversarial cases in `tests/ft2.cpp`
(including products of `0xFFFF...FFFF` and 12-way accumulator shapes) before it was ever benchmarked.

## What is left, in priority order

1. **The conditionals — the biggest single block.** 6,284 of ~23,100 VALU ops. After the work below,
   what remains is *not* the borrow branch (the compiler already proves that dead) but the carry fold
   `if (r < lo) r += EPS`, one per reduce, plus `gf_sub` and the final compare. Removing those needs
   a genuinely different representation — carry-save or redundant limbs. That approach lost on
   NVIDIA and would need its own correctness campaign here, but it is where the mass is.
2. **Explicit `v_mad_u64_u32` inline asm.** clang already emits 2,582 of them; hand-written asm may
   fold more adds into multiply accumulators. This is the RDNA3 analogue of the IMAD.WIDE
   free-addend trick that was worth +5.6 % on the NVIDIA kernel. clang 17 has no builtin, so it needs
   inline asm (rocRAND is the precedent).
3. **The B/C divergence — about +12 %, four hypotheses already falsified.** See
   [experiments/README.md](experiments/README.md). The next step there is instrumentation, not a
   fifth hypothesis.
4. **More nonces per thread, for ILP.** Untried. Occupancy is maxed, so ILP is the remaining latency
   lever.

## Do not retry these — all measured, all rejected

| attempt | result | why |
| --- | --- | --- |
| Lazy `gf_add` (skip canonicalisation) | 190 MH/s but **WRONG** | see above; latent, and the KAT does not catch it |
| `gf_red_s` specialisation (drop the provably-dead borrow branch at 4 sites) | **exact no-op** | conditionals 6,284 -> 6,284, VALU unchanged, bench unchanged. clang already eliminates it after inlining once `hh` is known zero. Bound validated first: `tests/ft3.cpp`, 915 cases, worst site hi = 7 against a 2^32 limit |
| One-reduce column reconstruction | **0.8 % slower** | gates clean, but it replaces the shift/add chains used for the small constants (1,2,3,4,6) with real 64x64 multiplies. The multiplies cost more than the ~7 reduces per lane they save |
| Fixing register pressure (3 variants) | **all lost** | see below |
| wave64 | **closed** | unsupported on gfx1100 for HIP |
| Building against the ROCm 6.4 runtime | **closed** | correct but 24 % slower. Same source, same codegen — it is the runtime |
| Unrolling the 4+4 external rounds | **closed** | makes all 96 round constants live at once: 208 SGPRs spilled. `#pragma unroll 1` on those loops is why they are rolled |

### Occupancy is not the metric

Three fixes for the sparse kernel's register pressure. Every one improved the resource numbers and
every one was slower:

| variant | VGPRs | occupancy | spill | MH/s |
| --- | ---: | ---: | ---: | ---: |
| **control (shipped)** | 102 | 12 | 8 | **147.5** |
| `__noinline__` precompute | 103 | 12 | 8 | 146.6 |
| rolled reconstruction loop | **74** | **16** | **0** | 141.5 |
| both | 102 | 12 | **0** | 141.5 |

The unrolled form wins on ILP and that outweighs occupancy and spills. The NVIDIA campaign on this
same algorithm reached the identical conclusion independently. **Chase instruction count, not
occupancy.**

## The ladder that got here

| build | MH/s | KAT | note |
| --- | ---: | --- | --- |
| Vulkan baseline (both existing miners) | 96 | — | neither ships a native AMD kernel |
| HIP naive `__uint128_t` | 144.2 | 8000/8000 | first correct build |
| + rolled external loops | 147.1 | 8000/8000 | killed 208 SGPR spills, occupancy 12 -> 16 waves/SIMD |
| *+ lazy gf_add* | *190.1* | *8000/8000* | **REJECTED — latently incorrect** |
| canonical reduce + u128 accumulation | 135.6 | 8000/8000 | correct, slower, and the right call |
| + `NONCE_DIR_SPARSE` | ~147.5 | 8000/8000 | +7.1 %; algebra, ported from the NVIDIA kernel unchanged |
| **+ non-canonical reduce** | **176.9** | **8000/8000** | +20.6 % |

Note the shape of it: both big wins (sparse first round, non-canonical reduce) **removed work**.
Every attempt that merely re-spelled, re-scheduled or re-represented the same work lost.

## The round structure, so you do not re-derive it

This took a long time to pin down and is not written down clearly anywhere upstream. Each round is,
uniformly for both external and internal rounds:

    add round constant  ->  S-box (x^7)  ->  matrix WITHOUT constant

The sponge: zero state, absorb the header as 8 little-endian u32 field elements, permute; absorb
nonce_hi, permute (this is the midstate, hoisted out of the inner loop); absorb nonce_lo, permute;
add [1,1] to lanes 0 and 1, permute; squeeze 32 bytes, permute, squeeze 32 more.

`M4` row i is `sum(x) + x[i] + 2*x[i+1]`. `M_E` is blockwise `M4` followed by
`y[i] + (y[i&3] + y[4+i&3] + y[8+i&3])`. `M_I` is `s[i]*MAT_DIAG[i] + sum(s)`.

## The sparse first round, in one paragraph

The nonce walk is chosen so consecutive nonces differ by a vector `v = (a, -a, 0)` with
`a = (17, -11, 3, -4)`, for which **`M_E * v = 35 * (e3 - e7)`**. The entire nonce delta therefore
lands on lanes 3 and 7 after the external MDS. Ten of the twelve lanes are constant across a batch
and are precomputed once per block, so each nonce skips ten first-round S-boxes and a whole `M_E`
layer. `sparsecheck` proves the result is bit-identical to the straightforward path over 1,048,576
nonces.

This is algebra, not hardware, so it transferred from the NVIDIA kernel unchanged. A related result
from that campaign: 11-of-12 is algebraically impossible (`M_E^-1` is dense), and the best 2D lattice
is 9-of-12 and worse. The shipped version is provably optimal for this approach — no point searching
for a better one.
