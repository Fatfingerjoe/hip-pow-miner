# Correctness harnesses

Standalone programs used to validate the miner. They are not part of the miner build and are
not needed to run it. Each includes `constants_hip.h` from the repo root, so build them **from the repository root** with `-I.`:

    hipcc -I. --offload-arch=gfx1100 -O3 -mcode-object-version=4 tests/gputest.cpp -o /tmp/gputest

Host-only files still go through `hipcc` because on a typical ROCm container it is the only
C++ compiler in `PATH` (`hipcc -x c++ ...`).

| file | what it checks |
| --- | --- |
| `fieldtest.cpp` | Goldilocks field arithmetic against exact 128-bit reference arithmetic |
| `ft2.cpp` | the non-canonical reduction `gf_red` — proves outputs are congruent mod p and that the downstream code tolerates non-canonical inputs |
| `ft3.cpp` | further reduction/edge-case cases |
| `diff.cpp` | **found the lazy-`gf_add` bug**: results differing from the reference by exactly 2^32-1 |
| `diff2.cpp` | narrowing follow-up to `diff.cpp` |
| `gputest.cpp` | reproduces the B/C divergence (see `experiments/README.md`) — GPU disagrees with CPU on one specific expression form |
| `minimal.cpp` | minimal GPU-vs-CPU `__uint128_t` cases (2/3/4-term and compound). All pass — falsified the "multi-term u128" hypothesis |
| `iso.cpp` | isolates the exact `S + x[i] + 2t` accumulation and `rd()`. GPU matches CPU — falsified the same hypothesis a second way |
| `col.cpp` | external-MDS column extraction, used to derive the sparse nonce direction |

## The three gates every published build must pass

Run against the built miner, not these harnesses:

    ./qpow_hip verify < big_vectors.txt   # 8000/8000 known-answer vectors
    ./qpow_hip sparsecheck                # 1,048,576 nonces, sparse path == straightforward path
    ./qpow_hip sharecheck 262144          # winning nonces re-hash correctly through the full-hash path

`sharecheck` is the one that proves shares will validate at the pool.
