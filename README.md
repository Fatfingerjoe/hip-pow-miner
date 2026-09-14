# qpow-hip — AMD GPU miner for the Quantus pool

A native AMD (ROCm/HIP) miner for QPoW. On an RX 7900 XTX it does **~175 MH/s**, versus **~96 MH/s**
for the official miner and quanpool on the same card — both of which fall back to a generic
wgpu/Vulkan path because neither has a native AMD kernel.

## Which file do I download?

**Two builds are published. They are not interchangeable — pick the one matching your ROCm.**

    rocm --version      # or:  ls -d /opt/rocm*   |   apt list --installed 2>/dev/null | grep rocm

| your ROCm | download | speed on RX 7900 XTX |
| --- | --- | ---: |
| **5.x** (e.g. Ubuntu 24.04 `apt install hipcc`) | `qpow-hip-linux-x86_64-rocm5` | **175 MH/s** |
| **6.x** (AMD's `amdgpu-install --usecase=rocm`) | `qpow-hip-linux-x86_64-rocm6` | **130 MH/s** |

The ROCm 5 build is ~34 % faster. That is a property of the ROCm *runtime*, not the kernel — the
identical source compiled against 6.4 runs slower. If you have a choice, ROCm 5.x is worth it.

### "It won't start"

If you picked the wrong one you will see, before the miner prints anything:

    ./qpow-hip-linux-x86_64-rocm5: error while loading shared libraries:
    libamdhip64.so.5: cannot open shared object file: No such file or directory

**That means you have ROCm 6 — download the `-rocm6` build** (and vice versa for `libamdhip64.so.6`).
This happens in the dynamic loader before the program runs, so the miner cannot warn you itself.

## Requirements

- **Linux x86_64**, glibc >= 2.38 (Ubuntu 24.04+, Debian 13+, Fedora 39+)
- **ROCm runtime installed** (the binary is ~1 MB and links against it; it is not self-contained)
- **AMD RDNA3 GPU — gfx1100 only**: RX 7900 XTX / XT / GRE.
  Other AMD GPUs are *not* supported by this build and will report a gfx mismatch.
- **`/dev/kfd` present.** Check with `ls -l /dev/kfd`. If missing, ROCm cannot open the GPU.
  In Docker you must pass:
  `--device=/dev/kfd --device=/dev/dri --group-add video --group-add render`
- Your user in the `render` and `video` groups (`id | grep -E "render|video"`).

The miner runs a preflight check and prints a specific checklist if any of this is wrong.

## Usage

    ./qpow-hip-linux-x86_64-rocm5 pool ws://quantusminer.com:9901 <YOUR_PAYOUT_ADDRESS> <worker-name>

Optional trailing args: `[nonces_per_thread] [threads_per_block]` (defaults `32 512`).

Other modes:

| mode | what it does |
| --- | --- |
| `version` | build + runtime ROCm version and detected device |
| `verify < big_vectors.txt` | 8,000-vector known-answer test against the reference hash |
| `sharecheck <n>` | proves submitted nonces re-hash correctly (share validity) |
| `benchsparse <secs>` | offline hashrate benchmark |

## Correctness

Every published build passes three gates:

- **VERIFY 8000/8000** — matches the reference implementation's known-answer vectors exactly.
- **sparsecheck** — the optimised sparse first-round path equals the straightforward path
  over 1,048,576 nonces, 0 wrong.
- **sharecheck** — a winning nonce, reconstructed and re-hashed through the verified full-hash
  path, matches the mining path over 262,144 cases, 0 wrong. This is what proves shares validate.

Measured on the pool: **1,248 of 1,255 shares accepted** over a 44-minute run; the 7 rejects were
stale shares during the initial vardiff ramp, none after it settled.

## Is it worth running?

Honestly: only if you already own the card. At ~175 MH/s an RX 7900 XTX is beaten by a ~$350
RTX 5060 at 265 MH/s. QPoW is dominated by 64-bit integer multiplies, which RDNA3 emulates.
This exists so AMD owners can mine at all — not because AMD is competitive here.

## Building from source

Requires a ROCm/HIP toolchain (`hipcc`) on Linux. There is no CMake step — two translation
units, compiled in one command.

    ./build.sh                 # -> ./qpow_hip
    HIPCC=/opt/rocm/bin/hipcc ./build.sh
    ARCH=gfx1100 ./build.sh

Equivalently:

    hipcc --offload-arch=gfx1100 -O3 -mcode-object-version=4 \
          qpow_hip.cpp easywsclient.cpp -o qpow_hip

`-mcode-object-version=4` is required whenever a 6.x `hipcc` wrapper is linking the 5.x
runtime — the usual state after running AMD's `amdgpu-install` on a box that already had
distro ROCm. Without it you get a binary that compiles and then dies at startup with
`create kernel metadata map using COMgr` / `shared object initialization failed`.

Do **not** build against the 6.4 runtime (`--rocm-path=/opt/rocm-6.4.4`) if you can avoid it:
the result is correct but about 24 % slower.

## Continuing the optimization work

See **[OPTIMIZING.md](OPTIMIZING.md)** — where the performance stands, how to measure a change without
fooling yourself, the correctness trap that makes a wrong kernel look 40 % faster, what is still
worth trying, and the full list of things already measured and rejected.

## Repository layout

| path | what it is |
| --- | --- |
| `qpow_hip.cpp` | the miner — kernels, nonce walk, all run modes |
| `pool.inc` | pool client (WebSocket, job handling, share submission), `#include`d by the above |
| `constants_hip.h` | Poseidon2 round constants and MDS diagonal |
| `easywsclient.{hpp,cpp}` | vendored WebSocket client, from https://github.com/dhbaird/easywsclient |
| `build.sh` | the build |
| `tests/` | correctness harnesses — see `tests/README.md` |
| `experiments/` | optimization variants kept as a record — see `experiments/README.md` |

## Algorithm notes

QPoW is Poseidon2 over the Goldilocks field (p = 2^64 - 2^32 + 1), width 12, 8 external and
22 partial rounds, S-box x^7. Each nonce costs two full permutations on the mining path.

The hot-path optimization is a **sparse first round**. The nonce walk is chosen so that
consecutive nonces differ by a vector `v` with `M_E * v = 35 * (e3 - e7)` — i.e. the external
MDS maps the whole nonce delta onto just two lanes. Ten of the twelve state lanes after the
first layer are therefore constant across an entire batch and are precomputed once per block
(`sparse_pre`), leaving only lanes 3 and 7 to recompute per nonce. The `sparsecheck` gate
proves this path is bit-identical to the straightforward one over 1,048,576 nonces.
