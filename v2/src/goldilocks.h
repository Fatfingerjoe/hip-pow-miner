#pragma once
#include "constants.h"
#include <hip/hip_runtime.h>

#define lo32(x) ((u32)(x))
#define hi32(x) ((u32)((x) >> 32))

static __host__ __device__ __forceinline__ u64 gf_canon(u64 x) {
    return (x >= GOLDI_P) ? (x - GOLDI_P) : x;
}

// Fast addition matching Rust qp-poseidon (result may be non-canonical, in [0,2^64)).
// For x+y: if x+y < 2^64 return x+y; else return x+y + EPS (since 2^64 ≡ EPS mod P).
static __host__ __device__ __forceinline__ u64 gf_addL(u64 a, u64 b) {
    u64 r = a + b;
    if (r < a) r += EPS;
    return r;
}

static __host__ __device__ __forceinline__ u64 eps_mul32(u32 x) {
    return ((u64)x << 32) - (u64)x;
}

static __device__ __forceinline__ void mul128(u64 a, u64 b, u64 &lo, u64 &hi) {
#if defined(__HIP_DEVICE_COMPILE__)
    lo = a * b;
    hi = __umul64hi(a, b);
#else
    unsigned __int128 p = (unsigned __int128)a * b;
    lo = (u64)p; hi = (u64)(p >> 64);
#endif
}

// Reduce 128-bit product to [0, 2^64). Matches Rust reduce128.
static __device__ __forceinline__ u64 gf_reduce128(u64 lo, u64 hi) {
    u32 hh = hi32(hi);
    u32 hl = lo32(hi);
    u64 t0 = lo - (u64)hh;
    if (lo < (u64)hh) t0 -= EPS;        // borrow: subtract NEG_ORDER
    u64 t1 = eps_mul32(hl);             // hl * EPS
    u64 r = t0 + t1;
    if (r < t1) r += EPS;              // carry into high word -> +EPS
    return r;
}

static __device__ __forceinline__ u64 gf_mul(u64 a, u64 b) {
    u64 lo, hi; mul128(a, b, lo, hi); return gf_reduce128(lo, hi);
}

static __device__ __forceinline__ u64 gf_sqr(u64 a) { return gf_mul(a, a); }

static __device__ __forceinline__ u64 gf_exp7(u64 x) {
    u64 x2 = gf_sqr(x);
    u64 x3 = gf_mul(x2, x);
    u64 x4 = gf_sqr(x2);
    return gf_mul(x4, x3);
}
