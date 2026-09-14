#pragma once
#include "goldilocks.h"

static __device__ __forceinline__ void apply_mat4(u64 *x) {
    u64 t01  = gf_addL(x[0], x[1]);
    u64 t23  = gf_addL(x[2], x[3]);
    u64 t0123 = gf_addL(t01, t23);
    u64 t01123 = gf_addL(t0123, x[1]);
    u64 t01233 = gf_addL(t0123, x[3]);
    x[3] = gf_addL(t01233, gf_addL(x[0], x[0]));
    x[1] = gf_addL(t01123, gf_addL(x[2], x[2]));
    x[0] = gf_addL(t01123, t01);
    x[2] = gf_addL(t01233, t23);
}

static __device__ __forceinline__ void external_linear(u64 *s) {
    u64 y[12];
    #pragma unroll
    for (int i=0;i<12;i++) y[i] = s[i];
    #pragma unroll
    for (int c=0;c<12;c+=4) apply_mat4(y + c);
    u64 sum[4];
    #pragma unroll
    for (int i=0;i<4;i++) sum[i] = gf_addL(gf_addL(y[i], y[4+i]), y[8+i]);
    #pragma unroll
    for (int i=0;i<12;i++) s[i] = gf_addL(y[i], sum[i & 3]);
}

static __device__ __forceinline__ void internal_linear(u64 *s) {
    u64 sum = s[0];
    #pragma unroll
    for (int i=1;i<12;i++) sum = gf_addL(sum, s[i]);
    #pragma unroll
    for (int i=0;i<12;i++) {
        u64 lo, hi;
        mul128(s[i], MAT_DIAG[i], lo, hi);
        u64 t = lo + sum;
        if (t < lo) hi += 1;
        s[i] = gf_reduce128(t, hi);
    }
}

static __device__ __forceinline__ void permute(u64 *s) {
    // Initial external linear layer (no constants)
    external_linear(s);

    // 4 initial external rounds: add RC -> S-box -> linear layer
    #pragma unroll
    for (int r=0;r<4;r++) {
        #pragma unroll
        for (int i=0;i<12;i++) s[i] = gf_addL(s[i], INIT_EXT_RC[r][i]);
        #pragma unroll
        for (int i=0;i<12;i++) s[i] = gf_exp7(s[i]);
        external_linear(s);
    }

    // 22 internal rounds
    #pragma unroll
    for (int r=0;r<22;r++) {
        s[0] = gf_addL(s[0], INTERNAL_RC[r]);
        s[0] = gf_exp7(s[0]);
        internal_linear(s);
    }

    // 4 terminal external rounds
    #pragma unroll
    for (int r=0;r<4;r++) {
        #pragma unroll
        for (int i=0;i<12;i++) s[i] = gf_addL(s[i], TERM_EXT_RC[r][i]);
        #pragma unroll
        for (int i=0;i<12;i++) s[i] = gf_exp7(s[i]);
        external_linear(s);
    }
}
