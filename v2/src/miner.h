#pragma once
#include "poseidon2.h"
#include <cstring>

static __device__ __forceinline__ u64 be64_load(const uint8_t *p) {
    return ((u64)p[0]<<56)|((u64)p[1]<<48)|((u64)p[2]<<40)|((u64)p[3]<<32)|
           ((u64)p[4]<<24)|((u64)p[5]<<16)|((u64)p[6]<<8)|(u64)p[7];
}
static __device__ __forceinline__ void be64_store(uint8_t *p, u64 v) {
    p[0]=v>>56;p[1]=v>>48;p[2]=v>>40;p[3]=v>>32;
    p[4]=v>>24;p[5]=v>>16;p[6]=v>>8;p[7]=v;
}
static __device__ __forceinline__ u64 bswap64(u64 x) {
    u32 lo = lo32(x), hi = hi32(x);
    return ((u64)__builtin_bswap32(lo) << 32) | (u64)__builtin_bswap32(hi);
}

// Build 8 LE-u32 felts from the 32-byte big-endian low-nonce. L[0..3] = 4 BE-u64 limbs.
static __device__ __forceinline__ void felts_from_low(const u64 L[4], u64 idx, u64 f[8]) {
    u64 l3 = L[3] + idx; u64 c = (l3 < L[3]) ? 1u : 0u;
    u64 l2 = L[2] + c;   c = (l2 < L[2]) ? 1u : 0u;
    u64 l1 = L[1] + c;   c = (l1 < L[1]) ? 1u : 0u;
    u64 l0 = L[0] + c;
    u64 lim[4] = {l0, l1, l2, l3};
    #pragma unroll
    for (int m=0;m<4;m++) {
        u32 hi = (u32)(lim[m] >> 32);
        u32 lo = (u32)lim[m];
        f[2*m]   = (u64)__builtin_bswap32(hi);
        f[2*m+1] = (u64)__builtin_bswap32(lo);
    }
}

static __device__ __forceinline__ int cmp256(const u64 *st, const uint8_t *target) {
    #pragma unroll
    for (int i=0;i<4;i++) {
        u64 hv = bswap64(gf_canon(st[i]));
        u64 tv = be64_load(target + 8*i);
        if (hv != tv) return (hv > tv) ? 1 : 2;
    }
    return 0;
}

// Midstate after header + nonce_high (2 permutations), canonical.
static __device__ __forceinline__ void compute_midstate(const uint8_t *header, const uint8_t *nonce_high, u64 *ms) {
    #pragma unroll
    for (int i=0;i<12;i++) ms[i] = 0;
    #pragma unroll
    for (int i=0;i<8;i++) {
        int b = i*4;
        u64 f = (u64)header[b] | ((u64)header[b+1]<<8) | ((u64)header[b+2]<<16) | ((u64)header[b+3]<<24);
        ms[i] = gf_addL(ms[i], f);
    }
    permute(ms);
    #pragma unroll
    for (int i=0;i<8;i++) {
        int b = i*4;
        u64 f = (u64)nonce_high[b] | ((u64)nonce_high[b+1]<<8) | ((u64)nonce_high[b+2]<<16) | ((u64)nonce_high[b+3]<<24);
        ms[i] = gf_addL(ms[i], f);
    }
    permute(ms);
    #pragma unroll
    for (int i=0;i<12;i++) ms[i] = gf_canon(ms[i]);
}

static __device__ __forceinline__ void hash_from_mid(const u64 *ms, const u64 *f, const uint8_t *target,
                                                    int *rej, int want_full, uint8_t *out64) {
    u64 s[12];
    #pragma unroll
    for (int i=0;i<12;i++) s[i] = ms[i];
    // perm 3: absorb 8 low-nonce felts
    #pragma unroll
    for (int i=0;i<8;i++) s[i] = gf_addL(s[i], f[i]);
    permute(s);

    // perm 4: absorb [1,1], then squeeze
    s[0] = gf_addL(s[0], 1);
    s[1] = gf_addL(s[1], 1);
    permute(s);
    *rej = (cmp256(s, target) == 2) ? 0 : 1;
    if (want_full) {
        #pragma unroll
        for (int i=0;i<4;i++) {
            u64 v = gf_canon(s[i]);
            #pragma unroll
            for (int j=0;j<8;j++) out64[i*8+j] = (uint8_t)(v >> (8*j));
        }
    }

    // perm 5: second squeeze
    permute(s);
    if (want_full) {
        #pragma unroll
        for (int i=0;i<4;i++) {
            u64 v = gf_canon(s[i]);
            #pragma unroll
            for (int j=0;j<8;j++) out64[32+i*8+j] = (uint8_t)(v >> (8*j));
        }
    }
}

// Direct full hash for verify/oracle comparison (no midstate shortcut).
static __device__ __forceinline__ void hash_from_nonce(const uint8_t *header, const uint8_t *nonce, uint8_t *out64) {
    u64 s[12];
    #pragma unroll
    for(int i=0;i<12;i++) s[i]=0;
    #pragma unroll
    for(int i=0;i<8;i++){
        int b=i*4;
        u64 f=(u64)header[b]|((u64)header[b+1]<<8)|((u64)header[b+2]<<16)|((u64)header[b+3]<<24);
        s[i]=gf_addL(s[i],f);
    }
    permute(s);
    #pragma unroll
    for(int i=0;i<8;i++){
        int b=i*4;
        u64 f=(u64)nonce[b]|((u64)nonce[b+1]<<8)|((u64)nonce[b+2]<<16)|((u64)nonce[b+3]<<24);
        s[i]=gf_addL(s[i],f);
    }
    permute(s);
    #pragma unroll
    for(int i=0;i<8;i++){
        int b=32+i*4;
        u64 f=(u64)nonce[b]|((u64)nonce[b+1]<<8)|((u64)nonce[b+2]<<16)|((u64)nonce[b+3]<<24);
        s[i]=gf_addL(s[i],f);
    }
    permute(s);
    s[0]=gf_addL(s[0],1); s[1]=gf_addL(s[1],1);
    permute(s);
    #pragma unroll
    for(int i=0;i<4;i++){
        u64 v=gf_canon(s[i]);
        #pragma unroll
        for(int j=0;j<8;j++) out64[i*8+j]=(uint8_t)(v>>(8*j));
    }
    permute(s);
    #pragma unroll
    for(int i=0;i<4;i++){
        u64 v=gf_canon(s[i]);
        #pragma unroll
        for(int j=0;j<8;j++) out64[32+i*8+j]=(uint8_t)(v>>(8*j));
    }
}
