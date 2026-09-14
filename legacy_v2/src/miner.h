#pragma once
#include "poseidon2.h"
#include <cstring>

static __device__ __forceinline__ u64 be64_load(const uint8_t *p) {
    return ((u64)p[0]<<56)|((u64)p[1]<<48)|((u64)p[2]<<40)|((u64)p[3]<<32)|
           ((u64)p[4]<<24)|((u64)p[5]<<16)|((u64)p[6]<<8)|(u64)p[7];
}

static __device__ __forceinline__ u64 le32_load(const uint8_t *p) {
    return (u64)p[0]|((u64)p[1]<<8)|((u64)p[2]<<16)|((u64)p[3]<<24);
}

static __device__ __forceinline__ void digest_to_bytes(const u64 *st, uint8_t *out32) {
    #pragma unroll
    for (int i=0;i<4;i++) {
        u64 v = gf_canon(st[i]);
        #pragma unroll
        for (int j=0;j<8;j++) out32[i*8+j] = (uint8_t)(v >> (8*j));
    }
}

static __device__ __forceinline__ int cmp256(const u64 *st, const uint8_t *target) {
    #pragma unroll
    for (int i=0;i<4;i++) {
        u64 hv = __builtin_bswap64(gf_canon(st[i]));
        u64 tv = be64_load(target + 8*i);
        if (hv != tv) return (hv > tv) ? 1 : 2;
    }
    return 0;
}

struct Sponge {
    u64 state[12];
    u64 buf[8];
    int buf_len;
};

static __device__ __forceinline__ void sponge_init(Sponge *sp) {
    #pragma unroll
    for (int i=0;i<12;i++) sp->state[i]=0;
    #pragma unroll
    for (int i=0;i<8;i++) sp->buf[i]=0;
    sp->buf_len=0;
}

static __device__ __forceinline__ void sponge_flush_buffer(Sponge *sp) {
    if (sp->buf_len == 8) {
        #pragma unroll
        for (int i=0;i<8;i++) sp->state[i] = gf_addL(sp->state[i], sp->buf[i]);
        permute(sp->state);
        sp->buf_len = 0;
    }
}

static __device__ __forceinline__ void sponge_push(Sponge *sp, u64 felt) {
    sp->buf[sp->buf_len++] = felt;
    sponge_flush_buffer(sp);
}

static __device__ __forceinline__ void sponge_append_bytes(Sponge *sp, const uint8_t *bytes, int n) {
    int pos=0;
    while (pos + 4 <= n) {
        u64 f = le32_load(bytes + pos);
        sponge_push(sp, f);
        pos += 4;
    }
    int rem = n - pos;
    uint8_t last[4] = {0,0,0,0};
    #pragma unroll
    for (int i=0;i<rem;i++) last[i] = bytes[pos+i];
    if (rem < 4) last[rem] = 1;
    sponge_push(sp, le32_load(last));
}

static __device__ __forceinline__ void sponge_finalize_squeeze_twice(Sponge *sp, uint8_t *out64) {
    sponge_push(sp, 1);
    while (sp->buf_len != 0) {
        sponge_push(sp, 0);
    }
    digest_to_bytes(sp->state, out64);
    permute(sp->state);
    digest_to_bytes(sp->state, out64 + 32);
}

static __device__ __forceinline__ void hash_from_nonce(const uint8_t *header, const uint8_t *nonce, uint8_t *out64) {
    Sponge sp;
    sponge_init(&sp);
    sponge_append_bytes(&sp, header, 32);
    sponge_append_bytes(&sp, nonce, 64);
    sponge_finalize_squeeze_twice(&sp, out64);
}

static __device__ __forceinline__ void compute_midstate(const uint8_t *header, const uint8_t *nonce_high, u64 *ms) {
    Sponge sp;
    sponge_init(&sp);
    sponge_append_bytes(&sp, header, 32);
    sponge_append_bytes(&sp, nonce_high, 32);
    sponge_flush_buffer(&sp);
    #pragma unroll
    for (int i=0;i<12;i++) ms[i] = gf_canon(sp.state[i]);
}

// Build 32-byte low-nonce by adding idx (big-endian uint256) to low_base bytes.
static __device__ __forceinline__ void low_nonce_from_idx(const uint8_t *low_base, unsigned idx, uint8_t *out32) {
    #pragma unroll
    for (int i=0;i<32;i++) out32[i] = low_base[i];
    unsigned c = idx;
    for (int i=31;i>=0 && c;i--) {
        unsigned sum = out32[i] + c;
        out32[i] = (uint8_t)sum;
        c = sum >> 8;
    }
}

static __device__ __forceinline__ void hash_from_mid_bytes(const u64 *ms, const uint8_t *low32,
                                                           const uint8_t *target,
                                                           int *rej, int want_full, uint8_t *out64) {
    u64 s[12];
    #pragma unroll
    for (int i=0;i<12;i++) s[i] = ms[i];
    // low nonce is exactly 32 bytes = 8 felts; buffer empty at midstate for aligned 64 bytes
    #pragma unroll
    for (int i=0;i<8;i++) s[i] = gf_addL(s[i], le32_load(low32 + 4*i));
    permute(s);
    // finalize_state
    #pragma unroll
    for (int i=0;i<8;i++) s[i] = gf_addL(s[i], (i==0)?1:0);
    permute(s);
    *rej = (cmp256(s, target) == 2) ? 0 : 1;
    if (want_full) digest_to_bytes(s, out64);
    permute(s);
    if (want_full) digest_to_bytes(s, out64 + 32);
}
