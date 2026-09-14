#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint.h>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include "miner.h"

__global__ void direct_hash_kernel(const uint8_t *header, const uint8_t *nonce, uint8_t *out) {
    if (threadIdx.x || blockIdx.x) return;
    hash_from_nonce(header, nonce, out);
}

__global__ void midstate_kernel(const uint8_t *header, const uint8_t *nonce_high, u64 *out_ms) {
    if (threadIdx.x || blockIdx.x) return;
    compute_midstate(header, nonce_high, out_ms);
}

__global__ void verify_kernel(const u64 *ms, const uint8_t *low_base, unsigned idx, uint8_t *out64) {
    if (threadIdx.x || blockIdx.x) return;
    uint8_t low32[32];
    low_nonce_from_idx(low_base, idx, low32);
    int rej;
    uint8_t target[64]; for(int i=0;i<64;i++) target[i]=0xff;
    hash_from_mid_bytes(ms, low32, target, &rej, 1, out64);
}

__global__ void mine_kernel(const u64 *ms, const uint8_t *low_base, const uint8_t *target,
                            unsigned long long nonce0, unsigned per_thread,
                            unsigned long long *work_counter, unsigned char *out_rej) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long start = nonce0 + tid * per_thread;
    unsigned char acc = 0;
    #pragma unroll 1
    for (unsigned k=0;k<per_thread;k++) {
        uint8_t low32[32];
        low_nonce_from_idx(low_base, (unsigned)(start + k), low32);
        int rej;
        hash_from_mid_bytes(ms, low32, target, &rej, 0, nullptr);
        acc ^= (unsigned char)(rej ^ k);
    }
    if (out_rej) out_rej[tid % (256*384)] = acc;
    atomicAdd(work_counter, (unsigned long long)per_thread);
}

static void hex2bytes(const char *hex, uint8_t *out, int n) {
    for (int i=0;i<n;i++) { unsigned v; sscanf(hex+2*i, "%2x", &v); out[i]=(uint8_t)v; }
}

struct KatFull { const char *h, *n, *hash; };
static const KatFull KATS[] = {
    {"0000000000000000000000000000000000000000000000000000000000000000",
     "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
     "8e64e3d8e0f38f882e8501f9e525df0a95d2e91e9cfc32c9248d756fb07780e2f8fdca2c5a54441e6fcd8d774a5f6aae72f36d1c76bc19f691a0d4f6c607e8cc"},
    {"0101010101010101010101010101010101010101010101010101010101010101",
     "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001",
     "e4ef79db80b642a093d7e38e6a6daac7ec1cca7293ddbe3710b4e6781be4add0b93f7e3440626df5bb23b757436787ba8d3fd0a2652e387f8af5bd8a5cc2ce2f"},
    {"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
     "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001234567890abcdef",
     "7f6c410ce62e7fc54811ad3b8b92664ac0f8b1d2cd6eb163129251146e7ee34d27794aaae1b31eaaeb98ab835bae6a2b9ac4ad6f08c95903e1423f0b865816ff"},
};

static int do_direct_verify() {
    uint8_t *d_h,*d_n,*d_o;
    hipMalloc(&d_h,32); hipMalloc(&d_n,64); hipMalloc(&d_o,64);
    int total=sizeof(KATS)/sizeof(KATS[0]), fail=0;
    for(int ki=0;ki<total;ki++){
        uint8_t h[32],n[64],o[64],eh[64];
        hex2bytes(KATS[ki].h,h,32); hex2bytes(KATS[ki].n,n,64); hex2bytes(KATS[ki].hash,eh,64);
        hipMemcpy(d_h,h,32,hipMemcpyHostToDevice); hipMemcpy(d_n,n,64,hipMemcpyHostToDevice);
        hipLaunchKernelGGL(direct_hash_kernel,1,1,0,0,d_h,d_n,d_o);
        hipDeviceSynchronize();
        hipMemcpy(o,d_o,64,hipMemcpyDeviceToHost);
        if(memcmp(o,eh,64)){printf("KAT #%d FAILED\n",ki+1); fail++;} else printf("KAT #%d passed\n",ki+1);
    }
    return fail;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1],"verify")==0) {
        int f = do_direct_verify();
        printf("verify done: %d/%d passed\n", (int)(sizeof(KATS)/sizeof(KATS[0])) - f, (int)(sizeof(KATS)/sizeof(KATS[0])));
        return f ? 1 : 0;
    }
    printf("usage: %s verify\n", argv[0]);
    return 1;
}
