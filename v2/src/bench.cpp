#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include "miner.h"

__global__ void midstate_kernel(const uint8_t *header, const uint8_t *nonce_high, u64 *out_ms) {
    if (threadIdx.x || blockIdx.x) return;
    compute_midstate(header, nonce_high, out_ms);
}

__global__ void verify_kernel(const u64 *ms, const uint8_t *low_base, unsigned idx, uint8_t *out64) {
    if (threadIdx.x || blockIdx.x) return;
    u64 L[4];
    #pragma unroll
    for (int i=0;i<4;i++) {
        int b=8*i;
        L[i] = ((u64)low_base[b]<<56)|((u64)low_base[b+1]<<48)|((u64)low_base[b+2]<<40)|((u64)low_base[b+3]<<32)|
               ((u64)low_base[b+4]<<24)|((u64)low_base[b+5]<<16)|((u64)low_base[b+6]<<8)|(u64)low_base[b+7];
    }
    u64 f[8]; felts_from_low(L, idx, f);
    int rej;
    uint8_t target[64]; for(int i=0;i<64;i++) target[i]=0xff;
    hash_from_mid(ms, f, target, &rej, 1, out64);
}

__global__ void mine_kernel(const u64 *ms, const uint8_t *low_base, const uint8_t *target,
                            unsigned long long nonce0, unsigned per_thread,
                            unsigned long long *work_counter, unsigned char *out_rej) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long start = nonce0 + tid * per_thread;
    u64 L[4];
    #pragma unroll
    for (int i=0;i<4;i++) {
        int b=8*i;
        L[i] = ((u64)low_base[b]<<56)|((u64)low_base[b+1]<<48)|((u64)low_base[b+2]<<40)|((u64)low_base[b+3]<<32)|
               ((u64)low_base[b+4]<<24)|((u64)low_base[b+5]<<16)|((u64)low_base[b+6]<<8)|(u64)low_base[b+7];
    }
    unsigned char acc = 0;
    #pragma unroll 1
    for (unsigned k=0;k<per_thread;k++) {
        u64 f[8];
        felts_from_low(L, start + k, f);
        int rej;
        hash_from_mid(ms, f, target, &rej, 0, nullptr);
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

static int do_verify() {
    uint8_t *d_h,*d_n,*d_o; u64 *d_ms;
    hipMalloc(&d_h,32); hipMalloc(&d_n,64); hipMalloc(&d_o,64); hipMalloc(&d_ms,12*sizeof(u64));
    int total=sizeof(KATS)/sizeof(KATS[0]), fail=0;
    for (int ki=0; ki<total; ki++) {
        const KatFull &k = KATS[ki];
        uint8_t hb[32],nb[64],want[64],got[64];
        hex2bytes(k.h,hb,32); hex2bytes(k.n,nb,64); hex2bytes(k.hash,want,64);
        hipMemcpy(d_h,hb,32,hipMemcpyHostToDevice);
        hipMemcpy(d_n,nb,32,hipMemcpyHostToDevice);
        midstate_kernel<<<1,1>>>(d_h, d_n, d_ms);
        hipDeviceSynchronize();
        verify_kernel<<<1,1>>>(d_ms, d_n+32, 0, d_o);
        hipDeviceSynchronize();
        hipMemcpy(got,d_o,64,hipMemcpyDeviceToHost);
        if (memcmp(got,want,64)!=0) {
            fail++;
            printf("KAT #%d mismatch\n  want ", ki);
            for(int i=0;i<64;i++) printf("%02x",want[i]); printf("\n  got  ");
            for(int i=0;i<64;i++) printf("%02x",got[i]); printf("\n");
        }
    }
    printf("VERIFY: %d/%d passed\n", total-fail, total);
    return fail ? 1 : 0;
}

static int do_bench(int argc, char **argv) {
    uint8_t header[32], start[64], target[64];
    for (int i=0;i<32;i++) header[i]=(uint8_t)(i*7+1);
    memset(start,0,64); start[40]=0x12;
    for(int i=0;i<64;i++) target[i]=0xff;

    uint8_t *d_h,*d_hi,*d_low,*d_t; u64 *d_ms;
    hipMalloc(&d_h,32); hipMalloc(&d_hi,32); hipMalloc(&d_low,32); hipMalloc(&d_t,64); hipMalloc(&d_ms,12*sizeof(u64));
    hipMemcpy(d_h,header,32,hipMemcpyHostToDevice);
    hipMemcpy(d_hi,start,32,hipMemcpyHostToDevice);
    hipMemcpy(d_low,start+32,32,hipMemcpyHostToDevice);
    hipMemcpy(d_t,target,64,hipMemcpyHostToDevice);
    midstate_kernel<<<1,1>>>(d_h, d_hi, d_ms);
    hipDeviceSynchronize();

    hipDeviceProp_t prop; hipGetDeviceProperties(&prop,0);
    int tpb = argc>3 ? atoi(argv[3]) : 256;
    int blocks = argc>4 ? atoi(argv[4]) : prop.multiProcessorCount * 8;
    unsigned per_thread = argc>2 ? (unsigned)atoi(argv[2]) : 16;

    unsigned long long *d_work; unsigned char *d_rej;
    hipMalloc(&d_work,sizeof(unsigned long long)); hipMalloc(&d_rej,256*384);
    hipMemset(d_work,0,sizeof(unsigned long long)); hipMemset(d_rej,0,256*384);
    mine_kernel<<<blocks,tpb>>>(d_ms,d_low,d_t,0,per_thread,d_work,d_rej);
    hipDeviceSynchronize();

    double secs = argc>5 ? atof(argv[5]) : 5.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    unsigned long long total_work=0; int launches=0; double elapsed=0;
    while (elapsed < secs) {
        hipMemset(d_work,0,sizeof(unsigned long long)); hipMemset(d_rej,0,256*384);
        mine_kernel<<<blocks,tpb>>>(d_ms,d_low,d_t,total_work,per_thread,d_work,d_rej);
        hipDeviceSynchronize();
        unsigned long long c=0; hipMemcpy(&c,d_work,sizeof(unsigned long long),hipMemcpyDeviceToHost);
        total_work += c; launches++;
        elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
    }
    double mhs = total_work / elapsed / 1e6;
    printf("BENCH %s: %.2f MH/s (blocks=%d tpb=%d per_thread=%u, %llu nonces, %.2fs, %d launches)\n",
           prop.name, mhs, blocks, tpb, per_thread, total_work, elapsed, launches);
    hipFree(d_h);hipFree(d_hi);hipFree(d_low);hipFree(d_t);hipFree(d_ms);hipFree(d_work);hipFree(d_rej);
    return 0;
}

int main(int argc, char **argv) {
    const char *mode = argc>1 ? argv[1] : "bench";
    if (strcmp(mode,"verify")==0) return do_verify();
    return do_bench(argc, argv);
}
