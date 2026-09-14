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

__global__ void mine_kernel(const u64 *ms, const uint8_t *low_base, const uint8_t *target,
                            unsigned long long nonce0, unsigned per_thread,
                            unsigned long long *work_counter,
                            unsigned char *out_rej) {
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
    // global side effects: work counter + checksum
    if (out_rej) out_rej[tid % (256*384)] = acc;
    atomicAdd(work_counter, (unsigned long long)per_thread);
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

static void hex2bytes(const char *hex, uint8_t *out, int n) {
    for (int i=0;i<n;i++) { unsigned v; sscanf(hex+2*i, "%2x", &v); out[i]=(uint8_t)v; }
}

int main(int argc, char **argv) {
    const char *mode = argc>1 ? argv[1] : "bench";
    if (strcmp(mode, "mid")==0) {
        struct Kat { const char *h, *n, *mid; };
        Kat kats[] = {
            {"0000000000000000000000000000000000000000000000000000000000000000",
             "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
             "8646d336b5a0fccd818ca5954891634549475e8bc9c928bcc469163bcdc4900be4421fafacd59087633f49b998698d664a9067e3883e76da193039e85ea14472203d2d3ee6a8eba8ef5b9ce39bcff072c16a3c8dd5680003d77c52ab5bf7cf1f"},
            {"0101010101010101010101010101010101010101010101010101010101010101",
             "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001",
             "788cf743676a5c85dce1d727e1189b5336e0e019f5a0bf3a806487dd9c3bf83fcb6d7300de1ab5fe7dbe191fa64fc6b795dedc7fe99861beb1a3885303f7467d891eb002a34a0f174be4cf057a87208ae9113e7dc541745ffaeda908097d00f1"},
            {"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
             "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001234567890abcdef",
             "f6fc848ab2dddc18fd88d1f0ccda99fe27cfa43b8ab9947a6a505b37f02183d35f98f1d3524fb0c6d6de4efd98793e2ef3f205bd1e4d4526c0c9046201e91dd6cd2d28eef32b18dea303ecbc8e98000c3db245e7738f3acabc55f7553023863e"},
        };
        uint8_t *d_h,*d_n; u64 *d_ms, ms[12];
        hipMalloc(&d_h,32); hipMalloc(&d_n,32); hipMalloc(&d_ms,12*sizeof(u64));
        int total=0,fail=0;
        for (int ki=0; ki<(int)(sizeof(kats)/sizeof(kats[0])); ki++) {
            const Kat &k = kats[ki];
            uint8_t hb[32],nb[32],want[96];
            hex2bytes(k.h,hb,32); hex2bytes(k.n,nb,32); hex2bytes(k.mid,want,96);
            hipMemcpy(d_h,hb,32,hipMemcpyHostToDevice);
            hipMemcpy(d_n,nb,32,hipMemcpyHostToDevice);
            midstate_kernel<<<1,1>>>(d_h, d_n, d_ms);
            hipDeviceSynchronize();
            hipMemcpy(ms,d_ms,12*sizeof(u64),hipMemcpyDeviceToHost);
            total++;
            if (memcmp(ms,want,96)!=0) {
                fail++;
                printf("MID MISMATCH KAT #%d
  want ",total);
                for(int i=0;i<96;i++) printf("%02x",want[i]); printf("
  got  ");
                for(int i=0;i<96;i++) printf("%02x",((uint8_t*)ms)[i]); printf("
");
            }
        }
        printf("MID: %d/%d passed (%d failed)
", total-fail, total, fail);
        return fail?1:0;
    }

    if (strcmp(mode, "verify")==0) {
        struct Kat { const char *h, *n, *hash, *mid; };
        Kat kats[] = {
            {"0000000000000000000000000000000000000000000000000000000000000000",
             "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
             "8e64e3d8e0f38f882e8501f9e525df0a95d2e91e9cfc32c9248d756fb07780e2f8fdca2c5a54441e6fcd8d774a5f6aae72f36d1c76bc19f691a0d4f6c607e8cc",
             "8646d336b5a0fccd818ca5954891634549475e8bc9c928bcc469163bcdc4900be4421fafacd59087633f49b998698d664a9067e3883e76da193039e85ea14472203d2d3ee6a8eba8ef5b9ce39bcff072c16a3c8dd5680003d77c52ab5bf7cf1f"},
            {"0101010101010101010101010101010101010101010101010101010101010101",
             "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001",
             "e4ef79db80b642a093d7e38e6a6daac7ec1cca7293ddbe3710b4e6781be4add0b93f7e3440626df5bb23b757436787ba8d3fd0a2652e387f8af5bd8a5cc2ce2f",
             "788cf743676a5c85dce1d727e1189b5336e0e019f5a0bf3a806487dd9c3bf83fcb6d7300de1ab5fe7dbe191fa64fc6b795dedc7fe99861beb1a3885303f7467d891eb002a34a0f174be4cf057a87208ae9113e7dc541745ffaeda908097d00f1"},
            {"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
             "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001234567890abcdef",
             "7f6c410ce62e7fc54811ad3b8b92664ac0f8b1d2cd6eb163129251146e7ee34d27794aaae1b31eaaeb98ab835bae6a2b9ac4ad6f08c95903e1423f0b865816ff",
             "f6fc848ab2dddc18fd88d1f0ccda99fe27cfa43b8ab9947a6a505b37f02183d35f98f1d3524fb0c6d6de4efd98793e2ef3f205bd1e4d4526c0c9046201e91dd6cd2d28eef32b18dea303ecbc8e98000c3db245e7738f3acabc55f7553023863e"},
        };
        uint8_t *d_h,*d_n,*d_o; u64 *d_ms;
        hipMalloc(&d_h,32); hipMalloc(&d_n,64); hipMalloc(&d_o,64); hipMalloc(&d_ms,12*sizeof(u64));
        int total=0,fail=0;
        for (int ki=0; ki<(int)(sizeof(kats)/sizeof(kats[0])); ki++) {
            const Kat &k = kats[ki];
            uint8_t hb[32],nb[64],want[64],got[64];
            hex2bytes(k.h,hb,32); hex2bytes(k.n,nb,64); hex2bytes(k.hash,want,64);
            hipMemcpy(d_h,hb,32,hipMemcpyHostToDevice);
            hipMemcpy(d_n,nb,32,hipMemcpyHostToDevice);
            midstate_kernel<<<1,1>>>(d_h, d_n, d_ms);
            hipDeviceSynchronize();
            verify_kernel<<<1,1>>>(d_ms, d_n+32, 0, d_o);
            hipDeviceSynchronize();
            hipMemcpy(got,d_o,64,hipMemcpyDeviceToHost);
            total++;
            if (memcmp(got,want,64)!=0) {
                fail++;
                printf("MISMATCH KAT #%d\n  want ",total);
                for(int i=0;i<64;i++) printf("%02x",want[i]); printf("\n  got  ");
                for(int i=0;i<64;i++) printf("%02x",got[i]); printf("\n");
            }
        }
        printf("VERIFY: %d/%d passed (%d failed)\n", total-fail, total, fail);
        return fail?1:0;
    }

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
    unsigned long long nonces_per_launch = (unsigned long long)blocks*tpb*per_thread;

    unsigned long long *d_work;
    unsigned char *d_rej;
    hipMalloc(&d_work, sizeof(unsigned long long));
    hipMalloc(&d_rej, 256*384);

    hipMemset(d_work, 0, sizeof(unsigned long long));
    hipMemset(d_rej, 0, 256*384);
    mine_kernel<<<blocks,tpb>>>(d_ms,d_low,d_t,0,per_thread,d_work,d_rej);
    hipError_t err = hipGetLastError();
    hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("Warmup launch failed: %s\n", hipGetErrorString(err));
        return 1;
    }

    double secs = argc>5 ? atof(argv[5]) : 5.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    unsigned long long total_work=0; int launches=0; double elapsed=0;
    while (elapsed < secs) {
        hipMemset(d_work, 0, sizeof(unsigned long long));
        hipMemset(d_rej, 0, 256*384);
        mine_kernel<<<blocks,tpb>>>(d_ms,d_low,d_t,total_work,per_thread,d_work,d_rej);
        err = hipGetLastError();
        hipDeviceSynchronize();
        if (err != hipSuccess) {
            printf("Launch failed at iteration %d: %s\n", launches, hipGetErrorString(err));
            return 1;
        }
        unsigned long long c=0;
        hipMemcpy(&c, d_work, sizeof(unsigned long long), hipMemcpyDeviceToHost);
        total_work += c;
        launches++;
        elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
    }
    double mhs = total_work / elapsed / 1e6;
    printf("BENCH %s: %.2f MH/s (blocks=%d tpb=%d per_thread=%u, %llu nonces, %.2fs, %d launches)\n",
           prop.name, mhs, blocks, tpb, per_thread, total_work, elapsed, launches);
    hipFree(d_h);hipFree(d_hi);hipFree(d_low);hipFree(d_t);hipFree(d_ms);hipFree(d_work);hipFree(d_rej);
    return 0;
}
