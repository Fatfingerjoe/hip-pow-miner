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
    if (strcmp(mode, "verify")==0) {
        char line[512];
        uint8_t *d_h,*d_n,*d_o;
        hipMalloc(&d_h,32); hipMalloc(&d_n,64); hipMalloc(&d_o,64);
        int total=0,fail=0;
        while (fgets(line,sizeof(line),stdin)) {
            char hh[65],nh[129],gh[129];
            if (sscanf(line, "%64s %128s %128s", hh,nh,gh)!=3) continue;
            uint8_t hb[32],nb[64],want[64],got[64];
            hex2bytes(hh,hb,32); hex2bytes(nh,nb,64); hex2bytes(gh,want,64);
            hipMemcpy(d_h,hb,32,hipMemcpyHostToDevice);
            hipMemcpy(d_n,nb,64,hipMemcpyHostToDevice);
            verify_kernel<<<1,1>>>((const u64 *)d_h, d_n, 0, d_o);
            hipDeviceSynchronize();
            hipMemcpy(got,d_o,64,hipMemcpyDeviceToHost);
            total++;
            if (memcmp(got,want,64)!=0) {
                fail++;
                if (fail<=3) {
                    printf("MISMATCH #%d\n  want ",total);
                    for(int i=0;i<32;i++) printf("%02x",want[i]); printf("...\n  got  ");
                    for(int i=0;i<32;i++) printf("%02x",got[i]); printf("...\n");
                }
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
