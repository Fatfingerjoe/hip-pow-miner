// QPoW Poseidon2-Goldilocks for AMD RDNA3 (gfx1100) via HIP.
// Milestone 1: CORRECTNESS. Plain __uint128_t arithmetic, one nonce per thread.
// Validated against big_vectors.txt (the same 8,000-vector KAT the CUDA miner uses).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <ctime>
#include <unistd.h>
#include <string>
#include <vector>
typedef unsigned long long u64;
typedef unsigned int u32;
#define P 0xFFFFFFFF00000001ULL      // Goldilocks 2^64 - 2^32 + 1

__device__ __constant__ u64 D_INTERNAL_RC[22];
__device__ __constant__ u64 D_MAT_DIAG[12];
__device__ __constant__ u64 D_INIT_EXT_RC[4][12];
__device__ __constant__ u64 D_TERM_EXT_RC[4][12];


// 128-bit -> Goldilocks. Uses 2^64 = 2^32-1 and 2^96 = -1 (mod p).
__device__ __forceinline__ u64 gf_red(__uint128_t x){
    u64 lo=(u64)x, hi=(u64)(x>>64);
    u32 hh=(u32)(hi>>32), hl=(u32)hi;
    u64 t = lo - (u64)hh;
    if (lo < (u64)hh) t -= 0xFFFFFFFFULL;
    u64 b = ((u64)hl<<32) - (u64)hl;
    u64 r = t + b;
    if (r < t) r += 0xFFFFFFFFULL;
    // NOTE: output is congruent mod P but NOT canonical (validated in ft2.cpp).
    // Safe: every consumer either accumulates in u128 (exact for any u64) or does one
    // conditional subtract, and 2^64-P = 2^32-1 < P so one subtract always suffices.
    return r;
}
__device__ __forceinline__ u64 gf_add(u64 a, u64 b){ return gf_red((__uint128_t)a + (__uint128_t)b); }
// 64x64 -> 128 multiply via inline GCN v_mad_u64_u32.
// Decompose a = a1<<32 + a0, b = b1<<32 + b0.
__device__ __forceinline__ u64 gf_mul(u64 a, u64 b){
    u32 a0=(u32)a, a1=(u32)(a>>32);
    u32 b0=(u32)b, b1=(u32)(b>>32);

    // a0*b0  = p0lo:p0hi
    u64 p0lo, p0hi;
    asm volatile("v_mad_u64_u32 %0, %1, %2, %3, 0" : "=v"(p0lo), "=v"(p0hi) : "v"(a0), "v"(b0));

    // cross0 = a0*b1 + p0hi  = c0lo:c0hi
    u64 c0lo, c0hi;
    asm volatile("v_mad_u64_u32 %0, %1, %2, %3, %4" : "=v"(c0lo), "=v"(c0hi) : "v"(a0), "v"(b1), "v"(p0hi));

    // cross1 = a1*b0 + c0lo  = c1lo:c1hi
    u64 c1lo, c1hi;
    asm volatile("v_mad_u64_u32 %0, %1, %2, %3, %4" : "=v"(c1lo), "=v"(c1hi) : "v"(a1), "v"(b0), "v"(c0lo));

    // high = a1*b1 + c0hi + c1hi
    u64 hcarry = c0hi + c1hi;
    u64 hlo, hhi;
    asm volatile("v_mad_u64_u32 %0, %1, %2, %3, %4" : "=v"(hlo), "=v"(hhi) : "v"(a1), "v"(b1), "v"(hcarry));

    // Reconstruct 128-bit product:
    // bits[0..31]  = p0lo
    // bits[32..63] = c1lo
    // bits[64..95] = hlo
    // bits[96..127]= hhi
    __uint128_t prod = (__uint128_t)p0lo
                     + ((__uint128_t)c1lo << 32)
                     + ((__uint128_t)hlo  << 64)
                     + ((__uint128_t)hhi  << 96);
    return gf_red(prod);
}
__device__ __forceinline__ u64 gf_x7(u64 x){
    u64 x2 = gf_mul(x,x), x4 = gf_mul(x2,x2), x6 = gf_mul(x4,x2); return gf_mul(x6,x);
}
// M4 row i = sum(x) + x[i] + 2*x[i+1]
__device__ __forceinline__ void mat4(u64 *x){
    u64 s = gf_add(gf_add(x[0],x[1]), gf_add(x[2],x[3]));
    u64 o[4];
    #pragma unroll
    for (int i=0;i<4;i++){ u64 t = x[(i+1)&3]; o[i] = gf_add(gf_add(s,x[i]), gf_add(t,t)); }
    #pragma unroll
    for (int i=0;i<4;i++) x[i]=o[i];
}
// M_E: blockwise M4, then y[i] + (y[i&3]+y[4+i&3]+y[8+i&3]) + rc[i]
__device__ __forceinline__ void ext_lin(u64 *s, const u64 *rc){
    #pragma unroll
    for (int c=0;c<12;c+=4) mat4(s+c);
    u64 sum[4];
    #pragma unroll
    for (int i=0;i<4;i++) sum[i]=gf_add(gf_add(s[i],s[4+i]),s[8+i]);
    #pragma unroll
    for (int i=0;i<12;i++){ __uint128_t z=(__uint128_t)s[i]+(__uint128_t)sum[i&3]; if(rc) z+=(__uint128_t)rc[i]; s[i]=gf_red(z); }
}
// internal: s[i] = s[i]*MAT_DIAG[i] + sum + (rc0 on lane 0)
__device__ __forceinline__ void int_lin(u64 *s, u64 rc0){
    __uint128_t acc=0;
    #pragma unroll
    for (int i=0;i<12;i++) acc += (__uint128_t)s[i];
    u64 sum = gf_red(acc);
    #pragma unroll
    for (int i=0;i<12;i++){ __uint128_t v=(__uint128_t)gf_mul(s[i],D_MAT_DIAG[i])+(__uint128_t)sum; if(i==0) v+=(__uint128_t)rc0; s[i]=gf_red(v); }
}
// permute tail: external rounds [r0..3], then 22 internal, then 4 terminal.
__device__ __forceinline__ void permute_tail(u64 *s, int r0){
    #pragma unroll 1
    for (int r=r0;r<4;r++){
        #pragma unroll
        for (int i=0;i<12;i++) s[i]=gf_x7(gf_add(s[i],D_INIT_EXT_RC[r][i]));
        ext_lin(s,nullptr);
    }
    #pragma unroll 1
    for (int r=0;r<22;r++){ s[0]=gf_x7(gf_add(s[0],D_INTERNAL_RC[r])); int_lin(s,0); }
    #pragma unroll 1
    for (int r=0;r<4;r++){
        #pragma unroll
        for (int i=0;i<12;i++) s[i]=gf_x7(gf_add(s[i],D_TERM_EXT_RC[r][i]));
        ext_lin(s,nullptr);
    }
}
__device__ __forceinline__ void permute(u64 *s){ ext_lin(s,nullptr); permute_tail(s,0); }

// ---- NONCE_DIR_SPARSE ----
// v=(a,-a,0), a=(17,-11,3,-4)  =>  M_E*v = 35*(e3-e7)  [verified: col.cpp]
// so after the initial M_E only lanes 3 and 7 vary per nonce; the other TEN first-round
// S-boxes are block-constant and are evaluated once per workgroup into LDS.
__device__ __constant__ int  D_VDIR[8]   = {17,-11,3,-4,-17,11,-3,4};
__device__ __constant__ u64  D_MECOL3[12]= {2,2,6,4,1,1,3,2,1,1,3,2};
__device__ __constant__ u64  D_MECOL7[12]= {1,1,3,2,2,2,6,4,1,1,3,2};
#define QBASE 0x40000000ULL
__device__ __forceinline__ u64 gf_sub(u64 a, u64 b){ u64 d=a-b; if(a<b) d+=P; if(d>=P) d-=P; return d; }
// small-constant multiply (columns are only 1,2,3,4,6)
__device__ __forceinline__ u64 gf_smul(u64 y, u64 c){
    u64 y2=gf_add(y,y);
    switch((int)c){ case 1:return y; case 2:return y2; case 3:return gf_add(y2,y);
                    case 4:return gf_add(y2,y2); case 6:return gf_add(gf_add(y2,y2),y2); }
    return gf_mul(y,c);
}
// the felts this nonce walk absorbs (base + idx*v), for the reference path
// Production nonce walk (matches cuda-miner/mine.cuh qpow_nonce_dir_felts exactly).
// The group is carried in felts 1,2,3 as 22+22+20 bits; those are CONSTANT within a block,
// so the sparsity in `local` is preserved while the nonce space stays a full 64 bits.
__host__ __device__ __forceinline__ void nd_felts(u64 group, u32 local, u32 f[8]){
    const u32 base=0x40000000u, mask=0x003fffffu;
    const u32 g0=(u32)group&mask, g1=(u32)(group>>22)&mask, g2=(u32)(group>>44);
    f[0]=base+17u*local;      f[1]=base+g0-11u*local;
    f[2]=base+g1+ 3u*local;   f[3]=base+g2- 4u*local;
    f[4]=base   -17u*local;   f[5]=base   +11u*local;
    f[6]=base   - 3u*local;   f[7]=base   + 4u*local;
}
// low 32 nonce bytes = the 8 felts, little-endian (matches qpow_nonce_dir_result_low)
__host__ __device__ __forceinline__ void nd_low_bytes(u64 group, u32 local, unsigned char out[32]){
    u32 f[8]; nd_felts(group,local,f);
    for (int i=0;i<8;i++){ out[4*i+0]=(unsigned char)f[i];        out[4*i+1]=(unsigned char)(f[i]>>8);
                           out[4*i+2]=(unsigned char)(f[i]>>16);  out[4*i+3]=(unsigned char)(f[i]>>24); }
}
// block precompute: 12 partial M_E outputs + the two varying S-box bases
__device__ __forceinline__ void sparse_pre(const u64 *ms, u64 group, u64 *pre){
    u64 z[12];
    #pragma unroll
    for (int i=0;i<12;i++) z[i]=ms[i];
    u32 f[8]; nd_felts(group,0u,f);                       // local = 0
    #pragma unroll
    for (int i=0;i<8;i++) z[i]=gf_add(z[i],(u64)f[i]);
    ext_lin(z,nullptr);
    pre[12]=gf_add(z[3],D_INIT_EXT_RC[0][3]);
    pre[13]=gf_add(z[7],D_INIT_EXT_RC[0][7]);
    u64 t[12];
    #pragma unroll
    for (int i=0;i<12;i++) t[i]=(i==3||i==7)?0ULL:gf_x7(gf_add(z[i],D_INIT_EXT_RC[0][i]));
    ext_lin(t,nullptr);
    #pragma unroll
    for (int i=0;i<12;i++) pre[i]=t[i];
}
// reconstruct the post-round-0 state for one nonce from the block precompute
__device__ __forceinline__ void sparse_state(const u64 *pre, u32 local, u64 *s){
    u64 d  = gf_mul(35ULL, (u64)local);   // M_E*v = 35*(e3-e7) per unit local
    u64 y3 = gf_x7(gf_add(pre[12],d));
    u64 y7 = gf_x7(gf_sub(pre[13],d));
    #pragma unroll
    for (int i=0;i<12;i++)
        s[i]=gf_add(pre[i], gf_add(gf_smul(y3,D_MECOL3[i]), gf_smul(y7,D_MECOL7[i])));
}
__device__ __forceinline__ void absorb(u64 *s, const unsigned char *b){
    #pragma unroll
    for (int i=0;i<8;i++){ int k=i*4;
        u64 f=(u64)b[k]|((u64)b[k+1]<<8)|((u64)b[k+2]<<16)|((u64)b[k+3]<<24);
        s[i]=gf_add(s[i],f); }
}
__device__ __forceinline__ void squeeze(const u64 *s, unsigned char *o){
    #pragma unroll
    for (int i=0;i<4;i++){ u64 v=s[i]; if(v>=P) v-=P;
        #pragma unroll
        for (int j=0;j<8;j++) o[i*8+j]=(unsigned char)(v>>(8*j)); }
}
// header[32] + nonce[64] -> hash[64]
__global__ void k_hash(const unsigned char *hdr, const unsigned char *non, unsigned char *out, int n){
    int t = blockIdx.x*blockDim.x + threadIdx.x; if (t>=n) return;
    u64 s[12];
    #pragma unroll
    for (int i=0;i<12;i++) s[i]=0;
    absorb(s, hdr+32*t);      permute(s);
    absorb(s, non+64*t);      permute(s);          // nonce high 32B
    absorb(s, non+64*t+32);   permute(s);          // nonce low 32B  (perm 3)
    s[0]=gf_add(s[0],1ULL); s[1]=gf_add(s[1],1ULL);
    permute(s);                                     // perm 4
    squeeze(s, out+64*t);
    permute(s);                                     // perm 5
    squeeze(s, out+64*t+32);
}


// ---- mining path: from a midstate, 2 permutations per nonce ----
__global__ void k_mine(const u64* __restrict__ ms, u64 base, u32 per_thread,
                       const u64* __restrict__ tgt, unsigned* __restrict__ found){
    u64 tid = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    u64 n0  = base + tid*per_thread;
    for (u32 j=0;j<per_thread;j++){
        u64 idx = n0 + j;
        u64 s[12];
        #pragma unroll
        for (int i=0;i<12;i++) s[i]=ms[i];
        // absorb the low nonce: 8 LE-u32 felts, idx varies the least-significant felt
        s[0]=gf_add(s[0],(u64)(u32)idx);
        s[1]=gf_add(s[1],(u64)(u32)(idx>>32));
        permute(s);
        s[0]=gf_add(s[0],1ULL); s[1]=gf_add(s[1],1ULL);
        permute(s);
        u64 h0=s[0]; if(h0>=P) h0-=P;
        if (h0 < tgt[0]) { unsigned k=atomicAdd(found,1u); (void)k; }
    }
}


struct FoundR { unsigned n; u64 group[8]; u32 local[8]; };
__device__ __forceinline__ u64 bswap64d(u64 x){
    return ((x&0xffull)<<56)|((x&0xff00ull)<<40)|((x&0xff0000ull)<<24)|((x&0xff000000ull)<<8)|
           ((x>>8)&0xff000000ull)|((x>>24)&0xff0000ull)|((x>>40)&0xff00ull)|((x>>56)&0xffull);
}
// strict big-endian U256 compare of the first squeeze against the target. 1=hash>target, 2=hash<target, 0=tie
__device__ __forceinline__ int cmp256d(const u64 *st, const unsigned char *target){
    #pragma unroll
    for (int i=0;i<4;i++){
        u64 v=st[i]; if(v>=P) v-=P;
        u64 hv=bswap64d(v);
        u64 tv=0; for(int j=0;j<8;j++) tv=(tv<<8)|(u64)target[8*i+j];
        if (hv!=tv) return hv>tv?1:2;
    }
    return 0;
}
// production mining kernel: ONE GROUP PER BLOCK, sparse first round from LDS
__global__ void k_mine_sparse(const u64* __restrict__ ms, u64 group_base, u32 per_thread,
                              const unsigned char* __restrict__ target, FoundR* __restrict__ res){
    __shared__ u64 pre[14];
    const u64 group = group_base + (u64)blockIdx.x;
    if (threadIdx.x==0) sparse_pre(ms,group,pre);
    __syncthreads();
    for (u32 j=0;j<per_thread;j++){
        u32 local = threadIdx.x*per_thread + j;
        u64 s[12];
        sparse_state(pre,local,s);
        permute_tail(s,1);
        s[0]=gf_add(s[0],1ULL); s[1]=gf_add(s[1],1ULL);
        permute(s);
        if (cmp256d(s,target)==2){                       // strictly below target
            unsigned k=atomicAdd(&res->n,1u);
            if(k<8){ res->group[k]=group; res->local[k]=local; }
        }
    }
}
// GATE 1: sparse path == straightforward path for the same (group,local)
__global__ void k_sparse_check(const u64* __restrict__ ms, u64 group_base, u32 n, unsigned* __restrict__ bad){
    __shared__ u64 pre[14];
    const u64 group = group_base + (u64)blockIdx.x;
    if (threadIdx.x==0) sparse_pre(ms,group,pre);
    __syncthreads();
    u32 local = threadIdx.x;
    if ((u64)blockIdx.x*blockDim.x+threadIdx.x >= n) return;
    u64 r[12]; u32 f[8];
    #pragma unroll
    for (int i=0;i<12;i++) r[i]=ms[i];
    nd_felts(group,local,f);
    #pragma unroll
    for (int i=0;i<8;i++) r[i]=gf_add(r[i],(u64)f[i]);
    permute(r);
    u64 q[12]; sparse_state(pre,local,q); permute_tail(q,1);
    #pragma unroll
    for (int i=0;i<12;i++){ u64 a=r[i],b=q[i]; if(a>=P)a-=P; if(b>=P)b-=P; if(a!=b){ atomicAdd(bad,1u); break; } }
}
// GATE 2 (share validity): reconstruct the nonce bytes, re-hash through the KAT-verified
// full-hash path, and confirm it equals the mining path. This is what proves a submitted
// share will validate at the pool.
__global__ void k_share_check(const unsigned char* __restrict__ hdr, const unsigned char* __restrict__ nhigh,
                              const u64* __restrict__ ms, u64 group_base, u32 n, unsigned* __restrict__ bad){
    __shared__ u64 pre[14];
    const u64 group = group_base + (u64)blockIdx.x;
    if (threadIdx.x==0) sparse_pre(ms,group,pre);
    __syncthreads();
    u32 local = threadIdx.x;
    if ((u64)blockIdx.x*blockDim.x+threadIdx.x >= n) return;
    // mining path -> first squeeze
    u64 q[12]; sparse_state(pre,local,q); permute_tail(q,1);
    q[0]=gf_add(q[0],1ULL); q[1]=gf_add(q[1],1ULL); permute(q);
    unsigned char mine32[32]; squeeze(q,mine32);
    // full path from the reconstructed 64-byte nonce
    unsigned char low[32]; nd_low_bytes(group,local,low);
    u64 r[12];
    #pragma unroll
    for (int i=0;i<12;i++) r[i]=0;
    absorb(r,hdr);   permute(r);
    absorb(r,nhigh); permute(r);
    absorb(r,low);   permute(r);
    r[0]=gf_add(r[0],1ULL); r[1]=gf_add(r[1],1ULL); permute(r);
    unsigned char ref32[32]; squeeze(r,ref32);
    #pragma unroll
    for (int i=0;i<32;i++) if(mine32[i]!=ref32[i]){ atomicAdd(bad,1u); break; }
}
// real midstate: header + high nonce, two permutations
__global__ void k_midstate(const unsigned char* __restrict__ hdr, const unsigned char* __restrict__ nhigh, u64* out){
    if (threadIdx.x||blockIdx.x) return;
    u64 s[12];
    #pragma unroll
    for (int i=0;i<12;i++) s[i]=0;
    absorb(s,hdr);   permute(s);
    absorb(s,nhigh); permute(s);
    #pragma unroll
    for (int i=0;i<12;i++){ u64 v=s[i]; if(v>=P) v-=P; out[i]=v; }
}

// ---------------- host ----------------
#include "constants_hip.h"
static void sleep_ms(int ms){ struct timespec t; t.tv_sec=ms/1000; t.tv_nsec=(long)(ms%1000)*1000000L; nanosleep(&t,0); }
#define QPOW_HIP_VERSION "0.1.0"
#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){ printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} }while(0)

static int hexval(char c){ return c>='0'&&c<='9'?c-'0':(c>='a'&&c<='f'?c-'a'+10:(c>='A'&&c<='F'?c-'A'+10:-1)); }
static bool unhex(const std::string&s, unsigned char*o, int n){
    if((int)s.size()<n*2) return false;
    for(int i=0;i<n;i++){ int a=hexval(s[2*i]),b=hexval(s[2*i+1]); if(a<0||b<0) return false; o[i]=(unsigned char)(a*16+b);} return true;
}
#include "pool.inc"


// ---- preflight: fail loudly and usefully instead of cryptically ----
// The commonest failure is a ROCm major-version mismatch between this binary and the
// installed runtime. A missing runtime cannot be caught here at all (the dynamic loader
// aborts before main), so the README documents that case.
static void arch_name(const hipDeviceProp_t& pr, char* out, size_t n){
#if HIP_VERSION_MAJOR >= 6
    snprintf(out,n,"%s",pr.gcnArchName);          // ROCm 6: "gfx1100"
#else
    snprintf(out,n,"gfx%d",pr.gcnArch);           // ROCm 5: integer
#endif
}
static int preflight(bool quiet){
    int built_major = HIP_VERSION_MAJOR;
    int rt=0; hipError_t e1=hipRuntimeGetVersion(&rt);
    int rt_major = (e1==hipSuccess)? rt/10000000 : -1;     // 50731921 -> 5, 60443484 -> 6
    int n=0; hipError_t e=hipGetDeviceCount(&n);
    if(e!=hipSuccess || n<1){
        fprintf(stderr,
          "ERROR: no AMD GPU is visible to ROCm (%s).%c%c"
          "  Checklist:%c"
          "   1. ROCm runtime installed?   Ubuntu: apt install rocm-hip-runtime%c"
          "                                AMD:    amdgpu-install --usecase=rocm%c"
          "   2. /dev/kfd present?         ls -l /dev/kfd   (absent => ROCm cannot open the GPU)%c"
          "      In a container you must pass:  --device=/dev/kfd --device=/dev/dri%c"
          "                                     --group-add video --group-add render%c"
          "   3. User in the render/video groups?   id | grep -E \"render|video\"%c"
          "   4. Supported GPU?  This build targets gfx1100 (RDNA3: RX 7900 XT/XTX/GRE).%c",
          hipGetErrorString(e),10,10,10,10,10,10,10,10,10,10);
        return 1;
    }
    hipDeviceProp_t pr; hipGetDeviceProperties(&pr,0);
    char arch[64]; arch_name(pr,arch,sizeof arch);
    if(rt_major>0 && rt_major!=built_major){
        fprintf(stderr,
          "%c*** ROCm VERSION MISMATCH ***%c"
          "  This binary was built for ROCm %d.x, but the installed runtime is ROCm %d.x.%c"
          "  It may refuse to load kernels, or run well below full speed.%c"
          "  Download the matching build instead:%c"
          "     ROCm 5.x  ->  qpow-hip-linux-x86_64-rocm5%c"
          "     ROCm 6.x  ->  qpow-hip-linux-x86_64-rocm6%c%c",
          10,10,built_major,rt_major,10,10,10,10,10,10);
    }
    if(!quiet){
        printf("qpow-hip %s | built for ROCm %d.%d | runtime ROCm %d.x | device: %s (%s)%c",
               QPOW_HIP_VERSION, HIP_VERSION_MAJOR, HIP_VERSION_MINOR, rt_major, pr.name, arch, 10);
    }
    // gfx1100 is what the kernel is compiled for; anything else will not have a code object
    if(!strstr(arch,"gfx1100"))
        fprintf(stderr,"WARNING: this build targets gfx1100 (RDNA3); your device reports %s — kernels may fail to load.%c",arch,10);
    return 0;
}

int main(int argc,char**argv){
    const char* mode = argc>1?argv[1]:"verify";
    if(preflight(!strcmp(mode,"version"))) return 2;
    HC(hipMemcpyToSymbol(HIP_SYMBOL(D_INTERNAL_RC), H_INTERNAL_RC, sizeof(H_INTERNAL_RC)));
    HC(hipMemcpyToSymbol(HIP_SYMBOL(D_MAT_DIAG),    H_MAT_DIAG,    sizeof(H_MAT_DIAG)));
    HC(hipMemcpyToSymbol(HIP_SYMBOL(D_INIT_EXT_RC), H_INIT_EXT_RC, sizeof(H_INIT_EXT_RC)));
    HC(hipMemcpyToSymbol(HIP_SYMBOL(D_TERM_EXT_RC), H_TERM_EXT_RC, sizeof(H_TERM_EXT_RC)));

    if(!strcmp(mode,"verify")){
        // stdin: "header_hex nonce_hex hash_hex" per line
        std::vector<unsigned char> H,N,W; int n=0; char line[512];
        while(fgets(line,sizeof line,stdin)){
            std::string a,b,c; { char x[200],y[200],z[200];
              if(sscanf(line,"%199s %199s %199s",x,y,z)!=3) continue; a=x;b=y;c=z; }
            unsigned char hb[32],nb[64],wb[64];
            if(!unhex(a,hb,32)||!unhex(b,nb,64)||!unhex(c,wb,64)) continue;
            H.insert(H.end(),hb,hb+32); N.insert(N.end(),nb,nb+64); W.insert(W.end(),wb,wb+64); n++;
        }
        if(!n){ printf("no vectors on stdin\n"); return 1; }
        unsigned char *dH,*dN,*dO; 
        HC(hipMalloc(&dH,32*n)); HC(hipMalloc(&dN,64*n)); HC(hipMalloc(&dO,64*n));
        HC(hipMemcpy(dH,H.data(),32*n,hipMemcpyHostToDevice));
        HC(hipMemcpy(dN,N.data(),64*n,hipMemcpyHostToDevice));
        int tpb=64, blocks=(n+tpb-1)/tpb;
        hipLaunchKernelGGL(k_hash, dim3(blocks), dim3(tpb), 0, 0, dH,dN,dO,n);
        HC(hipDeviceSynchronize());
        std::vector<unsigned char> O(64*n); HC(hipMemcpy(O.data(),dO,64*n,hipMemcpyDeviceToHost));
        int okF=0, okR=0;
        for(int i=0;i<n;i++){
            if(!memcmp(&O[64*i],&W[64*i],64)) okF++;
            unsigned char rev[64]; for(int j=0;j<64;j++) rev[j]=O[64*i+63-j];
            if(!memcmp(rev,&W[64*i],64)) okR++;
        }
        printf("VERIFY: %d/%d passed (forward)  %d/%d passed (byte-reversed)\n", okF,n,okR,n);
        return (okF==n||okR==n)?0:1;
    }
    if(!strcmp(mode,"bench")){
        double secs = argc>2?atof(argv[2]):8.0;
        u32 per_thread = argc>3?(u32)atoi(argv[3]):16;
        int tpb = argc>4?atoi(argv[4]):256;
        u64 hms[12]; for(int i2=0;i2<12;i2++) hms[i2]=0x0123456789abcdefULL*(i2+1);
        u64 htg[4]={0x0000000100000000ULL,0,0,0};
        u64 *dms,*dtg; unsigned *dfound;
        HC(hipMalloc(&dms,sizeof hms)); HC(hipMalloc(&dtg,sizeof htg)); HC(hipMalloc(&dfound,4));
        HC(hipMemcpy(dms,hms,sizeof hms,hipMemcpyHostToDevice));
        HC(hipMemcpy(dtg,htg,sizeof htg,hipMemcpyHostToDevice));
        HC(hipMemset(dfound,0,4));
        hipDeviceProp_t pr; HC(hipGetDeviceProperties(&pr,0));
        int blocks = pr.multiProcessorCount * 8;
        unsigned long long per_launch = (unsigned long long)blocks*tpb*per_thread;
        hipLaunchKernelGGL(k_mine,dim3(blocks),dim3(tpb),0,0,dms,0,per_thread,dtg,dfound);
        HC(hipDeviceSynchronize());
        double t0=(double)clock()/CLOCKS_PER_SEC; unsigned long long total=0, base=0;
        while(((double)clock()/CLOCKS_PER_SEC - t0) < secs){
            hipLaunchKernelGGL(k_mine,dim3(blocks),dim3(tpb),0,0,dms,base,per_thread,dtg,dfound);
            HC(hipDeviceSynchronize());
            base += per_launch; total += per_launch;
        }
        double el=(double)clock()/CLOCKS_PER_SEC - t0;
        printf("bench: %.2f MH/s  (%llu hashes, %.2fs, blocks=%d tpb=%d per_thread=%u)%c", total/el/1e6,total,el,blocks,tpb,per_thread,10);
        return 0;
    }
    // ---- real-job helpers ----
    if(!strcmp(mode,"sparsecheck")||!strcmp(mode,"sharecheck")||!strcmp(mode,"benchsparse")){
        unsigned char hdr[32], nhigh[32];
        for(int i2=0;i2<32;i2++){ hdr[i2]=(unsigned char)(0x11+i2); nhigh[i2]=(unsigned char)(0xA0^i2); }
        unsigned char *dh,*dnh,*dtg; u64 *dms;
        HC(hipMalloc(&dh,32)); HC(hipMalloc(&dnh,32)); HC(hipMalloc(&dms,12*sizeof(u64))); HC(hipMalloc(&dtg,32));
        HC(hipMemcpy(dh,hdr,32,hipMemcpyHostToDevice)); HC(hipMemcpy(dnh,nhigh,32,hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_midstate,dim3(1),dim3(1),0,0,dh,dnh,dms); HC(hipDeviceSynchronize());
        if(!strcmp(mode,"sparsecheck")||!strcmp(mode,"sharecheck")){
            u32 n = argc>2?(u32)atoi(argv[2]):262144; int tpb=256; int blocks=(int)((n+tpb-1)/tpb);
            unsigned *dbad; HC(hipMalloc(&dbad,4)); HC(hipMemset(dbad,0,4));
            if(!strcmp(mode,"sparsecheck"))
                 hipLaunchKernelGGL(k_sparse_check,dim3(blocks),dim3(tpb),0,0,dms,0ULL,n,dbad);
            else hipLaunchKernelGGL(k_share_check,dim3(blocks),dim3(tpb),0,0,dh,dnh,dms,0ULL,n,dbad);
            HC(hipDeviceSynchronize());
            unsigned bad=0; HC(hipMemcpy(&bad,dbad,4,hipMemcpyDeviceToHost));
            printf("%s: %u/%u match (%u wrong)%c", mode, n-bad, n, bad, 10);
            return bad?1:0;
        }
        double secs=argc>2?atof(argv[2]):8.0; u32 pt=argc>3?(u32)atoi(argv[3]):32; int tpb=argc>4?atoi(argv[4]):512;
        unsigned char tg[32]; memset(tg,0,32); tg[0]=0x00; tg[1]=0x00; tg[2]=0x01;
        HC(hipMemcpy(dtg,tg,32,hipMemcpyHostToDevice));
        FoundR* dres; HC(hipMalloc(&dres,sizeof(FoundR))); HC(hipMemset(dres,0,sizeof(FoundR)));
        hipDeviceProp_t pr; HC(hipGetDeviceProperties(&pr,0));
        int blocks=pr.multiProcessorCount*8;
        unsigned long long per_launch=(unsigned long long)blocks*tpb*pt, total=0, gb=0;
        hipLaunchKernelGGL(k_mine_sparse,dim3(blocks),dim3(tpb),0,0,dms,0ULL,pt,dtg,dres); HC(hipDeviceSynchronize());
        double t0=(double)clock()/CLOCKS_PER_SEC;
        while(((double)clock()/CLOCKS_PER_SEC-t0)<secs){
            hipLaunchKernelGGL(k_mine_sparse,dim3(blocks),dim3(tpb),0,0,dms,gb,pt,dtg,dres);
            HC(hipDeviceSynchronize()); gb+=blocks; total+=per_launch; }
        double el=(double)clock()/CLOCKS_PER_SEC-t0;
        printf("benchsparse: %.2f MH/s (%llu hashes, %.2fs)%c", total/el/1e6, total, el, 10);
        return 0;
    }
    if(!strcmp(mode,"pool")){
        if(argc<5){ printf("usage: %s pool <url> <payout> <worker> [per_thread] [tpb]%c",argv[0],10); return 1; }
        u32 pt = argc>5?(u32)atoi(argv[5]):32; int tpb = argc>6?atoi(argv[6]):512;
        return run_pool(argv[2],argv[3],argv[4],pt,tpb);
    }
    if(!strcmp(mode,"version")){
        hipDeviceProp_t pr; int n=0; hipGetDeviceCount(&n);
        if(n>0){ hipGetDeviceProperties(&pr,0);
            char a2[64]; arch_name(pr,a2,sizeof a2);
            printf("qpow-hip %s (built %s, HIP %d.%d) device: %s %s%c", QPOW_HIP_VERSION, __DATE__, HIP_VERSION_MAJOR, HIP_VERSION_MINOR, pr.name, a2, 10); }
        else printf("qpow-hip %s (built %s) — no AMD device visible%c", QPOW_HIP_VERSION, __DATE__, 10);
        return 0;
    }
    printf("usage: %s verify < vectors\n", argv[0]); return 1;
}
