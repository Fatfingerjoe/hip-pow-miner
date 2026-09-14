// QPoW Poseidon2-Goldilocks for AMD RDNA3 (gfx1100) via HIP.
// Milestone 1: CORRECTNESS. Plain __uint128_t arithmetic, one nonce per thread.
// Validated against big_vectors.txt (the same 8,000-vector KAT the CUDA miner uses).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
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
    if (r >= P) r -= P;
    return r;
}
__device__ __forceinline__ u64 gf_add(u64 a, u64 b){ return gf_red((__uint128_t)a + (__uint128_t)b); }
__device__ __forceinline__ u64 gf_mul(u64 a, u64 b){ return gf_red((__uint128_t)a * b); }
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
    for (int i=0;i<4;i++) sum[i]=gf_red((__uint128_t)s[i]+(__uint128_t)s[4+i]+(__uint128_t)s[8+i]);
    #pragma unroll
    for (int i=0;i<12;i++){ u64 z=gf_add(s[i],sum[i&3]); if(rc) z=gf_add(z,rc[i]); s[i]=z; }
}
// internal: s[i] = s[i]*MAT_DIAG[i] + sum + (rc0 on lane 0)
__device__ __forceinline__ void int_lin(u64 *s, u64 rc0){
    u64 sum=s[0];
    #pragma unroll
    for (int i=1;i<12;i++) sum=gf_add(sum,s[i]);
    #pragma unroll
    for (int i=0;i<12;i++){ u64 v=gf_add(gf_mul(s[i],D_MAT_DIAG[i]),sum); if(i==0) v=gf_add(v,rc0); s[i]=v; }
}
__device__ __forceinline__ void permute(u64 *s){
    ext_lin(s,nullptr);                                     // initial M_E, no RC
    #pragma unroll 1
    for (int r=0;r<4;r++){                                  // 4 initial external
        #pragma unroll
        for (int i=0;i<12;i++) s[i]=gf_x7(gf_add(s[i],D_INIT_EXT_RC[r][i]));
        ext_lin(s,nullptr);
    }
    #pragma unroll 1
    for (int r=0;r<22;r++){ s[0]=gf_x7(gf_add(s[0],D_INTERNAL_RC[r])); int_lin(s,0); }   // 22 internal
    #pragma unroll 1
    for (int r=0;r<4;r++){                                  // 4 terminal external
        #pragma unroll
        for (int i=0;i<12;i++) s[i]=gf_x7(gf_add(s[i],D_TERM_EXT_RC[r][i]));
        ext_lin(s,nullptr);
    }
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

// ---------------- host ----------------
#include "constants_hip.h"
#define HC(x) do{ hipError_t e=(x); if(e!=hipSuccess){ printf("HIP ERR %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} }while(0)

static int hexval(char c){ return c>='0'&&c<='9'?c-'0':(c>='a'&&c<='f'?c-'a'+10:(c>='A'&&c<='F'?c-'A'+10:-1)); }
static bool unhex(const std::string&s, unsigned char*o, int n){
    if((int)s.size()<n*2) return false;
    for(int i=0;i<n;i++){ int a=hexval(s[2*i]),b=hexval(s[2*i+1]); if(a<0||b<0) return false; o[i]=(unsigned char)(a*16+b);} return true;
}
int main(int argc,char**argv){
    const char* mode = argc>1?argv[1]:"verify";
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
    printf("usage: %s verify < vectors\n", argv[0]); return 1;
}
