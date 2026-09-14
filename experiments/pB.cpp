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
__device__ __forceinline__ void sparse_felts(u64 idx, u64 f[8]){
    #pragma unroll
    for (int i=0;i<8;i++){
        long long d=(long long)D_VDIR[i]*(long long)idx;
        f[i] = (u64)((long long)QBASE + d);
    }
}
// block precompute: 12 partial M_E outputs + the two varying S-box bases
__device__ __forceinline__ void sparse_pre(const u64 *ms, u64 *pre){
    u64 z[12];
    #pragma unroll
    for (int i=0;i<12;i++) z[i]=ms[i];
    #pragma unroll
    for (int i=0;i<8;i++) z[i]=gf_add(z[i],QBASE);
    ext_lin(z,nullptr);                                  // initial M_E
    pre[12]=gf_add(z[3],D_INIT_EXT_RC[0][3]);            // varying lane 3 base
    pre[13]=gf_add(z[7],D_INIT_EXT_RC[0][7]);            // varying lane 7 base
    u64 t[12];
    #pragma unroll
    for (int i=0;i<12;i++)
        t[i] = (i==3||i==7) ? 0ULL : gf_x7(gf_add(z[i],D_INIT_EXT_RC[0][i]));
    ext_lin(t,nullptr);                                  // M_E of the constant part
    #pragma unroll
    for (int i=0;i<12;i++) pre[i]=t[i];
}
// reconstruct the post-round-0 state for one nonce from the block precompute
__device__ __forceinline__ void sparse_state(const u64 *pre, u64 idx, u64 *s){
    u64 d  = gf_mul(35ULL, idx % P);
    u64 y3 = gf_x7(gf_add(pre[12],d));
    u64 y7 = gf_x7(gf_sub(pre[13],d));
    #pragma unroll 1
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


// sparse mining kernel: block-precompute 10 of 12 first-round S-boxes, resume at round 1
__global__ void k_mine_sparse(const u64* __restrict__ ms, u64 base, u32 per_thread,
                              const u64* __restrict__ tgt, unsigned* __restrict__ found){
    __shared__ u64 pre[14];
    if (threadIdx.x==0) sparse_pre(ms,pre);
    __syncthreads();
    u64 tid=(u64)blockIdx.x*blockDim.x+threadIdx.x;
    u64 n0 = base + tid*per_thread;
    for (u32 j=0;j<per_thread;j++){
        u64 s[12];
        sparse_state(pre, n0+j, s);      // state after initial M_E + round 0
        permute_tail(s,1);               // rest of permutation 3
        s[0]=gf_add(s[0],1ULL); s[1]=gf_add(s[1],1ULL);
        permute(s);                      // permutation 4
        u64 h0=s[0]; if(h0>=P) h0-=P;
        if (h0 < tgt[0]) { unsigned k=atomicAdd(found,1u); (void)k; }
    }
}
// GATE: sparse path must equal the straightforward path for the SAME nonce
__global__ void k_sparse_check(const u64* __restrict__ ms, u64 base, u32 n, unsigned* __restrict__ bad){
    __shared__ u64 pre[14];
    if (threadIdx.x==0) sparse_pre(ms,pre);
    __syncthreads();
    u64 t=(u64)blockIdx.x*blockDim.x+threadIdx.x; if(t>=n) return;
    u64 idx=base+t;
    // reference: absorb the same felts, run the full permutation
    u64 r[12]; u64 f[8];
    #pragma unroll
    for (int i=0;i<12;i++) r[i]=ms[i];
    sparse_felts(idx,f);
    #pragma unroll
    for (int i=0;i<8;i++) r[i]=gf_add(r[i],f[i]);
    permute(r);
    // sparse
    u64 q[12];
    sparse_state(pre,idx,q);
    permute_tail(q,1);
    #pragma unroll
    for (int i=0;i<12;i++){ u64 a=r[i],b=q[i]; if(a>=P)a-=P; if(b>=P)b-=P; if(a!=b){ atomicAdd(bad,1u); break; } }
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
    if(!strcmp(mode,"sparsecheck")){
        u32 n = argc>2?(u32)atoi(argv[2]):262144;
        u64 hms[12]; for(int i2=0;i2<12;i2++) hms[i2]=0x0123456789abcdefULL*(i2+1);
        u64 *dms; unsigned *dbad;
        HC(hipMalloc(&dms,sizeof hms)); HC(hipMalloc(&dbad,4));
        HC(hipMemcpy(dms,hms,sizeof hms,hipMemcpyHostToDevice)); HC(hipMemset(dbad,0,4));
        int tpb=256, blocks=(int)((n+tpb-1)/tpb);
        hipLaunchKernelGGL(k_sparse_check,dim3(blocks),dim3(tpb),0,0,dms,0ULL,n,dbad);
        HC(hipDeviceSynchronize());
        unsigned bad=0; HC(hipMemcpy(&bad,dbad,4,hipMemcpyDeviceToHost));
        printf("MINESPARSE: %u/%u match (%u wrong)%c", n-bad, n, bad, 10);
        return bad?1:0;
    }
    if(!strcmp(mode,"benchsparse")){
        double secs = argc>2?atof(argv[2]):8.0;
        u32 per_thread = argc>3?(u32)atoi(argv[3]):32;
        int tpb = argc>4?atoi(argv[4]):512;
        u64 hms[12]; for(int i2=0;i2<12;i2++) hms[i2]=0x0123456789abcdefULL*(i2+1);
        u64 htg[4]={0x0000000100000000ULL,0,0,0};
        u64 *dms,*dtg; unsigned *dfound;
        HC(hipMalloc(&dms,sizeof hms)); HC(hipMalloc(&dtg,sizeof htg)); HC(hipMalloc(&dfound,4));
        HC(hipMemcpy(dms,hms,sizeof hms,hipMemcpyHostToDevice));
        HC(hipMemcpy(dtg,htg,sizeof htg,hipMemcpyHostToDevice)); HC(hipMemset(dfound,0,4));
        hipDeviceProp_t pr; HC(hipGetDeviceProperties(&pr,0));
        int blocks = pr.multiProcessorCount * 8;
        unsigned long long per_launch=(unsigned long long)blocks*tpb*per_thread, total=0, base=0;
        hipLaunchKernelGGL(k_mine_sparse,dim3(blocks),dim3(tpb),0,0,dms,0,per_thread,dtg,dfound);
        HC(hipDeviceSynchronize());
        double t0=(double)clock()/CLOCKS_PER_SEC;
        while(((double)clock()/CLOCKS_PER_SEC - t0) < secs){
            hipLaunchKernelGGL(k_mine_sparse,dim3(blocks),dim3(tpb),0,0,dms,base,per_thread,dtg,dfound);
            HC(hipDeviceSynchronize()); base+=per_launch; total+=per_launch; }
        double el=(double)clock()/CLOCKS_PER_SEC - t0;
        printf("benchsparse: %.2f MH/s (%llu hashes, %.2fs)%c", total/el/1e6, total, el, 10);
        return 0;
    }
    printf("usage: %s verify < vectors\n", argv[0]); return 1;
}
