// Does the GPU compute the SAME thing as the CPU for the mat4 u128 expression?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
typedef unsigned long long u64; typedef unsigned __int128 u128; typedef unsigned u32;
#define P 0xFFFFFFFF00000001ULL

__host__ __device__ static u64 rd(u128 x){
  u64 lo=(u64)x,hi=(u64)(x>>64); u32 hh=(u32)(hi>>32),hl=(u32)hi;
  u64 t=lo-(u64)hh; if(lo<(u64)hh) t-=0xFFFFFFFFULL;
  u64 b=((u64)hl<<32)-(u64)hl;
  u64 r=t+b; if(r<t) r+=0xFFFFFFFFULL; if(r>=P) r-=P; return r;
}
__host__ __device__ static u64 ad(u64 a,u64 b){ return rd((u128)a+(u128)b); }
// the two forms
__host__ __device__ static void mat4_base(u64*x){u64 s=ad(ad(x[0],x[1]),ad(x[2],x[3])),o[4];
  for(int i=0;i<4;i++){u64 t=x[(i+1)&3]; o[i]=ad(ad(s,x[i]),ad(t,t));} for(int i=0;i<4;i++)x[i]=o[i];}
__host__ __device__ static void mat4_u128(u64*x){u128 S=(u128)x[0]+(u128)x[1]+(u128)x[2]+(u128)x[3];u64 o[4];
  for(int i=0;i<4;i++){u128 t=(u128)x[(i+1)&3]; o[i]=rd(S+(u128)x[i]+t+t);} for(int i=0;i<4;i++)x[i]=o[i];}


// Variant A: explicit cast on every term, fully parenthesized to prevent sub-u64 folding.
__host__ __device__ static void mat4_safeA(u64*x){
  u128 S=(u128)x[0]+(u128)x[1]+(u128)x[2]+(u128)x[3];
  u64 o[4];
  for(int i=0;i<4;i++){
    u128 t=(u128)x[(i+1)&3];
    o[i]=rd(((S+(u128)x[i])+t)+t);
  }
  for(int i=0;i<4;i++)x[i]=o[i];
}

// Variant B: accumulate the small terms in u128 separately, then add to S.
__host__ __device__ static void mat4_safeB(u64*x){
  u128 S=(u128)x[0]+(u128)x[1]+(u128)x[2]+(u128)x[3];
  u64 o[4];
  for(int i=0;i<4;i++){
    u128 Y=(u128)x[i];
    u128 t=(u128)x[(i+1)&3];
    o[i]=rd(S+Y+t+t);
  }
  for(int i=0;i<4;i++)x[i]=o[i];
}

// Variant C: keep S as two-limb carry-save (hi,lo) using only u64, then combine at the end.
__host__ __device__ static void mat4_safeC(u64*x){
  // S = sum(x) as exact 66-bit value: hi is carry count, lo is modular sum.
  u64 s=0, c=0;
  for(int i=0;i<4;i++){ u64 sum=s+x[i]; c += (sum<s)?1:0; s=sum; }
  u128 S = (u128)c + (u128)s;  // c is tiny, no overflow here
  u64 o[4];
  for(int i=0;i<4;i++){
    u128 t=(u128)x[(i+1)&3];
    o[i]=rd(S+(u128)x[i]+t+t);
  }
  for(int i=0;i<4;i++)x[i]=o[i];
}

__global__ void k(const u64* in, u64* outB, u64* outU, u64* outA, u64* outB2, u64* outC, int n){
  int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n) return;
  u64 a[4],b[4],sa[4],sb[4],sc[4];
  for(int i=0;i<4;i++){a[i]=b[i]=sa[i]=sb[i]=sc[i]=in[4*t+i];}
  mat4_base(a); mat4_u128(b); mat4_safeA(sa); mat4_safeB(sb); mat4_safeC(sc);
  for(int i=0;i<4;i++){outB[4*t+i]=a[i]; outU[4*t+i]=b[i]; outA[4*t+i]=sa[i]; outB2[4*t+i]=sb[i]; outC[4*t+i]=sc[i];}
}
int main(){
  const int N=6;
  u64 h[N*4]={1,2,3,4,  P-1,P-2,0,5,  0xFFFFFFFFFFFFFFFFULL,1,2,3,
              0x8000000000000000ULL,0x9000000000000000ULL,0xA000000000000000ULL,0xB000000000000000ULL,
              P-1,P-1,P-1,P-1,  0xDEADBEEFCAFEBABEULL,0x1234567890ABCDEFULL,P+7,0xFFFFFFFF00000000ULL};
  // CPU reference
  u64 cB[N*4], cU[N*4], cA[N*4], cB2[N*4], cC[N*4];
  for(int t=0;t<N;t++){u64 a[4],b[4],sa[4],sb[4],sc[4];
    for(int i=0;i<4;i++){a[i]=b[i]=sa[i]=sb[i]=sc[i]=h[4*t+i];}
    mat4_base(a); mat4_u128(b); mat4_safeA(sa); mat4_safeB(sb); mat4_safeC(sc);
    for(int i=0;i<4;i++){cB[4*t+i]=a[i]; cU[4*t+i]=b[i]; cA[4*t+i]=sa[i]; cB2[4*t+i]=sb[i]; cC[4*t+i]=sc[i];}}
  u64 *d,*oB,*oU,*oA,*oB2,*oC;
  hipMalloc(&d,sizeof h); hipMalloc(&oB,sizeof h); hipMalloc(&oU,sizeof h); hipMalloc(&oA,sizeof h); hipMalloc(&oB2,sizeof h); hipMalloc(&oC,sizeof h);
  hipMemcpy(d,h,sizeof h,hipMemcpyHostToDevice);
  hipLaunchKernelGGL(k,dim3(1),dim3(N),0,0,d,oB,oU,oA,oB2,oC,N);
  hipDeviceSynchronize();
  u64 gB[N*4],gU[N*4],gA[N*4],gB2[N*4],gC[N*4];
  hipMemcpy(gB,oB,sizeof h,hipMemcpyDeviceToHost);
  hipMemcpy(gU,oU,sizeof h,hipMemcpyDeviceToHost);
  hipMemcpy(gA,oA,sizeof h,hipMemcpyDeviceToHost);
  hipMemcpy(gB2,oB2,sizeof h,hipMemcpyDeviceToHost);
  hipMemcpy(gC,oC,sizeof h,hipMemcpyDeviceToHost);
  int bad=0;
  for(int t=0;t<N;t++)for(int i=0;i<4;i++){
    int k4=4*t+i;
    if(gB[k4]!=cB[k4]){printf("case %d lane %d BASE  cpu=%016llx gpu=%016llx  *** GPU DIFFERS ***\n",t,i,cB[k4],gB[k4]);bad++;}
    if(gU[k4]!=cU[k4]){printf("case %d lane %d U128  cpu=%016llx gpu=%016llx  *** GPU DIFFERS ***\n",t,i,cU[k4],gU[k4]);bad++;}
    if(gA[k4]!=cA[k4]){printf("case %d lane %d SAFEA cpu=%016llx gpu=%016llx  *** GPU DIFFERS ***\n",t,i,cA[k4],gA[k4]);bad++;}
    if(gB2[k4]!=cB2[k4]){printf("case %d lane %d SAFEB cpu=%016llx gpu=%016llx  *** GPU DIFFERS ***\n",t,i,cB2[k4],gB2[k4]);bad++;}
    if(gC[k4]!=cC[k4]){printf("case %d lane %d SAFEC cpu=%016llx gpu=%016llx  *** GPU DIFFERS ***\n",t,i,cC[k4],gC[k4]);bad++;}
    if(cB[k4]!=cU[k4]){printf("case %d lane %d CPU base!=u128 %016llx vs %016llx\n",t,i,cB[k4],cU[k4]);bad++;}
  }
  printf("%s\n", bad? "MISMATCHES FOUND":"GPU and CPU agree on ALL forms, and the forms agree");
  return 0;
}
