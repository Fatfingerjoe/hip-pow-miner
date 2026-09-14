#include <hip/hip_runtime.h>
#include <cstdio>
typedef unsigned long long u64; typedef unsigned __int128 u128; typedef unsigned u32;
#define P 0xFFFFFFFF00000001ULL
__host__ __device__ u64 rd(u128 x){
  u64 lo=(u64)x,hi=(u64)(x>>64); u32 hh=(u32)(hi>>32),hl=(u32)hi;
  u64 t=lo-(u64)hh; if(lo<(u64)hh) t-=0xFFFFFFFFULL;
  u64 b=((u64)hl<<32)-(u64)hl;
  u64 r=t+b; if(r<t) r+=0xFFFFFFFFULL; if(r>=P) r-=P; return r;
}
__global__ void k(const u64* x, u64* acc_lo, u64* acc_hi, u64* red){
  u128 S=(u128)x[0]+(u128)x[1]+(u128)x[2]+(u128)x[3];
  u128 t=(u128)x[1];
  u128 v=S+(u128)x[0]+t+t;
  acc_lo[0]=(u64)v; acc_hi[0]=(u64)(v>>64); red[0]=rd(v);
  // also probe rd() alone on a fixed large value
  u128 fixed=((u128)7<<64)|0x123456789abcdefULL;
  red[1]=rd(fixed);
}
int main(){
  u64 h[4]={0x8000000000000000ULL,0x9000000000000000ULL,0xA000000000000000ULL,0xB000000000000000ULL};
  u128 S=(u128)h[0]+(u128)h[1]+(u128)h[2]+(u128)h[3];
  u128 t=(u128)h[1];
  u128 v=S+(u128)h[0]+t+t;
  u128 fixed=((u128)7<<64)|0x123456789abcdefULL;
  u64 c_lo=(u64)v, c_hi=(u64)(v>>64), c_red=rd(v), c_fix=rd(fixed);
  u64 *d,*al,*ah,*rr; hipMalloc(&d,32);hipMalloc(&al,8);hipMalloc(&ah,8);hipMalloc(&rr,16);
  hipMemcpy(d,h,32,hipMemcpyHostToDevice);
  hipLaunchKernelGGL(k,dim3(1),dim3(1),0,0,d,al,ah,rr);
  hipDeviceSynchronize();
  u64 g_lo,g_hi,g[2];
  hipMemcpy(&g_lo,al,8,hipMemcpyDeviceToHost);
  hipMemcpy(&g_hi,ah,8,hipMemcpyDeviceToHost);
  hipMemcpy(g,rr,16,hipMemcpyDeviceToHost);
  printf("accumulation lo : cpu=%016llx gpu=%016llx  %s\n",c_lo,g_lo,c_lo==g_lo?"ok":"*** DIFFER ***");
  printf("accumulation hi : cpu=%016llx gpu=%016llx  %s\n",c_hi,g_hi,c_hi==g_hi?"ok":"*** DIFFER ***");
  printf("rd(accumulated) : cpu=%016llx gpu=%016llx  %s\n",c_red,g[0],c_red==g[0]?"ok":"*** DIFFER ***");
  printf("rd(fixed 7:...) : cpu=%016llx gpu=%016llx  %s\n",c_fix,g[1],c_fix==g[1]?"ok":"*** DIFFER ***");
  return 0;
}
