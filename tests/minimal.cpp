// Minimal reproducer: how many terms in one __uint128_t expression before AMDGCN miscompiles?
#include <hip/hip_runtime.h>
#include <cstdio>
typedef unsigned long long u64; typedef unsigned __int128 u128;
__host__ __device__ u64 t2(u64 a,u64 b){ u128 x=(u128)a+(u128)b; return (u64)(x>>64); }
__host__ __device__ u64 t3(u64 a,u64 b,u64 c){ u128 x=(u128)a+(u128)b+(u128)c; return (u64)(x>>64); }
__host__ __device__ u64 t4(u64 a,u64 b,u64 c,u64 d){ u128 x=(u128)a+(u128)b+(u128)c+(u128)d; return (u64)(x>>64); }
__host__ __device__ u64 t4step(u64 a,u64 b,u64 c,u64 d){ u128 x=(u128)a; x+=(u128)b; x+=(u128)c; x+=(u128)d; return (u64)(x>>64); }
__global__ void k(u64*o,u64 a,u64 b,u64 c,u64 d){ o[0]=t2(a,b); o[1]=t3(a,b,c); o[2]=t4(a,b,c,d); o[3]=t4step(a,b,c,d); }
int main(){
  u64 a=0xFFFFFFFFFFFFFFFFULL,b=0xFFFFFFFFFFFFFFFFULL,c=0xFFFFFFFFFFFFFFFFULL,d=0xFFFFFFFFFFFFFFFFULL;
  u64 cpu[4]={t2(a,b),t3(a,b,c),t4(a,b,c,d),t4step(a,b,c,d)};
  u64 *dv,gpu[4]; hipMalloc(&dv,32);
  hipLaunchKernelGGL(k,dim3(1),dim3(1),0,0,dv,a,b,c,d); hipDeviceSynchronize();
  hipMemcpy(gpu,dv,32,hipMemcpyDeviceToHost);
  const char* nm[4]={"2-term  a+b","3-term  a+b+c","4-term  a+b+c+d","4-step  x+=a;x+=b;..."};
  // exact expected high words for all-ones inputs
  u64 want[4]={1,2,3,3};
  for(int i=0;i<4;i++)
    printf("%-24s cpu_hi=%llu gpu_hi=%llu expect=%llu  %s\n", nm[i],cpu[i],gpu[i],want[i],
           (cpu[i]==want[i]&&gpu[i]==want[i])?"ok":"*** WRONG ***");
  return 0;
}
