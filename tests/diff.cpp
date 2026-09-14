#include <stdio.h>
#include <string.h>
typedef unsigned long long u64; typedef unsigned __int128 u128; typedef unsigned u32;
#define P 0xFFFFFFFF00000001ULL
#include "constants_hip.h"
static u64 ad(u64 a,u64 b){u64 s=a+b; if(s<a) s-=P; return s;}          // LAZY
static u64 rd(u128 x){u64 lo=(u64)x,hi=(u64)(x>>64);u32 hh=(u32)(hi>>32),hl=(u32)hi;
  u64 a=lo-(u64)hh; if(lo<(u64)hh) a+=P; u64 b=((u64)hl<<32)-(u64)hl; return ad(a,b);}
// two mat4 implementations
static void mat4_base(u64*x){u64 s=ad(ad(x[0],x[1]),ad(x[2],x[3])),o[4];
  for(int i=0;i<4;i++){u64 t=x[(i+1)&3]; o[i]=ad(ad(s,x[i]),ad(t,t));} memcpy(x,o,sizeof o);}
static void mat4_u128(u64*x){u128 S=(u128)x[0]+(u128)x[1]+(u128)x[2]+(u128)x[3];u64 o[4];
  for(int i=0;i<4;i++){u128 t=(u128)x[(i+1)&3]; o[i]=rd(S+(u128)x[i]+t+t);} memcpy(x,o,sizeof o);}
int main(){
  // differences mod p?  compare canonical forms
  u64 seeds[4][4]={{1,2,3,4},{P-1,P-2,0,5},{0xFFFFFFFFFFFFFFFFULL,1,2,3},{0x8000000000000000ULL,0x9000000000000000ULL,0xA000000000000000ULL,0xB000000000000000ULL}};
  for(int k=0;k<4;k++){
    u64 a[4],b[4]; memcpy(a,seeds[k],sizeof a); memcpy(b,seeds[k],sizeof b);
    mat4_base(a); mat4_u128(b);
    printf("case %d:\n",k);
    for(int i=0;i<4;i++){
      u64 ca=a[i]%P, cb=b[i]%P;
      printf("   lane%d base=%016llx u128=%016llx  canon %s\n", i,a[i],b[i], ca==cb?"MATCH":"*** DIFFER ***");
    }
  }
  return 0;
}
