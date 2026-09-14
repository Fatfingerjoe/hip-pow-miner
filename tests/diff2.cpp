#include <stdio.h>
#include <string.h>
typedef unsigned long long u64; typedef unsigned __int128 u128; typedef unsigned u32;
#define P 0xFFFFFFFF00000001ULL
#include "constants_hip.h"
static u64 rd(u128 x){
  u64 lo=(u64)x,hi=(u64)(x>>64); u32 hh=(u32)(hi>>32),hl=(u32)hi;
  u64 t=lo-(u64)hh; if(lo<(u64)hh) t-=0xFFFFFFFFULL;
  u64 b=((u64)hl<<32)-(u64)hl;
  u64 r=t+b; if(r<t) r+=0xFFFFFFFFULL; if(r>=P) r-=P; return r;
}
static u64 ad(u64 a,u64 b){ return rd((u128)a+(u128)b); }
static u64 ref(u128 x){ return (u64)(x % (u128)P); }
static void mat4_base(u64*x){u64 s=ad(ad(x[0],x[1]),ad(x[2],x[3])),o[4];
  for(int i=0;i<4;i++){u64 t=x[(i+1)&3]; o[i]=ad(ad(s,x[i]),ad(t,t));} memcpy(x,o,sizeof o);}
static void mat4_u128(u64*x){u128 S=(u128)x[0]+(u128)x[1]+(u128)x[2]+(u128)x[3];u64 o[4];
  for(int i=0;i<4;i++){u128 t=(u128)x[(i+1)&3]; o[i]=rd(S+(u128)x[i]+t+t);} memcpy(x,o,sizeof o);}
static void mat4_exact(u64*x){u128 S=(u128)x[0]+(u128)x[1]+(u128)x[2]+(u128)x[3];u64 o[4];
  for(int i=0;i<4;i++){u128 t=(u128)x[(i+1)&3]; o[i]=ref(S+(u128)x[i]+t+t);} memcpy(x,o,sizeof o);}
int main(){
  u64 seeds[5][4]={{1,2,3,4},{P-1,P-2,0,5},{0xFFFFFFFFFFFFFFFFULL,1,2,3},
                   {0x8000000000000000ULL,0x9000000000000000ULL,0xA000000000000000ULL,0xB000000000000000ULL},
                   {P-1,P-1,P-1,P-1}};
  int bad=0;
  for(int k=0;k<5;k++){
    u64 a[4],b[4],c[4]; memcpy(a,seeds[k],32);memcpy(b,seeds[k],32);memcpy(c,seeds[k],32);
    mat4_base(a); mat4_u128(b); mat4_exact(c);
    for(int i=0;i<4;i++){
      int okb = (a[i]==c[i]), oku = (b[i]==c[i]);
      if(!okb||!oku){ bad++;
        printf("case %d lane %d: base=%016llx u128=%016llx EXACT=%016llx  base%s u128%s\n",
               k,i,a[i],b[i],c[i], okb?"=OK":"=BAD", oku?"=OK":"=BAD"); }
    }
  }
  printf("mat4 comparison: %s\n", bad?"DIVERGENCE FOUND":"all agree with exact reference");
  return 0;
}
